/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// A batching Executor for a program the CUDA backend compiled into static
// graphs. The CUDA counterpart of ModuleExecutor: a session is one sequence of
// a cell-layout off-graph KV cache, and a batch is one or more forwards
// carrying every input's tokens on a single axis. Unlike ModuleExecutor it
// runs several methods, since a static graph serves one shape: static
// `forward_{N}` methods of exactly N tokens, each captured into a CUDA graph,
// and optionally one dynamic `forward_others` for anything wider. Each forward
// runs the method with the smallest width that holds it; a static method's
// extra rows are padding.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <executorch/backends/cuda/batching/step_plan.h>
#include <executorch/backends/cuda/runtime/cuda_kv_cache.h>
#include <executorch/extension/llm/batching/executor.h>
#include <executorch/extension/llm/batching/util/session_table.h>
#include <executorch/extension/llm/cache/cache.h>
#include <executorch/extension/llm/cache/cache_registry.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/compiler.h> // ET_EXPERIMENTAL

namespace executorch::backends::cuda::batching {

namespace llm_batching = ::executorch::extension::llm::batching;
namespace llm_cache = ::executorch::extension::llm::cache;

// The methods and metadata the program must export. Static methods are named
// kStaticMethodPrefix + N, for their exact token count N.
inline constexpr char kStaticMethodPrefix[] = "forward_";
inline constexpr char kDynamicMethod[] = "forward_others";
// The cell layout's pool size, which fixes the shape of its step buffers.
inline constexpr char kMaxCellsMethod[] = "get_offgraph_kv_max_cells";
// Optional: the fewest selected rows forward_others was exported for. Its
// selector is padded up to it. Defaults to 1.
inline constexpr char kMinSelectedRowsMethod[] = "get_min_prefill_chunk";
// Optional device samplers (extension/llm/batching/sampler.py). With them the
// forward methods may return device-resident logits: kSampleMethod takes
// (logits [R, V], params [R, 4] float32) and kArgmaxMethod (logits [R, V]),
// both returning Long [R], and only the tokens come back to the host.
inline constexpr char kSampleMethod[] = "sample";
inline constexpr char kArgmaxMethod[] = "sample_argmax";

// Process-wide CUDA backend options create() sets before the methods load.
struct CudaExecutorOptions {
  // The methods are compiled separately over the same weights; share them
  // rather than loading a copy per method.
  bool weight_sharing_across_methods = true;
  // Capture every static method into a CUDA graph. forward_others stays
  // eager: its width varies.
  bool cuda_graph_for_static_methods = true;
};

// How often each method ran, for reporting.
struct ForwardMethodCalls {
  std::string name;
  std::uint64_t calls;
};

class ET_EXPERIMENTAL CudaExecutor : public llm_batching::Executor {
 public:
  ~CudaExecutor() override;

  // Builds the cell cache from the layout the program publishes and sets the
  // CUDA backend options. The methods load in initialize(); the program must
  // be loaded and its methods must not be, since the delegate resolves the
  // cache while they load. `options` is applied process-wide, as backend
  // options are.
  //
  // `max_sessions` counts every resident session, clones included, and each
  // reserves `max_session_tokens` cells; together they must fit the program's
  // max_cells. `kv_dtype` is the ET ScalarType K/V is stored in; a negative
  // `initial_capacity` starts the pools at the cache's default.
  static ::executorch::runtime::Result<std::unique_ptr<CudaExecutor>> create(
      std::unique_ptr<::executorch::extension::Module> module,
      int max_sessions,
      int max_session_tokens,
      int kv_dtype,
      int initial_capacity = -1,
      CudaExecutorOptions options = {});

  // The widest method's step. A wider batch runs as several forwards.
  std::size_t preferred_batch_tokens() const override {
    return static_cast<std::size_t>(methods_.back().max_tokens);
  }

  // Loads every method here, so the delegate that resolves the cache binds on
  // the thread that runs it.
  bool initialize() override;

  std::optional<llm_batching::SessionId> open_session() override;
  void close_session(llm_batching::SessionId session) override;
  std::optional<llm_batching::SessionId> clone(
      llm_batching::SessionId source,
      llm_batching::Position upto) override;
  void set_sampling(
      llm_batching::SessionId session,
      const llm_batching::SamplingParams& params,
      std::optional<std::uint64_t> seed) override;
  bool execute(
      const llm_batching::BatchInput& batch,
      llm_batching::BatchOutput& out) override;

  // The cache's pool usage, for memory reporting.
  OffGraphKVMetrics kv_metrics() const;

  // Every method, narrowest first, with how many forwards it ran. Read it
  // between executes, not while one runs.
  std::vector<ForwardMethodCalls> method_calls() const;

  // Whether tokens are sampled on the device rather than on the host.
  bool samples_on_device() const {
    return has_sampler_;
  }

 private:
  CudaExecutor(
      std::unique_ptr<::executorch::extension::Module> module,
      std::shared_ptr<llm_cache::Cache> cache,
      int max_sessions,
      int max_session_tokens,
      std::string backend_id,
      std::int32_t vocab_size,
      std::vector<ForwardMethod> methods,
      int min_selected_rows,
      bool has_sampler,
      bool has_argmax);

  // Draws one token per row of `logits`, rows [0, sessions.size()), on the
  // device; returns them in row order.
  std::optional<std::vector<llm_batching::Token>> sample_on_device(
      const ::executorch::aten::Tensor& logits,
      const std::vector<llm_batching::SessionId>& sessions);

  // Ordered so the module dies first, releasing the delegates that resolved
  // the cache before the registry entry naming it goes.
  llm_cache::InstallGuard install_guard_;
  std::unique_ptr<::executorch::extension::Module> module_;
  llm_cache::BatchControl* const ctl_;
  const CudaKVCache* const kv_;
  std::string backend_id_;
  std::int32_t vocab_size_;
  // Sorted by max_tokens, ascending; the dynamic method, if any, is last.
  std::vector<ForwardMethod> methods_;
  std::vector<std::uint64_t> calls_;
  int min_selected_rows_;
  llm_batching::util::SessionTable sessions_;
  bool has_sampler_;
  bool has_argmax_;
  // Device staging for the sampler's params, grown to the widest row count.
  void* device_params_ = nullptr;
  std::size_t device_params_rows_ = 0;
};

} // namespace executorch::backends::cuda::batching
