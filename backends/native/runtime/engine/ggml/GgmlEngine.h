// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include <executorch/backends/native/runtime/engine/Engine.h>

namespace executorch::extension::llm::cache {
class Cache;
} // namespace executorch::extension::llm::cache

namespace ptn {

// CacheFactory backend id of the ggml engine's off-graph KV caches. A method
// that calls kvcache::update_and_attend compiles only in a context created
// with the kCacheKeyOption ("llm_cache_registry_key") option naming a cache
// built under this id: kind::kSingle for one sequence, or kind::kBatched /
// kind::kBatchedCell for many sequences over shared cells.
inline constexpr char kGgmlCacheBackendId[] = "NativeGgml";

// Registers the ggml cache builders with CacheFactory::global(). Idempotent.
void register_ggml_cache_builders();

enum class GgmlDevice {
  Cpu,
  // The GPU backend is whichever one the linked ggml variant registers (CUDA,
  // HIP, ...); device_index counts GPU devices only.
  Gpu,
};

struct GgmlEngineOptions {
  GgmlDevice device = GgmlDevice::Cpu;
  int device_index = 0;
  // Threads for the ggml CPU backend, which also runs ops the GPU backend
  // lacks. 0 keeps ggml's default of 4, not the host core count.
  int cpu_threads = 0;
  // Minimum node capacity of each ggml graph and of the backend scheduler;
  // a method whose lowerings need more nodes gets the room they need.
  size_t graph_size = 4096;
  // Graphs each executable keeps, keyed by the shapes of all inputs and, for
  // each attention mask that leaves only a key prefix open, that prefix
  // rounded up to 256 keys: a decode step crossing a 256-key boundary builds
  // a new graph. A call matching a cached key replays the graph with fresh
  // input bytes, skipping lowering. Entries share the context's compute
  // buffers, reserved at compile for each method's worst case, so a hit after
  // another graph ran is laid out in them again. 0 rebuilds the graph on every
  // call; at most 16.
  size_t max_cached_graphs = 1;
  // For testing: answers in place of the GPU whether it runs an attention
  // that reads a KV cache over F16 keys and values, so that a test on any
  // device can exercise the choice of F16 KV caches (see
  // f32_gpu_kv_cache_for_testing). It is asked only about attentions whose
  // shapes the lowering accepts.
  bool (*gpu_runs_f16_attention_for_testing)() = nullptr;
  // Prints each newly built graph's node count per ggml op to stderr, which
  // shows whether a lowering rewrite fired.
  bool print_graph_ops = false;
  // Prints, after each compile and when the context is destroyed, the
  // context's constant and compute buffer bytes and the device's free memory
  // to stderr.
  bool print_memory = false;
  // Testing only. Default (false): matrix multiplies with an F16 operand
  // accumulate in F16 on CUDA, as llama.cpp's cuBLAS GEMMs do. Set: they
  // accumulate in F32, at 2-3x prefill and 4x decode GEMM time plus a
  // transient F32 weight copy. Ops: F16 linear, matmul, mm, bmm and
  // convolution; CUDA only.
  bool cuda_f32_accumulation_for_testing = false;
  // ggml backends quantize the activations of a matmul with quantized
  // weights in their own block format, as llama.cpp does. False is a
  // debugging mode: weight-only quantized weights (int4 or GGUF) that a
  // linear or an expert gather reads load dequantized to F32, at 4-8x their
  // memory, and those matmuls run without activation quantization. Weights
  // of linears whose activations the model quantizes itself keep their
  // blocks, as do weights only embeddings read.
  bool allow_activation_quant_for_weight_only = true;
  // Testing only. Default (false): topk is llama.cpp's ggml_argsort_top_k, one
  // argsort that leaves a row holding NaN in an unspecified order.
  // Set: NaN ranks above every value, +inf included (below, for the smallest),
  // as eager does, at a NaN key map and a second argsort per topk.
  // Ops: topk.
  bool exact_topk_for_testing = false;
  // Testing only. Default (false): tanh GELU is ggml_gelu, as in llama.cpp;
  // ggml-cpu reads it from an F16 table, a few 1e-4 off eager in F32.
  // Set: the CPU builds tanh GELU from exact ops, x * (1 + tanh(c * (x +
  // 0.044715 x^3))) / 2, as eager does, at five more kernels per GELU.
  // Ops: gelu with approximate="tanh"; CPU only (GPU backends evaluate
  // ggml_gelu in F32 either way).
  bool exact_gelu_for_testing = false;
  // Testing only. Default (false): on CUDA and HIP, a mutable F32 buffer that
  // starts zeroed, that only row writes update and that only attention reads
  // is stored in F16, as llama.cpp stores KV caches. Those backends' flash
  // attention rounds F32 keys and values to F16 on every call, exactly as the
  // F16 rows written once round them, so results match while decoding moves
  // half the cache bytes; a cache an attention without a GPU kernel reads
  // stays F32. Set: KV caches stay F32. In-graph caches only: an off-graph
  // cache takes its storage from CacheConfig::kv_dtype. Ops: KV-cache buffers
  // read by scaled_dot_product_attention and the custom SDPA; CUDA and HIP
  // only.
  bool f32_gpu_kv_cache_for_testing = false;
  // Default (true): nodes that read only unquantized constants are computed
  // once, when the executable is compiled, and every call reads their
  // results. False (testing only): they are computed on every call.
  bool fold_constants = true;
};

class GgmlEngineContext;
struct GgmlKvCache;

class GgmlEngineHost final : public EngineHost {
 public:
  static std::unique_ptr<GgmlEngineHost> create(GgmlEngineOptions options = {});

  ~GgmlEngineHost() override;

  const std::string& name() const override;
  const std::string& device_name() const override;

  std::unique_ptr<EngineContext> create_context(
      std::shared_ptr<const Program> program,
      std::shared_ptr<const Package> package) override;

  // Accepts kCacheKeyOption, resolved through CacheRegistry::global(). Contexts
  // bound to one cache share its sequences: the caller must serialize them.
  std::unique_ptr<EngineContext> create_context_with_options(
      std::shared_ptr<const Program> program,
      std::shared_ptr<const Package> package,
      const EngineContextOptions& options) override;

  std::unique_ptr<GgmlEngineContext> create_ggml_context(
      std::shared_ptr<const Program> program,
      std::shared_ptr<const Package> package,
      const EngineContextOptions& options = {});

 private:
  struct Impl;

  explicit GgmlEngineHost(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

class GgmlEngineContext final : public EngineContext {
 public:
  ~GgmlEngineContext() override;

  size_t materialized_constant_bytes() const;
  // Constant bytes copied onto the heap on their way into ggml buffers, rather
  // than read in place from the package.
  size_t heap_copied_constant_bytes() const;
  // Bytes of F32 KV caches stored in F16 (see f32_gpu_kv_cache_for_testing).
  size_t f16_kv_cache_bytes() const;
  // Bytes of the compute buffers every executable of the context shares, and
  // how often they grew: once per method whose worst case needs more room.
  size_t compute_buffer_bytes() const;
  size_t compute_buffer_allocations() const;

 private:
  friend class GgmlEngineHost;

  struct Impl;

  GgmlEngineContext(
      std::shared_ptr<const Program> program,
      std::shared_ptr<const Package> package,
      GgmlEngineOptions options,
      std::shared_ptr<executorch::extension::llm::cache::Cache> cache,
      GgmlKvCache kv_cache);

  std::unique_ptr<EngineExecutable> compile_method(
      const Method& method) override;

  void print_memory(const std::string& when) const;

  std::unique_ptr<Impl> impl_;
};

} // namespace ptn
