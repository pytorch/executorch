/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// An Executor that runs an ExecuTorch Module over a registered KV cache, built
// from the layout the program publishes. A session is one cache sequence, a
// batch is one forward carrying every input's tokens on a single axis, and the
// cache's mask keeps the sequences apart.

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <executorch/extension/llm/batching/executor.h>
#include <executorch/extension/llm/cache/cache.h>
#include <executorch/extension/llm/cache/cache_registry.h>
#include <executorch/extension/llm/runner/model_metadata.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/compiler.h> // ET_EXPERIMENTAL

namespace executorch {
namespace extension {
namespace llm {

class Sampler;

namespace batching {

namespace cache = ::executorch::extension::llm::cache;

class ET_EXPERIMENTAL ModuleExecutor : public Executor {
 public:
  ~ModuleExecutor() override;

  // Builds the cache from the layout `module` publishes and pairs it with the
  // backend, which is read from the program -- so the method's attention must
  // be delegated to just one. The method itself loads in initialize(); the
  // program must be loaded and its method must not be, since the delegate
  // resolves the cache while that load runs.
  //
  // Capacity is `max_sessions` x `max_session_tokens` cells exactly, and
  // open_session() holds the count, so exhaustion is unreachable rather than
  // handled. `kv_dtype` is the ET ScalarType K/V is stored in; a negative
  // `initial_capacity` leaves the pools to grow from their own default.
  // `cache_kind` must name a builder that carries batch control -- a cache
  // serving one sequence cannot back a batch of them.
  //
  // Returns an error for unusable limits, no published KV layout, a method
  // spanning several backends, or no such cache for the backend it names. A
  // method that will not load is reported by initialize().
  static ::executorch::runtime::Result<std::unique_ptr<ModuleExecutor>> create(
      std::unique_ptr<Module> module,
      int max_sessions,
      int max_session_tokens,
      int kv_dtype,
      int initial_capacity = -1,
      std::string cache_kind = cache::kind::kBatchedCell,
      std::string method = "forward");

  // The widest step this method takes, from the shape its token input was
  // traced at. A wider batch is sliced; a narrower one leaves the forward
  // partly unused.
  std::size_t preferred_batch_tokens() const override {
    return static_cast<std::size_t>(max_step_tokens_);
  }

  // Loads the method here so the delegate that resolves the cache binds on the
  // thread that runs it.
  bool initialize() override;

  std::optional<SessionId> open_session() override;
  void close_session(SessionId session) override;
  void set_sampling(
      SessionId session,
      const SamplingParams& params,
      std::optional<std::uint64_t> seed) override;
  bool execute(const BatchInput& batch, BatchOutput& out) override;

 private:
  struct SessionState {
    std::int32_t seq_id;
    std::unique_ptr<Sampler> sampler;
  };

  struct Step {
    std::vector<std::int64_t> tokens;
    std::vector<std::int64_t> positions;
    std::vector<std::int32_t> seq_ids;
    std::vector<int> logit_indices;
  };

  ::executorch::runtime::Result<Step> build_step(const BatchInput& batch);

  ModuleExecutor(
      std::unique_ptr<Module> module,
      std::shared_ptr<cache::Cache> cache,
      int max_sessions,
      int max_session_tokens,
      std::string backend_id,
      std::string method,
      std::int32_t vocab_size,
      int max_step_tokens,
      LogitsToKeepMode logits_to_keep_mode);

  // Draw the token an input produced from its row of `logits`, which the
  // session's sampler consumes in place.
  std::optional<Token>
  sample_row(::executorch::aten::Tensor& logits, int row, SessionId session);

  // Ordered so the module dies first, releasing the delegate that resolved the
  // cache before the registry entry naming it goes.
  cache::InstallGuard install_guard_;
  std::unique_ptr<Module> module_;
  cache::BatchControl* const ctl_;
  int max_sessions_;
  int max_session_tokens_;
  std::string backend_id_;
  std::string method_;
  // The method's logits width, so a sampler can be built by its policy.
  std::int32_t vocab_size_;
  int max_step_tokens_;
  LogitsToKeepMode logits_to_keep_mode_;

  SessionId next_session_ = 1; // never reused, unlike the cache's sequence ids
  std::unordered_map<SessionId, SessionState> sessions_;
};

} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch
