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
#include <executorch/extension/module/module.h>

namespace executorch {
namespace extension {
namespace llm {

class Sampler;

namespace batching {

namespace cache = ::executorch::extension::llm::cache;

// A session's cache sequence and the sampler its generation draws from.
struct SessionInfo {
  std::int32_t seq;
  std::unique_ptr<Sampler> sampler;
};

// One forward's inputs, flattened across the batch. Entry i of `tokens` and of
// `positions` names the same token, which is how the cache pairs them.
struct Step {
  // Signed to match the model's token input, not Token.
  std::vector<std::int64_t> tokens;
  std::vector<std::int64_t> positions;
  // Per input: the logits row it draws from, or -1 when its prediction is
  // discarded. An input of any width contributes one, since only its last row
  // predicts a token the session does not hold.
  std::vector<int> logit_indices;
};

// Flatten the batch, truncate whatever it reopens, and declare it to the cache.
// A per-sequence cursor carries the batch's own writes, so consecutive chunks
// of one prompt abut and only the first can reopen committed ground. Every
// input is checked before any is truncated, so a refusal leaves the cache
// untouched.
//
// nullopt = an input names an unknown session, starts past the end of its
// sequence, carries it past `max_session_tokens`, reopens from the start, or
// the cache turned the declaration down. Width is not checked; execute()
// slices a step wider than the method takes.
std::optional<Step> build_step(
    cache::BatchControl& ctl,
    const BatchInput& batch,
    const std::unordered_map<SessionId, SessionInfo>& sessions,
    int max_session_tokens);

class ModuleExecutor : public Executor {
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
  // nullptr = unusable limits, no published KV layout, a method spanning
  // several backends, or no such cache for the backend it names. A method that
  // will not load is reported by initialize().
  static std::unique_ptr<ModuleExecutor> create(
      std::unique_ptr<Module> module,
      int max_sessions,
      int max_session_tokens,
      int kv_dtype,
      int initial_capacity = -1,
      std::string cache_kind = "cell",
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
  ModuleExecutor(
      std::unique_ptr<Module> module,
      std::shared_ptr<cache::CacheBase> cache,
      std::unique_ptr<cache::CacheSession> session,
      int max_sessions,
      int max_session_tokens,
      std::string backend_id,
      std::string method,
      std::int32_t vocab_size,
      int max_step_tokens);

  // Draw the token an input produced from its row of `logits`, which the
  // session's sampler consumes in place.
  std::optional<Token>
  sample_row(::executorch::aten::Tensor& logits, int row, SessionId session);

  // Ordered so the module dies first, releasing the delegate that resolved the
  // cache before the registry entry naming it goes.
  std::unique_ptr<cache::CacheSession> session_;
  std::shared_ptr<cache::CacheBase> cache_;
  std::unique_ptr<Module> module_;
  cache::BatchControl* ctl_;
  int max_sessions_;
  int max_session_tokens_;
  std::string backend_id_;
  std::string method_;
  // The method's logits width, so a sampler can be built by its policy.
  std::int32_t vocab_size_;
  int max_step_tokens_;

  SessionId next_session_ = 1; // never reused, unlike the cache's sequence ids
  std::unordered_map<SessionId, SessionInfo> sessions_;
};

} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch
