/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// An Executor over the cell KV cache. A session is one cache sequence, a batch
// is one forward carrying every input's tokens end to end on a single axis,
// and the cache's mask is what keeps the sequences apart.
//
// The backend id, cache kind, and option key are configuration, so any backend
// registering a cell cache is served.

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

// One forward's inputs, flattened across the batch. `tokens` and `positions`
// are filled in one pass so entry i of each names the same token, which is how
// the cache pairs them when it places cells and builds the mask.
struct Step {
  std::vector<Token> tokens;
  std::vector<std::int64_t> positions;
  // One entry per input: the logits row it draws from, or -1 when it produces
  // none. An input of any width contributes one, since only its last row
  // predicts a token the session does not already hold.
  std::vector<int> logit_indices;
};

// Flatten the batch, truncate whatever it reopens, and declare it to the cache.
// `ctl` does not move as inputs are laid down, so a per-sequence cursor carries
// the batch's own writes -- consecutive chunks of one prompt abut, and only the
// first can reopen committed ground.
//
// Every input is checked before any is truncated, so a batch refused on its
// contents leaves the cache untouched.
//
// nullopt = an input names a session not in `seqs`, starts past the end of its
// sequence, carries it past `max_session_tokens`, reopens from the start, the
// batch is wider than `max_step_tokens`, or the cache turned the declaration
// down.
std::optional<Step> build_step(
    cache::BatchControl& ctl,
    const BatchInput& batch,
    const std::unordered_map<SessionId, SessionInfo>& sessions,
    int max_session_tokens,
    int max_step_tokens);

class CellExecutor : public Executor {
 public:
  ~CellExecutor() override;

  // Builds the cell cache from the layout `module` publishes, binds it to the
  // backend, and loads the method. The program must be loaded; its method must
  // not be, since the delegate resolves the cache while that load runs.
  //
  // The table holds `max_sessions` sessions of `max_session_tokens` cells. A
  // short table refuses the whole batch and takes every session in it down, so
  // capacity is that product exactly and open_session() holds the count --
  // exhaustion is kept unreachable rather than handled.
  //
  // `kv_dtype` is the ET ScalarType the cache stores K/V in; a negative
  // `initial_capacity` leaves the pools to grow from their own default.
  //
  // The backend is read from the program: the executor drives one cache, so
  // the method's attention must be delegated to a single backend.
  //
  // nullptr = unusable limits, a program publishing no KV layout, a method
  // spanning several backends, or no cell cache registered for the one it
  // names.
  static std::unique_ptr<CellExecutor> create(
      std::unique_ptr<Module> module,
      int max_sessions,
      int max_session_tokens,
      int kv_dtype,
      int initial_capacity = -1,
      std::string method = "forward");

  // The widest step this method takes, read from the shape its token input was
  // traced at. A batch carrying more is refused when the input is resized, so
  // the caller bounds its batches by this.
  int max_step_tokens() const {
    return max_step_tokens_;
  }

  std::optional<SessionId> open_session() override;
  void close_session(SessionId session) override;
  void set_sampling(
      SessionId session,
      const SamplingParams& params,
      std::optional<std::uint64_t> seed) override;
  bool execute(const BatchInput& batch, BatchOutput& out) override;

 private:
  CellExecutor(
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
