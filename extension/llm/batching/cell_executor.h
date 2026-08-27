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
namespace batching {

namespace cache = ::executorch::extension::llm::cache;

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
// sequence, carries it past `max_session_tokens`, reopens from the start, or
// the cache turned the declaration down.
std::optional<Step> build_step(
    cache::BatchControl& ctl,
    const BatchInput& batch,
    const std::unordered_map<SessionId, std::int32_t>& seqs,
    int max_session_tokens);

class CellExecutor : public Executor {
 public:
  struct Config {
    cache::CacheConfig cache;
    // Sessions open at once, and the cells each may hold. A short table refuses
    // the whole batch and takes every session in it down, so create() rejects
    // limits the table cannot honor and open_session() holds the count --
    // exhaustion is kept unreachable rather than handled.
    int max_sessions = 0;
    int max_session_tokens = 0;
    std::string backend_id;
    std::string cache_kind = "cell";
    // Backend-load option through which the delegate finds the cache.
    std::string cache_key_option = "cache_key";
    std::string method = "forward";
  };

  ~CellExecutor() override;

  // Takes the program loaded but its method not. The cache is sized from a
  // layout the program publishes, so the program has to be readable first.
  //
  // The method is left for the first execute(). A delegate may bind per-thread
  // state as it initializes, which happens during that load, and construction
  // is the only entry point that does not run on the engine thread. So a
  // method that will not load fails a batch rather than this call.
  //
  // nullptr = limits the cell table cannot honor, or an unregistered
  // (backend_id, cache_kind).
  static std::unique_ptr<CellExecutor> create(
      std::unique_ptr<Module> module,
      Config config);

  std::optional<SessionId> open_session() override;
  void close_session(SessionId session) override;
  void set_sampling(
      SessionId session,
      const SamplingParams& params,
      std::optional<std::uint64_t> seed) override;
  bool execute(const BatchInput& batch, BatchOutput& out) override;

 private:
  struct Sampling {
    SamplingParams params;
    std::uint64_t seed;
  };

  CellExecutor(
      Config config,
      std::unique_ptr<Module> module,
      std::shared_ptr<cache::CacheBase> cache,
      std::unique_ptr<cache::CacheSession> session);

  // Load the method, naming the cache to the backend, on first use. Runs on
  // the execute() thread so what the delegate binds there is reachable later.
  bool ensure_method_loaded();

  // Draw from one logits row. Randomness comes from the session's seed and the
  // position the token will occupy, so it does not follow how batches formed.
  std::optional<Token> sample_row(
      const ::executorch::aten::Tensor& logits,
      int row,
      SessionId session,
      std::int64_t position) const;

  Config config_;
  // Ordered so the module dies first, releasing the delegate that resolved the
  // cache before the registry entry naming it goes.
  std::unique_ptr<cache::CacheSession> session_;
  std::shared_ptr<cache::CacheBase> cache_;
  std::unique_ptr<Module> module_;
  cache::BatchControl* ctl_;

  bool method_loaded_ = false;
  SessionId next_session_ = 1; // never reused, unlike the cache's sequence ids
  std::unordered_map<SessionId, std::int32_t> seqs_;
  std::unordered_map<SessionId, Sampling> sampling_;
};

} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch
