/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// What every Executor over a multi-sequence cache does the same way, whatever
// runs its forwards: mapping sessions to cache sequences, packing a batch onto
// one token axis, and sampling each session's rows with its own policy.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include <executorch/extension/llm/batching/types.h>
#include <executorch/extension/llm/cache/cache.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/platform/compiler.h> // ET_EXPERIMENTAL

namespace executorch {
namespace extension {
namespace llm {

class Sampler;

namespace batching {
namespace util {

namespace cache = ::executorch::extension::llm::cache;

// A batch flattened onto one token axis, in input order.
struct ET_EXPERIMENTAL PackedStep {
  std::vector<std::int64_t> tokens;
  std::vector<std::int64_t> positions;
  std::vector<std::int32_t> seq_ids;
  // Per input: the token whose logits it wants, or -1 for none.
  std::vector<int> logit_indices;
};

// The logits rows a forward over tokens [offset, offset + length) must select:
// `selector` indexes that forward's tokens, `inputs` names the batch input
// each row belongs to. A forward with no wanted row still selects its last
// token, which nothing reads, since the program produces at least one row.
struct ET_EXPERIMENTAL SliceRows {
  std::vector<std::int64_t> selector;
  std::vector<std::size_t> inputs;
};

ET_EXPERIMENTAL SliceRows
select_rows(const PackedStep& step, int offset, int length);

// One row's policy for a sampler that runs on the device, laid out as the
// float32 [temperature, top_p, top_k, coin] row
// extension/llm/batching/sampler.py reads.
struct ET_EXPERIMENTAL DeviceSamplingRow {
  float temperature;
  float top_p;
  float top_k;
  float coin;
};

class ET_EXPERIMENTAL SessionTable {
 public:
  // `control` must outlive the table. `max_sessions` counts every resident
  // session, clones included; `max_session_tokens` bounds each one.
  SessionTable(
      cache::BatchControl& control,
      int max_sessions,
      int max_session_tokens,
      std::int32_t vocab_size);
  ~SessionTable();

  SessionTable(const SessionTable&) = delete;
  SessionTable& operator=(const SessionTable&) = delete;

  std::optional<SessionId> open();
  // Frees the session's cells and hands its sequence id back. Session ids
  // are never reused.
  void close(SessionId session);
  // A new session holding the source's [0, upto).
  std::optional<SessionId> clone(SessionId source, Position upto);
  void set_sampling(
      SessionId session,
      const SamplingParams& params,
      std::optional<std::uint64_t> seed);

  // Flattens the batch and truncates whatever it reopens. A per-sequence
  // cursor carries the batch's own writes, so consecutive chunks of one prompt
  // abut and only the first can reopen committed ground. Every input is
  // checked before any sequence is truncated, so a refusal leaves the cache
  // untouched. Declaring each forward's tokens is the caller's, per forward.
  ::executorch::runtime::Result<PackedStep> pack(const BatchInput& batch);

  // Draws the session's token from `row` of `logits`, [rows, vocab], which
  // the sampler reduces in place.
  std::optional<Token>
  sample(SessionId session, ::executorch::aten::Tensor& logits, int row);

  // The session's next row for a device sampler. Draws the coin from the same
  // seeded generator, in the same order, as the host sampler would, so a seed
  // reproduces a generation either way; a greedy row draws none, as on host.
  std::optional<DeviceSamplingRow> device_sampling(SessionId session);
  // Whether the session samples greedily (temperature 0).
  bool greedy(SessionId session) const;

  std::size_t size() const {
    return sessions_.size();
  }

 private:
  struct Session {
    std::int32_t seq_id;
    std::unique_ptr<Sampler> sampler;
    SamplingParams params;
    // The device path's generator: Sampler's xorshift state, seeded alike.
    std::uint64_t rng_state = 0;
  };

  std::optional<SessionId> publish(std::int32_t seq_id, Position position);

  cache::BatchControl& ctl_;
  const int max_sessions_;
  const int max_session_tokens_;
  const std::int32_t vocab_size_;
  SessionId next_session_ = 1;
  std::unordered_map<SessionId, Session> sessions_;
};

} // namespace util
} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch
