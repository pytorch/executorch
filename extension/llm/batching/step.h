/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// The vocabulary of batched LLM serving: what a step is, how to execute one,
// and what running one produces. Shared by the scheduler that orders steps,
// the executor that runs them, and the runner that drives both.
//
// A Request is one step for one sequence -- a single decode token, or one
// prefill chunk -- not a whole generation.
//
// No tensors, no cache, no model, no locks. Where a step's KV lives is the
// cache's business, and how steps are ordered is the scheduler's.

#include <cstdint>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

namespace executorch {
namespace extension {
namespace llm {
namespace batching {

using Token = std::int64_t;
using RequestId = std::int64_t;
using SessionId = std::int64_t;
using Position = std::int32_t;

// --- How to execute a step -------------------------------------------------

// Carried per step, so a caller may change sampling between turns.
//
// Every field is a literal value; there is no "implementation default"
// sentinel. This differs from llm::SamplingConfig in
// extension/llm/runner/llm_session.h, where temperature == -1 means "let the
// implementation choose" and is validated as a distinct legal value. Anything
// bridging the two must resolve that sentinel first: passing -1 through here
// would be read as a negative temperature rather than a request for a default.
struct SamplingParams {
  // 0 = greedy. Higher is more random.
  float temperature = 0.0f;
  float top_p = 1.0f;
  std::int32_t top_k = 0; // 0 = disabled
  std::uint64_t seed = 0; // 0 = unset
};

// Which positions of a step the model must produce output for. A verify step
// needs every row -- it inspects the prediction at each drafted position --
// while a decode or prefill chunk only needs the last. A sampled step yields
// one token per row; an unsampled one yields one distribution per row.
enum class OutputRows {
  Last,
  All,
};

// How to execute a step. Promote to a variant if a second step kind ever needs
// different fields rather than different values.
struct RequestParams {
  Position position = 0;
  // Absent means: do not sample, return the raw distribution. Rejection
  // sampling and constrained decoding need the distribution; greedy
  // speculative verification does not, since argmax at every row is just
  // sampling at temperature zero.
  std::optional<SamplingParams> sampling;
  OutputRows output_rows = OutputRows::Last;
};

// --- What running a step produces ------------------------------------------

// Raw model output for a step whose Request asked not to sample.
struct LogitsBlock {
  std::vector<float> data;
  std::int32_t n_rows = 0;
  std::int32_t vocab = 0;
};
using LogitsPtr = std::shared_ptr<const LogitsBlock>;

// Which alternative the executor produces follows from whether the Request
// asked to sample. A sampled step yields one token per output row, so a greedy
// verify round returns the prediction at every drafted position; deciding how
// many to accept, and where the session therefore continues, belongs to
// whoever ran the batch. This never passes through the scheduler.
using ResponsePayload = std::variant<std::vector<Token>, LogitsPtr>;

// --- The step itself -------------------------------------------------------

// One step for one sequence. request_id names the submission and is always
// unique, so the same work submitted twice is two requests that both run.
// Only request_id, session_id, and tokens.size() are read by the scheduler;
// `params` travels through it to the executor, one way, untouched.
struct Request {
  RequestId request_id = 0;
  SessionId session_id = 0;
  std::vector<Token> tokens;
  RequestParams params;

  std::int32_t n_tokens() const {
    return static_cast<std::int32_t>(tokens.size());
  }

  // A one-token prefill is indistinguishable from a decode and is treated as
  // one: same single row of output, one decode slot.
  bool is_decode() const {
    return tokens.size() == 1;
  }
};

// One batch of steps for a single forward. Decodes first, then prefills, in
// admission order.
struct Batch {
  std::vector<Request> requests;

  bool empty() const {
    return requests.empty();
  }
  std::int32_t n_tokens() const {
    std::int32_t n = 0;
    for (const Request& r : requests) {
      n += r.n_tokens();
    }
    return n;
  }
};

} // namespace batching
} // namespace llm
} // namespace extension
} // namespace executorch
