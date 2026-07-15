/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Pure warm-resume prefill decision for the model worker (no model/session/ET
// dependency, so it is unit-testable in isolation). Given a named session's
// resident token ids (exactly the tokens currently in its KV/recurrent state),
// the new request's prompt token ids, and whether the session is dirty, decide
// whether to reset + full-prefill or to keep the state and prefill only the
// suffix. The decision is exact-token (never string / retokenized text), so a
// kSuffix plan is always a correct continuation; anything uncertain falls back
// to kFull. See worker_loop.h for how the plan is executed.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace executorch {
namespace extension {
namespace llm {

struct PrefillPlan {
  enum Action {
    kFull, // reset + prefill the whole prompt
    kSuffix // keep state, prefill prompt_ids[suffix_start:] at pos>0
  } action;
  size_t suffix_start; // index in prompt_ids where prefill begins (0 for kFull)
  // Reported as session_reset_reason: "new" (no resident), "exact_prefix"
  // (suffix reuse), "dirty", "mismatch", "equal" (prompt == resident).
  const char* reason;
};

inline PrefillPlan plan_prefill(
    const std::vector<uint64_t>& resident,
    const std::vector<uint64_t>& prompt,
    bool dirty) {
  if (dirty) {
    return {PrefillPlan::kFull, 0, "dirty"};
  }
  if (resident.empty()) {
    return {PrefillPlan::kFull, 0, "new"};
  }
  if (prompt.size() < resident.size()) {
    return {PrefillPlan::kFull, 0, "mismatch"};
  }
  for (size_t i = 0; i < resident.size(); ++i) {
    if (prompt[i] != resident[i]) {
      return {PrefillPlan::kFull, 0, "mismatch"};
    }
  }
  if (prompt.size() == resident.size()) {
    // Prompt is exactly the resident state (no new tokens). The ideal would be
    // to skip prefill and decode straight from the session's pending token, but
    // the LLMSession API exposes no "is there a valid pending token?" query, so
    // we conservatively reset + full prefill rather than risk
    // prefill_tokens([]) or decoding from a stale/absent pending. Rare in
    // practice (a new turn adds tokens); a session pending-state query is a
    // possible later optimization.
    return {PrefillPlan::kFull, 0, "equal"};
  }
  return {PrefillPlan::kSuffix, resident.size(), "exact_prefix"};
}

} // namespace llm
} // namespace extension
} // namespace executorch
