/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Pure exact-token continuation decision. `resident` is the history against
// which the full prompt is compared. It can be committed KV/recurrent tokens,
// or logical history including a pending token if the caller's continuation
// path feeds that token before the suffix. Dirty or uncertain histories replay
// the full prompt; this helper neither inspects nor changes execution state.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <executorch/extension/llm/batching/types.h>
#include <executorch/runtime/platform/compiler.h>

namespace executorch {
namespace extension {
namespace llm {

struct ET_EXPERIMENTAL PrefillPlan {
  enum Action {
    kFull, // reset + prefill the whole prompt
    kSuffix // keep state, prefill prompt_ids[suffix_start:] at pos>0
  } action;
  size_t suffix_start; // index in prompt_ids where prefill begins (0 for kFull)
  // Reported as session_reset_reason: "new" (no resident), "exact_prefix"
  // (suffix reuse), "dirty", "mismatch", "equal" (prompt == resident),
  // "suffix_unavailable" (runtime cold fallback after optional reuse refusal).
  std::string reason;
};

ET_EXPERIMENTAL inline PrefillPlan plan_prefill_identity(
    const batching::PrefixIdentity& resident,
    const batching::PrefixIdentity& prompt,
    bool dirty) {
  if (dirty) {
    return {PrefillPlan::kFull, 0, "dirty"};
  }
  if (!resident.size()) {
    return {PrefillPlan::kFull, 0, "new"};
  }
  if (prompt.size() < resident.size()) {
    return {PrefillPlan::kFull, 0, "mismatch"};
  }
  if (batching::common_prefix(resident, prompt) != resident.size()) {
    return {PrefillPlan::kFull, 0, "mismatch"};
  }
  if (prompt.size() == resident.size()) {
    // Equal histories conservatively replay: this helper cannot establish that
    // decoding without a non-empty suffix is supported or safe for the caller.
    return {PrefillPlan::kFull, 0, "equal"};
  }
  return {PrefillPlan::kSuffix, resident.size(), "exact_prefix"};
}

ET_EXPERIMENTAL inline PrefillPlan plan_prefill(
    const std::vector<uint64_t>& resident,
    const std::vector<uint64_t>& prompt,
    bool dirty) {
  return plan_prefill_identity(
      batching::token_identity(
          std::make_shared<const std::vector<uint64_t>>(resident)),
      batching::token_identity(
          std::make_shared<const std::vector<uint64_t>>(prompt)),
      dirty);
}

} // namespace llm
} // namespace extension
} // namespace executorch
