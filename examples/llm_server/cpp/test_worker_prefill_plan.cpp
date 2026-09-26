/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Unit tests for plan_prefill() (warm-resume decision). No model/session/ET
// runtime dependency -- the header is pure, so this compiles and runs
// standalone. Self-contained assertions (no gtest) so it has no build deps.

#include <executorch/examples/llm_server/cpp/worker_prefill_plan.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using executorch::extension::llm::plan_prefill;
using executorch::extension::llm::PrefillPlan;

namespace {
int g_failures = 0;

void expect(
    const char* name,
    const PrefillPlan& p,
    PrefillPlan::Action action,
    size_t suffix_start,
    const char* reason) {
  bool ok = p.action == action && p.suffix_start == suffix_start &&
      std::strcmp(p.reason, reason) == 0;
  if (!ok) {
    ++g_failures;
    printf(
        "  [FAIL] %s: got action=%d suffix_start=%zu reason=%s\n",
        name,
        (int)p.action,
        p.suffix_start,
        p.reason);
  } else {
    printf("  [PASS] %s\n", name);
  }
}
} // namespace

int main() {
  using V = std::vector<uint64_t>;

  // First request: nothing resident -> full prefill, "new".
  expect(
      "new (resident empty)",
      plan_prefill(V{}, V{1, 2, 3}, false),
      PrefillPlan::kFull,
      0,
      "new");

  // Exact token extension -> prefill only the suffix.
  expect(
      "exact_prefix (suffix reuse)",
      plan_prefill(V{1, 2, 3}, V{1, 2, 3, 4, 5}, false),
      PrefillPlan::kSuffix,
      3,
      "exact_prefix");

  // Single-token extension still reuses.
  expect(
      "exact_prefix (one-token suffix)",
      plan_prefill(V{1, 2, 3}, V{1, 2, 3, 4}, false),
      PrefillPlan::kSuffix,
      3,
      "exact_prefix");

  // Divergent token -> mismatch, full reset.
  expect(
      "mismatch (divergent token)",
      plan_prefill(V{1, 2, 3}, V{1, 2, 9, 4}, false),
      PrefillPlan::kFull,
      0,
      "mismatch");

  // Prompt shorter than resident (rewind) -> mismatch, full reset.
  expect(
      "mismatch (prompt shorter)",
      plan_prefill(V{1, 2, 3}, V{1, 2}, false),
      PrefillPlan::kFull,
      0,
      "mismatch");

  // Dirty wins even over an otherwise-exact extension.
  expect(
      "dirty (overrides exact prefix)",
      plan_prefill(V{1, 2, 3}, V{1, 2, 3, 4}, true),
      PrefillPlan::kFull,
      0,
      "dirty");

  // Prompt identical to resident -> reset + full (no empty-suffix prefill).
  expect(
      "equal (prompt == resident)",
      plan_prefill(V{1, 2, 3}, V{1, 2, 3}, false),
      PrefillPlan::kFull,
      0,
      "equal");

  // Dirty + empty resident still resets as dirty (dirty checked first).
  expect(
      "dirty (empty resident)",
      plan_prefill(V{}, V{1, 2}, true),
      PrefillPlan::kFull,
      0,
      "dirty");

  printf(
      "\n%s (%d failure(s))\n",
      g_failures == 0 ? "ALL PASS" : "FAILED",
      g_failures);
  return g_failures == 0 ? 0 : 1;
}
