/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/muse-glimmer/runtime/engine/dflash_timing.h>

#include <cstdio>
#include <vector>

namespace {

using executorch::extension::llm::DFlashCycleTiming;
using executorch::extension::llm::DFlashDecodeTiming;

bool test_accumulates_speculative_and_target_only_cycles() {
  DFlashDecodeTiming timing;
  DFlashCycleTiming speculative;
  speculative.draft_execute_ms = 1.0;
  speculative.draft_logits_copy_ms = 2.0;
  speculative.draft_sampling_ms = 3.0;
  speculative.target_execute_ms = 4.0;
  speculative.target_logits_copy_ms = 5.0;
  speculative.target_hidden_copy_ms = 6.0;
  speculative.accept_correction_ms = 7.0;
  speculative.state_commit_ms = 8.0;
  speculative.total_cycle_ms = 40.0;
  speculative.draft_attempts_by_row = {1, 1, 1};
  speculative.draft_accepts_by_row = {1, 1, 0};
  timing.record(speculative);

  DFlashCycleTiming early_rejection;
  early_rejection.draft_attempts_by_row = {1};
  early_rejection.draft_accepts_by_row = {0};
  timing.record(early_rejection);

  DFlashCycleTiming target_only;
  target_only.target_execute_ms = 10.0;
  target_only.target_logits_copy_ms = 11.0;
  target_only.target_hidden_copy_ms = 12.0;
  target_only.accept_correction_ms = 13.0;
  target_only.state_commit_ms = 14.0;
  target_only.total_cycle_ms = 65.0;
  target_only.target_only = true;
  timing.record(target_only);

  return timing.cycles == 3 && timing.speculative_cycles == 2 &&
      timing.target_only_cycles == 1 && timing.draft_execute_ms == 1.0 &&
      timing.draft_logits_copy_ms == 2.0 && timing.draft_sampling_ms == 3.0 &&
      timing.target_execute_ms == 14.0 &&
      timing.target_logits_copy_ms == 16.0 &&
      timing.target_hidden_copy_ms == 18.0 &&
      timing.accept_correction_ms == 20.0 && timing.state_commit_ms == 22.0 &&
      timing.total_cycle_ms == 105.0 &&
      timing.draft_attempts_by_row == std::vector<int64_t>({2, 1, 1}) &&
      timing.draft_accepts_by_row == std::vector<int64_t>({1, 1, 0});
}

bool test_reset_clears_all_metrics() {
  DFlashDecodeTiming timing;
  DFlashCycleTiming cycle;
  cycle.draft_execute_ms = 1.0;
  cycle.total_cycle_ms = 2.0;
  cycle.draft_attempts_by_row = {1};
  cycle.draft_accepts_by_row = {1};
  timing.record(cycle);
  timing.reset();

  return timing.cycles == 0 && timing.speculative_cycles == 0 &&
      timing.target_only_cycles == 0 && timing.draft_execute_ms == 0.0 &&
      timing.draft_logits_copy_ms == 0.0 && timing.draft_sampling_ms == 0.0 &&
      timing.target_execute_ms == 0.0 && timing.target_logits_copy_ms == 0.0 &&
      timing.target_hidden_copy_ms == 0.0 &&
      timing.accept_correction_ms == 0.0 && timing.state_commit_ms == 0.0 &&
      timing.total_cycle_ms == 0.0 && timing.draft_attempts_by_row.empty() &&
      timing.draft_accepts_by_row.empty();
}

} // namespace

int main() {
  if (!test_accumulates_speculative_and_target_only_cycles()) {
    std::fprintf(
        stderr, "FAILED: accumulates speculative and target-only cycles\n");
    return 1;
  }
  if (!test_reset_clears_all_metrics()) {
    std::fprintf(stderr, "FAILED: reset clears all metrics\n");
    return 1;
  }
  std::printf("All DFlash timing tests passed.\n");
  return 0;
}
