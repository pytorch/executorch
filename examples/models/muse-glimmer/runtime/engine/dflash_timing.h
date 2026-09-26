// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace executorch::extension::llm {

struct DFlashCycleTiming {
  double draft_execute_ms = 0.0;
  double draft_logits_copy_ms = 0.0;
  double draft_sampling_ms = 0.0;
  double target_execute_ms = 0.0;
  double target_logits_copy_ms = 0.0;
  double target_hidden_copy_ms = 0.0;
  double accept_correction_ms = 0.0;
  double state_commit_ms = 0.0;
  double total_cycle_ms = 0.0;
  std::vector<int64_t> draft_attempts_by_row;
  std::vector<int64_t> draft_accepts_by_row;
  bool target_only = false;
};

struct DFlashDecodeTiming {
  int64_t cycles = 0;
  int64_t speculative_cycles = 0;
  int64_t target_only_cycles = 0;
  double draft_execute_ms = 0.0;
  double draft_logits_copy_ms = 0.0;
  double draft_sampling_ms = 0.0;
  double target_execute_ms = 0.0;
  double target_logits_copy_ms = 0.0;
  double target_hidden_copy_ms = 0.0;
  double accept_correction_ms = 0.0;
  double state_commit_ms = 0.0;
  double total_cycle_ms = 0.0;
  std::vector<int64_t> draft_attempts_by_row;
  std::vector<int64_t> draft_accepts_by_row;

  void record(const DFlashCycleTiming& timing) {
    ++cycles;
    if (timing.target_only) {
      ++target_only_cycles;
    } else {
      ++speculative_cycles;
    }
    draft_execute_ms += timing.draft_execute_ms;
    draft_logits_copy_ms += timing.draft_logits_copy_ms;
    draft_sampling_ms += timing.draft_sampling_ms;
    target_execute_ms += timing.target_execute_ms;
    target_logits_copy_ms += timing.target_logits_copy_ms;
    target_hidden_copy_ms += timing.target_hidden_copy_ms;
    accept_correction_ms += timing.accept_correction_ms;
    state_commit_ms += timing.state_commit_ms;
    total_cycle_ms += timing.total_cycle_ms;
    if (draft_attempts_by_row.size() < timing.draft_attempts_by_row.size()) {
      draft_attempts_by_row.resize(timing.draft_attempts_by_row.size());
      draft_accepts_by_row.resize(timing.draft_accepts_by_row.size());
    }
    for (size_t row = 0; row < timing.draft_attempts_by_row.size(); ++row) {
      draft_attempts_by_row[row] += timing.draft_attempts_by_row[row];
      draft_accepts_by_row[row] += timing.draft_accepts_by_row[row];
    }
  }

  void reset() {
    *this = DFlashDecodeTiming{};
  }
};

} // namespace executorch::extension::llm
