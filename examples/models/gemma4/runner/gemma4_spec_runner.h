/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/result.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace executorch::examples::gemma4 {

struct Gemma4K2Output {
  std::array<int64_t, 2> candidates{};
  std::array<int64_t, 3> target_greedy{};
  int64_t match_count = -1;
  int64_t bonus = -1;
  float state_probe = 0.0f;
};

struct Gemma4K2Decision {
  bool valid = false;
  bool stopped = false;
  int64_t stop_token = -1;
  int64_t next_position = -1;
  int64_t next_seed = -1;
  size_t accepted_drafts = 0;
  std::vector<int64_t> selected;
  std::vector<int64_t> committed;
  std::vector<int64_t> discarded;
};

inline Gemma4K2Decision reconcile_gemma4_k2(
    const Gemma4K2Output& output,
    int64_t start_position,
    size_t token_budget,
    const std::vector<int64_t>& stop_tokens,
    int64_t vocab_size = 262144) {
  Gemma4K2Decision decision;
  if (start_position < 2 || token_budget == 0 || vocab_size <= 0 ||
      output.match_count < 0 || output.match_count > 2 ||
      !std::isfinite(output.state_probe)) {
    return decision;
  }
  const auto valid_token = [vocab_size](int64_t token) {
    return token >= 0 && token < vocab_size;
  };
  for (int64_t token : output.candidates) {
    if (!valid_token(token)) {
      return decision;
    }
  }
  for (int64_t token : output.target_greedy) {
    if (!valid_token(token)) {
      return decision;
    }
  }
  int64_t expected_matches = 0;
  if (output.candidates[0] == output.target_greedy[0]) {
    expected_matches = output.candidates[1] == output.target_greedy[1] ? 2 : 1;
  }
  if (output.match_count != expected_matches || !valid_token(output.bonus) ||
      output.bonus != output.target_greedy[output.match_count]) {
    return decision;
  }

  decision.accepted_drafts = static_cast<size_t>(output.match_count);
  decision.next_position = start_position + output.match_count + 1;
  decision.next_seed = output.bonus;
  for (int64_t index = 0; index < output.match_count; ++index) {
    decision.selected.push_back(output.candidates[index]);
  }
  decision.selected.push_back(output.bonus);

  for (size_t index = 0; index < decision.selected.size(); ++index) {
    const int64_t token = decision.selected[index];
    if (std::find(stop_tokens.begin(), stop_tokens.end(), token) !=
        stop_tokens.end()) {
      decision.stopped = true;
      decision.stop_token = token;
      decision.discarded.insert(
          decision.discarded.end(),
          decision.selected.begin() + index + 1,
          decision.selected.end());
      break;
    }
    if (decision.committed.size() == token_budget) {
      decision.discarded.insert(
          decision.discarded.end(),
          decision.selected.begin() + index,
          decision.selected.end());
      break;
    }
    decision.committed.push_back(token);
  }
  decision.valid = true;
  return decision;
}

struct Gemma4SpecRunnerConfig {
  int64_t vocab_size = 262144;
  int64_t max_input_length = 512;
  int64_t target_capacity = 8960;
  int64_t donor_capacity = 8960;
  std::string method_name = "k2_round";
};

runtime::Error validate_gemma4_spec_request(
    const Gemma4SpecRunnerConfig& config,
    const std::vector<int64_t>& prompt_ids,
    size_t token_budget,
    const std::vector<int64_t>& stop_tokens);

enum class Gemma4SpecLoadMode {
  File,
  Mmap,
};

struct Gemma4SpecTrace {
  int64_t prefill_token = -1;
  std::optional<int64_t> stop_token;
  size_t execute_count = 0;
  size_t accepted_drafts = 0;
  size_t discarded_tokens = 0;
  std::vector<int64_t> tokens;
  std::vector<Gemma4K2Decision> rounds;
};

class Gemma4SpecRunner final {
 public:
  explicit Gemma4SpecRunner(Gemma4SpecRunnerConfig config = {});
  ~Gemma4SpecRunner();

  Gemma4SpecRunner(const Gemma4SpecRunner&) = delete;
  Gemma4SpecRunner& operator=(const Gemma4SpecRunner&) = delete;

  runtime::Error load(
      const std::string& pte_path,
      std::vector<std::string> ptd_paths,
      Gemma4SpecLoadMode load_mode = Gemma4SpecLoadMode::File);
  runtime::Error reset();
  runtime::Error unload();

  bool is_loaded() const;
  runtime::Result<Gemma4K2Output> execute(
      const std::vector<int64_t>& input_ids,
      const std::vector<int64_t>& input_positions,
      bool is_round,
      int64_t donor_length);
  runtime::Result<int64_t> prefill(
      const std::vector<int64_t>& input_ids,
      int64_t start_position);
  runtime::Error prefill_step(int64_t token, int64_t position);
  runtime::Result<int64_t> step(int64_t seed_token, int64_t seed_position);
  runtime::Result<Gemma4SpecTrace> generate(
      const std::vector<int64_t>& prompt_ids,
      size_t token_budget,
      const std::vector<int64_t>& stop_tokens);

  void set_profiling_enabled(bool enabled);
  std::string profile_json();

  size_t execute_count() const;
  size_t accepted_drafts() const;
  size_t buffered_tokens() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace executorch::examples::gemma4
