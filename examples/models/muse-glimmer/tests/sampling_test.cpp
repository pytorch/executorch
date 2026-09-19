/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/muse-glimmer/runtime/engine/sampling.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

namespace {

constexpr float kTolerance = 1e-6f;

bool near(float actual, float expected) {
  return std::abs(actual - expected) <= kTolerance;
}

bool check_normalized(const std::vector<float>& probs) {
  double sum = 0.0;
  for (const float probability : probs) {
    if (probability < 0.0f) {
      return false;
    }
    sum += probability;
  }
  return std::abs(sum - 1.0) <= kTolerance;
}

std::vector<float> reference_probabilities(
    const std::vector<float>& logits,
    double temperature,
    int32_t top_k,
    double top_p) {
  std::vector<int64_t> indices(logits.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&](int64_t lhs, int64_t rhs) {
    if (logits[lhs] != logits[rhs]) {
      return logits[lhs] > logits[rhs];
    }
    return lhs < rhs;
  });

  int64_t retained = logits.size();
  if (top_k > 0 && top_k < retained) {
    retained = top_k;
  }
  std::vector<float> probs(logits.size(), 0.0f);
  const float max_val = logits[indices.front()];
  double sum = 0.0;
  for (int64_t i = 0; i < retained; ++i) {
    probs[indices[i]] = std::exp(
        (logits[indices[i]] - max_val) / static_cast<float>(temperature));
    sum += probs[indices[i]];
  }
  if (top_p > 0.0 && top_p < 1.0) {
    double nucleus_sum = 0.0;
    int64_t cutoff = 0;
    while (cutoff < retained && nucleus_sum < top_p * sum) {
      nucleus_sum += probs[indices[cutoff++]];
    }
    for (int64_t i = cutoff; i < retained; ++i) {
      probs[indices[i]] = 0.0f;
    }
    retained = cutoff;
    sum = nucleus_sum;
  }
  for (int64_t i = 0; i < retained; ++i) {
    probs[indices[i]] /= static_cast<float>(sum);
  }
  return probs;
}

bool test_argmax_index_tie_and_tail() {
  const std::vector<float> values = {-4.0f, 3.0f, 1.0f, 3.0f, 2.0f, 3.0f, 0.0f};
  return muse_glimmer::argmax_index(values.data(), values.size()) == 1;
}

bool test_max_value_all_negative_with_tail() {
  const std::vector<float> values = {-9.0f, -4.0f, -7.0f, -8.0f, -5.0f};
  return muse_glimmer::max_value(values.data(), values.size()) == -4.0f;
}

bool test_normalize_probabilities_with_tail() {
  std::vector<float> probabilities = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  muse_glimmer::normalize_probabilities(
      probabilities.data(), probabilities.size(), 15.0f);
  const std::vector<float> expected = {
      1.0f / 15.0f, 2.0f / 15.0f, 3.0f / 15.0f, 4.0f / 15.0f, 5.0f / 15.0f};
  for (size_t i = 0; i < probabilities.size(); ++i) {
    if (!near(probabilities[i], expected[i])) {
      return false;
    }
  }
  return check_normalized(probabilities);
}

bool test_unfiltered_softmax() {
  const std::vector<float> logits = {0.0f, 1.0f, 2.0f};
  const auto probs = muse_glimmer::sampling_probabilities(
      logits.data(), logits.size(), 1.0, 0, 1.0);
  const double sum = 1.0 + std::exp(1.0) + std::exp(2.0);
  return check_normalized(probs) && near(probs[0], 1.0 / sum) &&
      near(probs[1], std::exp(1.0) / sum) &&
      near(probs[2], std::exp(2.0) / sum);
}

bool test_top_k() {
  const std::vector<float> logits = {0.0f, 3.0f, 1.0f, 2.0f};
  const auto probs = muse_glimmer::sampling_probabilities(
      logits.data(), logits.size(), 1.0, 2, 1.0);
  const double sum = std::exp(3.0) + std::exp(2.0);
  return check_normalized(probs) && probs[0] == 0.0f &&
      near(probs[1], std::exp(3.0) / sum) && probs[2] == 0.0f &&
      near(probs[3], std::exp(2.0) / sum);
}

bool test_top_p_includes_crossing_token() {
  const std::vector<float> logits = {
      std::log(0.5f), std::log(0.3f), std::log(0.15f), std::log(0.05f)};
  const auto probs = muse_glimmer::sampling_probabilities(
      logits.data(), logits.size(), 1.0, 0, 0.75);
  return check_normalized(probs) && near(probs[0], 0.625f) &&
      near(probs[1], 0.375f) && probs[2] == 0.0f && probs[3] == 0.0f;
}

bool test_top_p_tie_breaks_by_token_id() {
  const std::vector<float> logits = {-0.0f, 0.0f, 0.0f};
  const auto probs = muse_glimmer::sampling_probabilities(
      logits.data(), logits.size(), 1.0, 0, 0.5);
  return check_normalized(probs) && near(probs[0], 0.5f) &&
      near(probs[1], 0.5f) && probs[2] == 0.0f;
}

bool test_top_k_then_top_p() {
  const std::vector<float> logits = {
      std::log(0.4f), std::log(0.3f), std::log(0.2f), std::log(0.1f)};
  const auto probs = muse_glimmer::sampling_probabilities(
      logits.data(), logits.size(), 1.0, 3, 0.6);
  return check_normalized(probs) && near(probs[0], 4.0f / 7.0f) &&
      near(probs[1], 3.0f / 7.0f) && probs[2] == 0.0f && probs[3] == 0.0f;
}

bool test_top_k_tie_breaks_by_token_id() {
  const std::vector<float> logits = {1.0f, 1.0f, 1.0f, 0.0f};
  const auto probs = muse_glimmer::sampling_probabilities(
      logits.data(), logits.size(), 1.0, 2, 1.0);
  return check_normalized(probs) && near(probs[0], 0.5f) &&
      near(probs[1], 0.5f) && probs[2] == 0.0f && probs[3] == 0.0f;
}

bool test_full_vocab_overwrites_reused_probability_buffer() {
  const std::vector<float> logits = {
      std::log(0.5f), std::log(0.3f), std::log(0.15f), std::log(0.05f)};
  muse_glimmer::SamplingWorkspace workspace;
  workspace.probabilities.assign(logits.size(), 123.0f);
  const float* probabilities_data = workspace.probabilities.data();
  muse_glimmer::fill_sampling_probabilities(
      logits.data(),
      logits.size(),
      1.0,
      0,
      0.75,
      workspace.probabilities,
      workspace.candidates);
  return workspace.probabilities.data() == probabilities_data &&
      check_normalized(workspace.probabilities) &&
      near(workspace.probabilities[0], 0.625f) &&
      near(workspace.probabilities[1], 0.375f) &&
      workspace.probabilities[2] == 0.0f && workspace.probabilities[3] == 0.0f;
}

bool test_reuses_workspace_buffers() {
  const std::vector<float> logits = {0.0f, 3.0f, 1.0f, 2.0f};
  muse_glimmer::SamplingWorkspace workspace;
  muse_glimmer::fill_sampling_probabilities(
      logits.data(),
      logits.size(),
      1.0,
      0,
      0.95,
      workspace.probabilities,
      workspace);
  const float* probabilities_data = workspace.probabilities.data();
  const int32_t* candidates_data = workspace.candidates.data();
  const uint32_t* radix_keys_data = workspace.radix_keys.data();
  const int32_t* radix_scratch_data = workspace.radix_scratch.data();
  muse_glimmer::fill_sampling_probabilities(
      logits.data(),
      logits.size(),
      0.7,
      0,
      0.8,
      workspace.probabilities,
      workspace);
  return workspace.probabilities.data() == probabilities_data &&
      workspace.candidates.data() == candidates_data &&
      workspace.radix_keys.data() == radix_keys_data &&
      workspace.radix_scratch.data() == radix_scratch_data &&
      check_normalized(workspace.probabilities);
}

bool test_accept_with_probability() {
  std::mt19937 rejected_rng(42);
  const auto rejected_rng_before = rejected_rng;
  if (muse_glimmer::accept_with_probability(rejected_rng, 0.0f) ||
      rejected_rng == rejected_rng_before) {
    return false;
  }

  std::mt19937 accepted_rng(42);
  const std::vector<float> probabilities = {0.2f, 0.5f, 0.3f};
  const auto probabilities_before = probabilities;
  return muse_glimmer::accept_with_probability(accepted_rng, 1.0f) &&
      probabilities == probabilities_before;
}

bool test_excluded_token_correction() {
  std::mt19937 rng(42);
  std::vector<float> probabilities = {0.2f, 0.5f, 0.3f};
  const float* probabilities_data = probabilities.data();
  const uint64_t sampled =
      muse_glimmer::sample_excluding_token_in_place(rng, probabilities, 1);
  return probabilities.data() == probabilities_data && sampled != 1 &&
      check_normalized(probabilities) && near(probabilities[0], 0.4f) &&
      probabilities[1] == 0.0f && near(probabilities[2], 0.6f);
}

bool test_argmax_rejection_rng_draws() {
  std::mt19937 actual(42);
  std::mt19937 expected(42);
  std::vector<float> probabilities = {0.4f, 0.0f, 0.6f};
  if (muse_glimmer::accept_with_probability(actual, 0.0f)) {
    return false;
  }
  muse_glimmer::sample_excluding_token_in_place(actual, probabilities, 1);

  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  uniform(expected);
  uniform(expected);
  return actual == expected;
}

bool test_residual_in_place() {
  std::mt19937 rng(42);
  std::vector<float> target = {0.1f, 0.6f, 0.3f};
  const std::vector<float> draft = {0.2f, 0.2f, 0.6f};
  const float* target_data = target.data();
  const uint64_t sampled =
      muse_glimmer::sample_from_residual_in_place(rng, target, draft);
  return target.data() == target_data && target[0] == 0.0f &&
      near(target[1], 1.0f) && target[2] == 0.0f && sampled == 1;
}

bool probabilities_match(
    const std::vector<float>& actual,
    const std::vector<float>& expected) {
  if (!check_normalized(actual) || actual.size() != expected.size()) {
    return false;
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    if ((actual[i] == 0.0f) != (expected[i] == 0.0f) ||
        !near(actual[i], expected[i])) {
      return false;
    }
  }
  return true;
}

bool test_matches_full_sort_reference() {
  const std::vector<float> logits = {
      -2.0f, 1.2f, 0.3f, 4.0f, -0.7f, 2.1f, 1.2f, 0.0f};
  const struct {
    double temperature;
    int32_t top_k;
    double top_p;
  } configs[] = {
      {0.7, 0, 1.0},
      {1.0, 0, 0.95},
      {1.3, 5, 1.0},
      {0.9, 5, 0.8},
      {2.0, 1, 0.1},
  };

  for (const auto& config : configs) {
    const auto actual = muse_glimmer::sampling_probabilities(
        logits.data(),
        logits.size(),
        config.temperature,
        config.top_k,
        config.top_p);
    const auto expected = reference_probabilities(
        logits, config.temperature, config.top_k, config.top_p);
    if (!probabilities_match(actual, expected)) {
      return false;
    }
  }
  return true;
}

bool test_partition_top_p_matches_full_sort_reference() {
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> logit_distribution(-8.0f, 8.0f);
  const double temperatures[] = {0.5, 1.0, 1.7};
  const double top_ps[] = {0.1, 0.5, 0.8, 0.95, 0.999};
  const int32_t top_ks[] = {0, 1, 7, 31, 127};

  for (int iteration = 0; iteration < 20; ++iteration) {
    std::vector<float> logits(257);
    for (float& logit : logits) {
      logit = logit_distribution(rng);
    }
    for (size_t i = 0; i < logits.size(); i += 17) {
      logits[i] = 2.0f;
    }

    for (const double temperature : temperatures) {
      for (const double top_p : top_ps) {
        for (const int32_t top_k : top_ks) {
          const auto actual = muse_glimmer::sampling_probabilities(
              logits.data(), logits.size(), temperature, top_k, top_p);
          const auto expected =
              reference_probabilities(logits, temperature, top_k, top_p);
          if (!probabilities_match(actual, expected)) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

} // namespace

int main() {
  const struct {
    const char* name;
    bool (*run)();
  } tests[] = {
      {"argmax_index_tie_and_tail", test_argmax_index_tie_and_tail},
      {"max_value_all_negative_with_tail",
       test_max_value_all_negative_with_tail},
      {"normalize_probabilities_with_tail",
       test_normalize_probabilities_with_tail},
      {"unfiltered_softmax", test_unfiltered_softmax},
      {"top_k", test_top_k},
      {"top_p_includes_crossing_token", test_top_p_includes_crossing_token},
      {"top_p_tie_breaks_by_token_id", test_top_p_tie_breaks_by_token_id},
      {"top_k_then_top_p", test_top_k_then_top_p},
      {"top_k_tie_breaks_by_token_id", test_top_k_tie_breaks_by_token_id},
      {"full_vocab_overwrites_reused_probability_buffer",
       test_full_vocab_overwrites_reused_probability_buffer},
      {"reuses_workspace_buffers", test_reuses_workspace_buffers},
      {"accept_with_probability", test_accept_with_probability},
      {"excluded_token_correction", test_excluded_token_correction},
      {"argmax_rejection_rng_draws", test_argmax_rejection_rng_draws},
      {"residual_in_place", test_residual_in_place},
      {"matches_full_sort_reference", test_matches_full_sort_reference},
      {"partition_top_p_matches_full_sort_reference",
       test_partition_top_p_matches_full_sort_reference},
  };

  for (const auto& test : tests) {
    if (!test.run()) {
      std::fprintf(stderr, "FAILED: %s\n", test.name);
      return 1;
    }
  }
  std::printf("All sampling tests passed.\n");
  return 0;
}
