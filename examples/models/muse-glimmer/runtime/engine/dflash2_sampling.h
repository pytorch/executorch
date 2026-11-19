/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/examples/models/muse-glimmer/runtime/engine/sampling.h>

namespace muse_glimmer {

// candidate_ids: [steps, K], scores: [steps, K, K]. candidates[0] is the
// verified anchor; each probability row describes the predecessor actually
// selected. Target top-k/top-p filtering does not change this proposal q.
inline bool sample_dflash2_path(
    std::mt19937& rng,
    const int64_t* candidate_ids,
    const float* scores,
    int64_t steps,
    int64_t top_k,
    int64_t vocab_size,
    double temperature,
    bool greedy,
    std::vector<uint64_t>& candidates,
    std::vector<std::vector<float>>& probabilities,
    SamplingWorkspace& workspace) {
  if (steps < 1 || top_k < 1 || top_k > vocab_size ||
      candidates.size() != static_cast<size_t>(steps + 1) ||
      (!greedy && (!std::isfinite(temperature) || temperature <= 0))) {
    return false;
  }
  probabilities.resize(steps + 1);
  int64_t previous = 0;
  for (int64_t step = 0; step < steps; ++step) {
    const int64_t* ids = candidate_ids + step * top_k;
    const float* row = scores + (step * top_k + previous) * top_k;
    for (int64_t k = 0; k < top_k; ++k) {
      if (ids[k] < 0 || ids[k] >= vocab_size || !std::isfinite(row[k])) {
        return false;
      }
      for (int64_t j = 0; j < k; ++j) {
        if (ids[j] == ids[k]) {
          return false;
        }
      }
    }
    int64_t selected;
    if (greedy) {
      selected = argmax_index(row, top_k);
    } else {
      fill_sampling_probabilities(row, top_k, temperature, 0, 1.0, workspace);
      selected = categorical_sample(rng, workspace.probabilities.data(), top_k);
      auto& q = probabilities[step + 1];
      q.assign(vocab_size, 0.0f);
      for (int64_t k = 0; k < top_k; ++k) {
        q[ids[k]] = workspace.probabilities[k];
      }
    }
    candidates[step + 1] = ids[selected];
    previous = selected;
  }
  return true;
}

} // namespace muse_glimmer
