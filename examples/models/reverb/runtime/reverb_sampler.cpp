/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "reverb_sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace reverb {
namespace {

// Marks the `count` entries with the highest (or lowest) finite scores. Ties
// go to the lower index.
std::vector<bool>
top(const std::vector<float>& score, int64_t count, bool highest) {
  std::vector<int> order(score.size());
  std::iota(order.begin(), order.end(), 0);
  count = std::min<int64_t>(count, static_cast<int64_t>(score.size()));
  std::vector<bool> chosen(score.size(), false);
  if (count <= 0) {
    return chosen;
  }
  std::partial_sort(
      order.begin(), order.begin() + count, order.end(), [&](int a, int b) {
        if (score[a] != score[b]) {
          return highest ? score[a] > score[b] : score[a] < score[b];
        }
        return a < b;
      });
  for (int64_t i = 0; i < count; ++i) {
    if (std::isfinite(score[order[i]])) {
      chosen[order[i]] = true;
    }
  }
  return chosen;
}

} // namespace

void validate(const SamplerConfig& config) {
  if (config.steps < 1) {
    throw std::invalid_argument("steps must be at least 1");
  }
  if (!(config.remask_ratio >= 0.0f && config.remask_ratio < 1.0f)) {
    throw std::invalid_argument("remask_ratio must be in [0, 1)");
  }
  if (!(config.t_shift > 0.0f && config.t_shift <= 1.0f)) {
    throw std::invalid_argument("t_shift must be in (0, 1]");
  }
  if (!(config.position_temperature > 0.0f)) {
    throw std::invalid_argument("position_temperature must be positive");
  }
}

std::vector<float> commit_schedule(const SamplerConfig& config) {
  std::vector<float> t(config.steps + 1);
  for (int i = 0; i <= config.steps; ++i) {
    const float x = static_cast<float>(i) / static_cast<float>(config.steps);
    t[i] = config.t_shift * x / (1.0f - (1.0f - config.t_shift) * x);
  }
  return t;
}

float Noise::uniform() {
  return std::generate_canonical<float, 24>(engine_);
}

void apply_main_pass(
    std::vector<int64_t>& tokens,
    int frames,
    const MainPass& pass,
    int step,
    const SamplerConfig& config,
    const std::vector<float>& schedule,
    Noise& noise) {
  const int slots = kMainCodebooks * frames;
  if (tokens.size() != static_cast<size_t>(kCodebooks) * frames) {
    throw std::invalid_argument("tokens must hold [8, frames] codes");
  }
  if (step < 0 || step >= config.steps ||
      schedule.size() != static_cast<size_t>(config.steps) + 1) {
    throw std::invalid_argument("step is outside the commit schedule");
  }

  int64_t masked_count = 0;
  int64_t revealed_count = 0;
  for (int i = 0; i < slots; ++i) {
    (tokens[i] == kMask ? masked_count : revealed_count) += 1;
  }
  const bool last = step == config.steps - 1;
  const int64_t quota = static_cast<int64_t>(std::ceil(
      2.0 * frames *
      (static_cast<double>(schedule[step + 1]) - schedule[step])));
  const int64_t commits = last ? masked_count : std::min(masked_count, quota);

  // Gumbel-perturbed confidence, penalised per codebook, over masked slots.
  constexpr float kInf = std::numeric_limits<float>::infinity();
  std::vector<float> score(slots, -kInf);
  for (int i = 0; i < slots; ++i) {
    const float u = noise.uniform();
    const float gumbel = -std::log(-std::log(u + 1e-10f) + 1e-10f);
    if (tokens[i] == kMask) {
      const float penalty =
          config.codebook_penalty * static_cast<float>(i / frames);
      score[i] =
          (pass.confidence[i] - penalty) / config.position_temperature + gumbel;
    }
  }
  const std::vector<bool> chosen = top(score, commits, /*highest=*/true);

  std::vector<bool> doubted(slots, false);
  if (config.remasking && !last) {
    // Withdraw the held codes the model now finds least likely: remask_ratio
    // as many as this pass commits, rounding half to even.
    const int64_t wanted = static_cast<int64_t>(
        std::nearbyint(static_cast<double>(config.remask_ratio) * commits));
    std::vector<float> likely(slots, kInf);
    for (int i = 0; i < slots; ++i) {
      if (tokens[i] != kMask) {
        likely[i] = pass.held[i];
      }
    }
    doubted = top(likely, std::min(wanted, revealed_count), /*highest=*/false);
  }

  for (int i = 0; i < slots; ++i) {
    if (chosen[i]) {
      tokens[i] = pass.prediction[i];
    } else if (doubted[i]) {
      tokens[i] = kMask;
    }
  }
}

} // namespace reverb
