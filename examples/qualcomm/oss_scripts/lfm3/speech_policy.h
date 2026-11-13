/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

namespace example::speech {

struct SamplingPolicy {
  float temperature = 0;
  int32_t top_k = 0;
  float top_p = 1;
  float min_p = 0;
  float repetition_penalty = 1;

  bool valid() const {
    return std::isfinite(temperature) && temperature >= 0 && top_k >= 0 &&
        std::isfinite(top_p) && top_p >= 0 && top_p <= 1 &&
        std::isfinite(min_p) && min_p >= 0 && min_p <= 1 &&
        std::isfinite(repetition_penalty) && repetition_penalty > 0;
  }
};

inline SamplingPolicy audio_overrides(
    SamplingPolicy policy,
    std::optional<float> temperature,
    std::optional<float> penalty) {
  if (temperature.has_value()) {
    policy.temperature = *temperature;
  }
  if (penalty.has_value()) {
    policy.repetition_penalty = *penalty;
  }
  return policy;
}

inline const uint16_t*
last_logits_row(const uint16_t* logits, size_t valid_rows, size_t vocab_size) {
  if (logits == nullptr || valid_rows == 0 || vocab_size == 0) {
    return nullptr;
  }
  return logits + (valid_rows - 1) * vocab_size;
}

template <typename T>
void append_history(std::deque<T>& history, T value, size_t window) {
  if (window == 0) {
    history.clear();
    return;
  }
  history.push_back(std::move(value));
  while (history.size() > window) {
    history.pop_front();
  }
}

inline std::vector<float> dequantize_logits(
    const uint16_t* logits,
    size_t size,
    float scale,
    int32_t zero_point) {
  std::vector<float> scores(size);
  for (size_t i = 0; i < size; ++i) {
    scores[i] = (static_cast<int32_t>(logits[i]) - zero_point) * scale;
  }
  return scores;
}

inline void penalize(
    std::vector<float>& scores,
    const std::unordered_set<int32_t>& seen,
    float penalty) {
  for (int32_t token : seen) {
    scores[token] =
        scores[token] < 0 ? scores[token] * penalty : scores[token] / penalty;
  }
}

inline std::vector<double> sampling_weights(
    std::vector<float> scores,
    const SamplingPolicy& policy) {
  // Warper order in Omni: temperature, top-k, top-p, min-p.
  for (float& score : scores) {
    score /= policy.temperature;
  }
  if (policy.top_k > 0 && static_cast<size_t>(policy.top_k) < scores.size()) {
    auto sorted = scores;
    std::nth_element(
        sorted.begin(),
        sorted.begin() + policy.top_k - 1,
        sorted.end(),
        std::greater<float>());
    const float cutoff = sorted[policy.top_k - 1];
    for (float& score : scores) {
      if (score < cutoff) {
        score = -std::numeric_limits<float>::infinity();
      }
    }
  }
  const float maximum = *std::max_element(scores.begin(), scores.end());
  std::vector<double> weights(scores.size());
  for (size_t i = 0; i < scores.size(); ++i) {
    weights[i] = std::exp(static_cast<double>(scores[i] - maximum));
  }
  if (policy.top_p < 1) {
    std::vector<size_t> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
      return scores[a] < scores[b];
    });
    const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
    double cumulative = 0;
    for (size_t j = 0; j + 1 < order.size(); ++j) {
      const size_t i = order[j];
      cumulative += weights[i] / total;
      if (cumulative <= 1.0 - policy.top_p) {
        weights[i] = 0;
      }
    }
  }
  if (policy.min_p > 0) {
    const double cutoff =
        *std::max_element(weights.begin(), weights.end()) * policy.min_p;
    for (double& weight : weights) {
      if (weight < cutoff) {
        weight = 0;
      }
    }
  }
  return weights;
}

inline int32_t categorical(const std::vector<double>& weights, double draw) {
  int32_t last_positive = -1;
  for (size_t i = 0; i < weights.size(); ++i) {
    if (weights[i] <= 0) {
      continue;
    }
    last_positive = static_cast<int32_t>(i);
    if (draw < weights[i]) {
      return last_positive;
    }
    draw -= weights[i];
  }
  // A rounding residual must not invent a sample with zero probability.
  return last_positive;
}

template <typename RNG>
int32_t sample_scores(
    std::vector<float> scores,
    const SamplingPolicy& policy,
    RNG& rng) {
  const float maximum = scores.empty()
      ? -std::numeric_limits<float>::infinity()
      : *std::max_element(scores.begin(), scores.end());
  if (!policy.valid() || !std::isfinite(maximum) ||
      !std::all_of(scores.begin(), scores.end(), [](float score) {
        return std::isfinite(score) ||
            score == -std::numeric_limits<float>::infinity();
      })) {
    return -1;
  }
  if (policy.temperature == 0) {
    return std::max_element(scores.begin(), scores.end()) - scores.begin();
  }
  const auto weights = sampling_weights(std::move(scores), policy);
  const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
  if (!std::isfinite(total) || total <= 0) {
    return -1;
  }
  return categorical(
      weights, std::uniform_real_distribution<double>(0.0, total)(rng));
}

template <typename RNG, typename Suppressed>
int32_t sample_text_logits(
    const uint16_t* logits,
    size_t vocab_size,
    float scale,
    int32_t zero_point,
    const SamplingPolicy& policy,
    std::deque<int32_t>& history,
    size_t window,
    const Suppressed& suppressed,
    RNG& rng) {
  auto scores = dequantize_logits(logits, vocab_size, scale, zero_point);
  const std::unordered_set<int32_t> seen(history.begin(), history.end());
  penalize(scores, seen, policy.repetition_penalty);
  for (uint64_t token : suppressed) {
    scores[token] = -std::numeric_limits<float>::infinity();
  }
  const int32_t token = sample_scores(std::move(scores), policy, rng);
  if (token >= 0) {
    append_history(history, token, window);
  }
  return token;
}

template <typename Frame>
bool normalize_audio_eos(Frame& codes, int32_t eos) {
  if (std::find(codes.begin(), codes.end(), eos) == codes.end()) {
    return false;
  }
  codes.fill(eos);
  return true;
}

enum class Mode { kText, kAudio, kInterleaved };

struct ModeState {
  Mode mode = Mode::kText;
  bool speaking = false;
  bool text_done = false;
  int slots_left = 6;

  bool audio_slot() const {
    return mode == Mode::kAudio || (mode == Mode::kInterleaved && speaking);
  }

  bool text_token(
      uint64_t token,
      bool eos,
      uint64_t text_end,
      uint64_t audio_start,
      uint64_t mixed_start) {
    if (token == text_end) {
      if (mode == Mode::kInterleaved) {
        text_done = speaking = true;
      } else {
        return true;
      }
    } else if (eos) {
      return true;
    } else if (mode != Mode::kInterleaved && token == audio_start) {
      mode = Mode::kAudio;
    } else if (mode != Mode::kInterleaved && token == mixed_start) {
      mode = Mode::kInterleaved;
      slots_left = 6;
      speaking = text_done = false;
    }
    return false;
  }

  void advance(bool was_interleaved) {
    if (was_interleaved && !text_done && --slots_left == 0) {
      speaking = !speaking;
      slots_left = speaking ? 12 : 6;
    }
  }
};

} // namespace example::speech
