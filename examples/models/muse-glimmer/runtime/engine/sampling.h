// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// Shared sampling utilities for speculative decoding runners (DFlash).

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif

#include <executorch/extension/tensor/tensor.h>

namespace muse_glimmer {
namespace detail {

inline uint64_t argmax_index(const float* values, int64_t size) {
#if defined(__aarch64__) || defined(_M_ARM64)
  int64_t offset = 0;
  float32x4_t best_values =
      vdupq_n_f32(-std::numeric_limits<float>::infinity());
  uint32x4_t best_indices = vdupq_n_u32(0);
  uint32x4_t indices = {0, 1, 2, 3};
  const uint32x4_t stride = vdupq_n_u32(4);
  for (; offset + 4 <= size; offset += 4) {
    const float32x4_t current = vld1q_f32(values + offset);
    const uint32x4_t better = vcgtq_f32(current, best_values);
    best_values = vbslq_f32(better, current, best_values);
    best_indices = vbslq_u32(better, indices, best_indices);
    indices = vaddq_u32(indices, stride);
  }

  float lane_values[4];
  uint32_t lane_indices[4];
  vst1q_f32(lane_values, best_values);
  vst1q_u32(lane_indices, best_indices);
  float best_value = lane_values[0];
  uint64_t best_index = lane_indices[0];
  for (int lane = 1; lane < 4; ++lane) {
    if (lane_values[lane] > best_value ||
        (lane_values[lane] == best_value && lane_indices[lane] < best_index)) {
      best_value = lane_values[lane];
      best_index = lane_indices[lane];
    }
  }
  for (; offset < size; ++offset) {
    if (values[offset] > best_value) {
      best_value = values[offset];
      best_index = offset;
    }
  }
  return best_index;
#else
  return static_cast<uint64_t>(
      std::distance(values, std::max_element(values, values + size)));
#endif
}

inline float max_value(const float* values, int64_t size) {
#if defined(__aarch64__) || defined(_M_ARM64)
  int64_t offset = 0;
  float32x4_t maximum = vdupq_n_f32(-std::numeric_limits<float>::infinity());
  for (; offset + 4 <= size; offset += 4) {
    maximum = vmaxq_f32(maximum, vld1q_f32(values + offset));
  }
  float result = vmaxvq_f32(maximum);
  for (; offset < size; ++offset) {
    result = std::max(result, values[offset]);
  }
  return result;
#else
  return *std::max_element(values, values + size);
#endif
}

inline void
normalize_probabilities(float* probabilities, int64_t size, float sum) {
#if defined(__aarch64__) || defined(_M_ARM64)
  int64_t offset = 0;
  const float32x4_t denominator = vdupq_n_f32(sum);
  for (; offset + 4 <= size; offset += 4) {
    vst1q_f32(
        probabilities + offset,
        vdivq_f32(vld1q_f32(probabilities + offset), denominator));
  }
  for (; offset < size; ++offset) {
    probabilities[offset] /= sum;
  }
#else
  for (int64_t offset = 0; offset < size; ++offset) {
    probabilities[offset] /= sum;
  }
#endif
}

} // namespace detail

inline uint64_t argmax_index(const float* values, int64_t size) {
  return detail::argmax_index(values, size);
}

inline float max_value(const float* values, int64_t size) {
  return detail::max_value(values, size);
}

inline void
normalize_probabilities(float* probabilities, int64_t size, float sum) {
  detail::normalize_probabilities(probabilities, size, sum);
}

inline uint64_t
categorical_sample(std::mt19937& rng, const float* probs, int64_t n) {
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float r = dist(rng);
  float cumsum = 0.0f;
  for (int64_t i = 0; i < n; i++) {
    cumsum += probs[i];
    if (cumsum >= r) {
      return static_cast<uint64_t>(i);
    }
  }
  return static_cast<uint64_t>(n - 1);
}

struct SamplingWorkspace {
  std::vector<float> probabilities;
  std::vector<int32_t> candidates;
  std::vector<uint32_t> radix_keys;
  std::vector<int32_t> radix_scratch;
  std::array<int64_t, 256> radix_counts{};
};

namespace detail {

inline uint32_t float_to_descending_radix_key(float value) {
  static_assert(sizeof(float) == sizeof(uint32_t));
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7fffffffu) == 0) {
    bits = 0;
  }
  const uint32_t ascending_key =
      bits ^ ((bits & 0x80000000u) != 0 ? 0xffffffffu : 0x80000000u);
  return ~ascending_key;
}

inline void stable_radix_sort_indices(
    const float* logits,
    int64_t vocab_size,
    SamplingWorkspace& workspace) {
  auto& candidates = workspace.candidates;
  auto& keys = workspace.radix_keys;
  auto& scratch = workspace.radix_scratch;
  auto& counts = workspace.radix_counts;

  candidates.resize(vocab_size);
  std::iota(candidates.begin(), candidates.end(), int32_t{0});
  keys.resize(vocab_size);
  scratch.resize(vocab_size);
  for (int64_t token = 0; token < vocab_size; ++token) {
    keys[token] = float_to_descending_radix_key(logits[token]);
  }

  auto* source = &candidates;
  auto* destination = &scratch;
  for (uint32_t shift = 0; shift < 32; shift += 8) {
    counts.fill(0);
    for (const int32_t token : *source) {
      ++counts[(keys[token] >> shift) & 0xffu];
    }

    int64_t offset = 0;
    for (int64_t& count : counts) {
      const int64_t bucket_size = count;
      count = offset;
      offset += bucket_size;
    }
    for (const int32_t token : *source) {
      const uint32_t bucket = (keys[token] >> shift) & 0xffu;
      (*destination)[counts[bucket]++] = token;
    }
    std::swap(source, destination);
  }
  ET_CHECK(source == &candidates);
}

inline double fill_dense_weights(
    const float* logits,
    int64_t vocab_size,
    double temperature,
    std::vector<float>& probabilities) {
  probabilities.resize(vocab_size);
  const float maximum = max_value(logits, vocab_size);
  double sum = 0.0;
  for (int64_t token = 0; token < vocab_size; ++token) {
    probabilities[token] =
        std::exp((logits[token] - maximum) / static_cast<float>(temperature));
    sum += probabilities[token];
  }
  return sum;
}

inline double fill_top_k_weights(
    const float* logits,
    int64_t vocab_size,
    double temperature,
    int32_t top_k,
    SamplingWorkspace& workspace) {
  auto& probabilities = workspace.probabilities;
  auto& candidates = workspace.candidates;
  probabilities.assign(vocab_size, 0.0f);
  candidates.resize(vocab_size);
  std::iota(candidates.begin(), candidates.end(), int32_t{0});
  const auto higher_logit = [logits](int32_t lhs, int32_t rhs) {
    if (logits[lhs] != logits[rhs]) {
      return logits[lhs] > logits[rhs];
    }
    return lhs < rhs;
  };
  std::nth_element(
      candidates.begin(),
      candidates.begin() + top_k,
      candidates.end(),
      higher_logit);
  candidates.resize(top_k);

  const auto max_it = std::max_element(
      candidates.begin(), candidates.end(), [logits](int32_t lhs, int32_t rhs) {
        return logits[lhs] < logits[rhs];
      });
  const float maximum = logits[*max_it];
  double sum = 0.0;
  for (const int32_t token : candidates) {
    probabilities[token] =
        std::exp((logits[token] - maximum) / static_cast<float>(temperature));
    sum += probabilities[token];
  }
  return sum;
}

inline void normalize_candidates(
    std::vector<float>& probabilities,
    const std::vector<int32_t>& candidates,
    size_t selected_begin,
    size_t selected_end,
    double sum) {
  for (size_t index = selected_begin; index < selected_end; ++index) {
    probabilities[candidates[index]] /= static_cast<float>(sum);
  }
}

inline void apply_top_p_radix(
    const float* logits,
    int64_t vocab_size,
    double top_p,
    double sum,
    SamplingWorkspace& workspace) {
  stable_radix_sort_indices(logits, vocab_size, workspace);
  auto& probabilities = workspace.probabilities;
  const auto& candidates = workspace.candidates;
  const double nucleus_target = top_p * sum;
  double nucleus_sum = 0.0;
  size_t selected_count = 0;
  while (selected_count < candidates.size() && nucleus_sum < nucleus_target) {
    nucleus_sum += probabilities[candidates[selected_count]];
    ++selected_count;
  }
  for (size_t index = selected_count; index < candidates.size(); ++index) {
    probabilities[candidates[index]] = 0.0f;
  }
  normalize_candidates(
      probabilities, candidates, 0, selected_count, nucleus_sum);
}

inline void apply_top_p_heap(
    const float* logits,
    double top_p,
    double sum,
    SamplingWorkspace& workspace) {
  auto& probabilities = workspace.probabilities;
  auto& candidates = workspace.candidates;
  const auto lower_logit = [logits](int32_t lhs, int32_t rhs) {
    if (logits[lhs] != logits[rhs]) {
      return logits[lhs] < logits[rhs];
    }
    return lhs > rhs;
  };
  std::make_heap(candidates.begin(), candidates.end(), lower_logit);
  auto heap_end = candidates.end();
  const double nucleus_target = top_p * sum;
  double nucleus_sum = 0.0;
  while (heap_end != candidates.begin() && nucleus_sum < nucleus_target) {
    std::pop_heap(candidates.begin(), heap_end, lower_logit);
    --heap_end;
    nucleus_sum += probabilities[*heap_end];
  }
  for (auto it = candidates.begin(); it != heap_end; ++it) {
    probabilities[*it] = 0.0f;
  }
  normalize_candidates(
      probabilities,
      candidates,
      static_cast<size_t>(std::distance(candidates.begin(), heap_end)),
      candidates.size(),
      nucleus_sum);
}

inline void fill_sampling_probabilities(
    const float* logits,
    int64_t vocab_size,
    double temperature,
    int32_t top_k,
    double top_p,
    SamplingWorkspace& workspace) {
  const bool use_top_k = top_k > 0 && top_k < vocab_size;
  const bool use_top_p = top_p > 0.0 && top_p < 1.0;
  auto& probabilities = workspace.probabilities;

  if (!use_top_k) {
    const double sum =
        fill_dense_weights(logits, vocab_size, temperature, probabilities);
    if (use_top_p) {
      ET_CHECK_MSG(
          vocab_size <= std::numeric_limits<int32_t>::max(),
          "Sampling vocabulary exceeds 32-bit candidate index range");
      apply_top_p_radix(logits, vocab_size, top_p, sum, workspace);
    } else {
      normalize_probabilities(
          probabilities.data(), vocab_size, static_cast<float>(sum));
    }
    return;
  }

  const double sum =
      fill_top_k_weights(logits, vocab_size, temperature, top_k, workspace);
  if (use_top_p) {
    apply_top_p_heap(logits, top_p, sum, workspace);
  } else {
    normalize_candidates(
        probabilities,
        workspace.candidates,
        0,
        workspace.candidates.size(),
        sum);
  }
}

} // namespace detail

inline void fill_sampling_probabilities(
    const float* logits,
    int64_t vocab_size,
    double temperature,
    int32_t top_k,
    double top_p,
    SamplingWorkspace& workspace) {
  detail::fill_sampling_probabilities(
      logits, vocab_size, temperature, top_k, top_p, workspace);
}

inline void fill_sampling_probabilities(
    const float* logits,
    int64_t vocab_size,
    double temperature,
    int32_t top_k,
    double top_p,
    std::vector<float>& probabilities,
    SamplingWorkspace& workspace) {
  if (&probabilities == &workspace.probabilities) {
    fill_sampling_probabilities(
        logits, vocab_size, temperature, top_k, top_p, workspace);
    return;
  }

  probabilities.swap(workspace.probabilities);
  fill_sampling_probabilities(
      logits, vocab_size, temperature, top_k, top_p, workspace);
  probabilities.swap(workspace.probabilities);
}

inline void fill_sampling_probabilities(
    const float* logits,
    int64_t vocab_size,
    double temperature,
    int32_t top_k,
    double top_p,
    std::vector<float>& probabilities,
    std::vector<int32_t>& candidates) {
  SamplingWorkspace workspace;
  probabilities.swap(workspace.probabilities);
  candidates.swap(workspace.candidates);
  fill_sampling_probabilities(
      logits, vocab_size, temperature, top_k, top_p, workspace);
  probabilities.swap(workspace.probabilities);
  candidates.swap(workspace.candidates);
}

inline std::vector<float> sampling_probabilities(
    const float* logits,
    int64_t vocab_size,
    double temperature,
    int32_t top_k,
    double top_p) {
  SamplingWorkspace workspace;
  fill_sampling_probabilities(
      logits,
      vocab_size,
      temperature,
      top_k,
      top_p,
      workspace.probabilities,
      workspace);
  return std::move(workspace.probabilities);
}

inline uint64_t sample_token(
    std::mt19937& rng,
    const executorch::aten::Tensor& logits,
    int64_t seq_pos,
    double temperature,
    int32_t top_k = 0,
    double top_p = 1.0,
    std::vector<float>* out_probs = nullptr,
    bool probs_only = false,
    SamplingWorkspace* workspace = nullptr) {
  int64_t vocab_size = logits.size(logits.dim() - 1);
  const float* src =
      static_cast<const float*>(logits.const_data_ptr()) + seq_pos * vocab_size;

  if (temperature <= 0.0) {
    return argmax_index(src, vocab_size);
  }

  SamplingWorkspace local_workspace;
  SamplingWorkspace& scratch =
      workspace == nullptr ? local_workspace : *workspace;
  std::vector<float>& probs =
      out_probs == nullptr ? scratch.probabilities : *out_probs;
  fill_sampling_probabilities(
      src, vocab_size, temperature, top_k, top_p, probs, scratch);
  if (out_probs && probs_only) {
    return 0;
  }
  return categorical_sample(rng, probs.data(), vocab_size);
}

inline bool accept_with_probability(std::mt19937& rng, float probability) {
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  return uniform(rng) < probability;
}

inline uint64_t sample_excluding_token_in_place(
    std::mt19937& rng,
    std::vector<float>& probabilities,
    uint64_t excluded_token) {
  ET_CHECK(excluded_token < probabilities.size());
  probabilities[excluded_token] = 0.0f;
  double sum = 0.0;
  for (const float probability : probabilities) {
    sum += probability;
  }
  ET_CHECK_MSG(sum > 0.0, "Cannot sample after excluding all probability mass");
  normalize_probabilities(
      probabilities.data(), probabilities.size(), static_cast<float>(sum));

  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  const float sample = uniform(rng);
  float cumulative = 0.0f;
  uint64_t last_supported_token = 0;
  for (uint64_t token = 0; token < probabilities.size(); ++token) {
    const float probability = probabilities[token];
    if (probability == 0.0f) {
      continue;
    }
    last_supported_token = token;
    cumulative += probability;
    if (sample < cumulative) {
      return token;
    }
  }
  return last_supported_token;
}

inline uint64_t sample_from_residual_in_place(
    std::mt19937& rng,
    std::vector<float>& p,
    const std::vector<float>& q) {
  int64_t n = static_cast<int64_t>(p.size());
  double sum = 0.0;
  for (int64_t i = 0; i < n; i++) {
    sum += std::max(0.0f, p[i] - q[i]);
  }
  if (sum <= 0.0) {
    return categorical_sample(rng, p.data(), n);
  }
  for (int64_t i = 0; i < n; i++) {
    p[i] = std::max(0.0f, p[i] - q[i]) / static_cast<float>(sum);
  }
  return categorical_sample(rng, p.data(), n);
}

inline uint64_t sample_from_residual(
    std::mt19937& rng,
    const std::vector<float>& p,
    const std::vector<float>& q) {
  std::vector<float> residual = p;
  return sample_from_residual_in_place(rng, residual, q);
}

} // namespace muse_glimmer
