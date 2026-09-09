/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/muse-glimmer/runtime/engine/sampling.h>
#include <executorch/examples/models/muse-glimmer/runtime/engine/sampling_cuda.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <random>
#include <tuple>
#include <vector>

namespace {

#define ASSERT_CUDA_SUCCESS(expression)               \
  do {                                                \
    const cudaError_t error = (expression);           \
    ASSERT_EQ(error, cudaSuccess)                     \
        << cudaGetErrorString(error);                 \
  } while (false)

bool expect_cuda_success(cudaError_t error) {
  if (error == cudaSuccess) {
    return true;
  }
  ADD_FAILURE() << cudaGetErrorString(error);
  return false;
}

std::vector<float> cuda_sampling_probabilities(
    const std::vector<float>& logits,
    int64_t row_count,
    int64_t row_size,
    double temperature,
    int32_t top_k,
    double top_p,
    muse_glimmer::cuda::SamplingWorkspace& workspace) {
  float* device_logits = nullptr;
  float* device_probabilities = nullptr;
  const size_t bytes = logits.size() * sizeof(float);
  if (!expect_cuda_success(cudaMalloc(&device_logits, bytes)) ||
      !expect_cuda_success(cudaMalloc(&device_probabilities, bytes)) ||
      !expect_cuda_success(cudaMemcpy(
          device_logits,
          logits.data(),
          bytes,
          cudaMemcpyHostToDevice)) ||
      !expect_cuda_success(muse_glimmer::cuda::fill_sampling_probabilities(
          device_logits,
          row_count,
          row_size,
          temperature,
          top_k,
          top_p,
          device_probabilities,
          workspace,
          nullptr))) {
    cudaFree(device_probabilities);
    cudaFree(device_logits);
    return {};
  }

  std::vector<float> probabilities(logits.size());
  if (!expect_cuda_success(cudaMemcpy(
          probabilities.data(),
          device_probabilities,
          bytes,
          cudaMemcpyDeviceToHost))) {
    probabilities.clear();
  }
  expect_cuda_success(cudaFree(device_probabilities));
  expect_cuda_success(cudaFree(device_logits));
  return probabilities;
}

TEST(CudaSamplingTest, ArgmaxMatchesHostForBatchedRowsAndTies) {
  constexpr int64_t kRows = 3;
  constexpr int64_t kRowSize = 513;
  std::vector<float> host_values(kRows * kRowSize, -1000.0f);

  host_values[0 * kRowSize + 1] = 8.0f;
  host_values[0 * kRowSize + 257] = 8.0f;
  host_values[1 * kRowSize + 512] = -2.0f;
  host_values[1 * kRowSize + 17] = -3.0f;
  host_values[2 * kRowSize + 256] = 4.0f;
  host_values[2 * kRowSize + 511] = 4.0f;

  float* device_values = nullptr;
  uint64_t* device_indices = nullptr;
  ASSERT_CUDA_SUCCESS(
      cudaMalloc(&device_values, host_values.size() * sizeof(float)));
  ASSERT_CUDA_SUCCESS(cudaMalloc(&device_indices, kRows * sizeof(uint64_t)));
  ASSERT_CUDA_SUCCESS(cudaMemcpy(
      device_values,
      host_values.data(),
      host_values.size() * sizeof(float),
      cudaMemcpyHostToDevice));

  ASSERT_CUDA_SUCCESS(muse_glimmer::cuda::argmax_index(
      device_values, kRows, kRowSize, device_indices, nullptr));
  std::vector<uint64_t> actual(kRows);
  ASSERT_CUDA_SUCCESS(cudaMemcpy(
      actual.data(),
      device_indices,
      actual.size() * sizeof(uint64_t),
      cudaMemcpyDeviceToHost));

  for (int64_t row = 0; row < kRows; ++row) {
    const uint64_t expected = muse_glimmer::argmax_index(
        host_values.data() + row * kRowSize, kRowSize);
    EXPECT_EQ(actual[row], expected) << "row " << row;
  }

  ASSERT_CUDA_SUCCESS(cudaFree(device_indices));
  ASSERT_CUDA_SUCCESS(cudaFree(device_values));
}

TEST(CudaSamplingTest, ArgmaxRejectsInvalidArguments) {
  EXPECT_EQ(
      muse_glimmer::cuda::argmax_index(nullptr, 1, 1, nullptr, nullptr),
      cudaErrorInvalidValue);
}

TEST(CudaSamplingTest, SamplingProbabilitiesMatchHost) {
  constexpr int64_t kRows = 2;
  constexpr int64_t kRowSize = 257;
  std::vector<float> logits(kRows * kRowSize);
  for (int64_t row = 0; row < kRows; ++row) {
    for (int64_t token = 0; token < kRowSize; ++token) {
      logits[row * kRowSize + token] =
          static_cast<float>(((token * 37 + row * 11) % 101) - 50) / 7.0f;
    }
    // Exercise stable token-id tie breaks at the filtering boundary.
    logits[row * kRowSize + 3] = 5.0f;
    logits[row * kRowSize + 19] = 5.0f;
  }

  const std::tuple<double, int32_t, double> configurations[] = {
      {0.5, 0, 1.0},
      {1.0, 7, 1.0},
      {0.8, 0, 0.9},
      {1.7, 31, 0.75},
      {1.0, 1, 0.1},
  };
  muse_glimmer::cuda::SamplingWorkspace workspace;
  for (const auto& [temperature, top_k, top_p] : configurations) {
    const auto actual = cuda_sampling_probabilities(
        logits,
        kRows,
        kRowSize,
        temperature,
        top_k,
        top_p,
        workspace);
    ASSERT_EQ(actual.size(), logits.size());
    for (int64_t row = 0; row < kRows; ++row) {
      const auto expected = muse_glimmer::sampling_probabilities(
          logits.data() + row * kRowSize,
          kRowSize,
          temperature,
          top_k,
          top_p);
      double sum = 0.0;
      for (int64_t token = 0; token < kRowSize; ++token) {
        const float probability = actual[row * kRowSize + token];
        EXPECT_EQ(probability == 0.0f, expected[token] == 0.0f)
            << "token " << token;
        EXPECT_NEAR(probability, expected[token], 2e-5f)
            << "token " << token;
        sum += probability;
      }
      EXPECT_NEAR(sum, 1.0, 2e-5);
    }
  }
}

TEST(CudaSamplingTest, CategoricalSampleMatchesHostDistribution) {
  constexpr int64_t kRows = 20000;
  constexpr int64_t kRowSize = 4;
  constexpr std::array<float, kRowSize> kDistribution = {
      0.15f, 0.5f, 0.05f, 0.3f};
  std::vector<float> probabilities(kRows * kRowSize);
  for (int64_t row = 0; row < kRows; ++row) {
    std::copy(
        kDistribution.begin(),
        kDistribution.end(),
        probabilities.begin() + row * kRowSize);
  }

  float* device_probabilities = nullptr;
  uint64_t* device_tokens = nullptr;
  const size_t probability_bytes = probabilities.size() * sizeof(float);
  ASSERT_CUDA_SUCCESS(cudaMalloc(&device_probabilities, probability_bytes));
  ASSERT_CUDA_SUCCESS(cudaMalloc(&device_tokens, kRows * sizeof(uint64_t)));
  ASSERT_CUDA_SUCCESS(cudaMemcpy(
      device_probabilities,
      probabilities.data(),
      probability_bytes,
      cudaMemcpyHostToDevice));

  muse_glimmer::cuda::SamplingWorkspace workspace;
  ASSERT_CUDA_SUCCESS(workspace.reserve(kRows, kRowSize, nullptr));
  ASSERT_CUDA_SUCCESS(workspace.set_seed(1234, nullptr));
  ASSERT_CUDA_SUCCESS(muse_glimmer::cuda::categorical_sample(
      device_probabilities,
      kRows,
      kRowSize,
      device_tokens,
      workspace,
      nullptr));
  std::vector<uint64_t> first(kRows);
  ASSERT_CUDA_SUCCESS(cudaMemcpy(
      first.data(),
      device_tokens,
      first.size() * sizeof(uint64_t),
      cudaMemcpyDeviceToHost));

  ASSERT_CUDA_SUCCESS(workspace.set_seed(1234, nullptr));
  ASSERT_CUDA_SUCCESS(muse_glimmer::cuda::categorical_sample(
      device_probabilities,
      kRows,
      kRowSize,
      device_tokens,
      workspace,
      nullptr));
  std::vector<uint64_t> repeated(kRows);
  ASSERT_CUDA_SUCCESS(cudaMemcpy(
      repeated.data(),
      device_tokens,
      repeated.size() * sizeof(uint64_t),
      cudaMemcpyDeviceToHost));
  EXPECT_EQ(first, repeated);

  std::array<int64_t, kRowSize> cuda_counts{};
  for (const uint64_t token : first) {
    ASSERT_LT(token, kRowSize);
    ++cuda_counts[token];
  }
  std::mt19937 host_rng(1234);
  std::array<int64_t, kRowSize> host_counts{};
  for (int64_t sample = 0; sample < kRows; ++sample) {
    ++host_counts[muse_glimmer::categorical_sample(
        host_rng, kDistribution.data(), kRowSize)];
  }
  for (int64_t token = 0; token < kRowSize; ++token) {
    const double cuda_frequency =
        static_cast<double>(cuda_counts[token]) / kRows;
    const double host_frequency =
        static_cast<double>(host_counts[token]) / kRows;
    EXPECT_NEAR(cuda_frequency, kDistribution[token], 0.015);
    EXPECT_NEAR(cuda_frequency, host_frequency, 0.02);
  }

  ASSERT_CUDA_SUCCESS(cudaFree(device_tokens));
  ASSERT_CUDA_SUCCESS(cudaFree(device_probabilities));
}

TEST(CudaSamplingTest, AcceptanceMatchesHostSemantics) {
  constexpr int64_t kSamplesPerProbability = 10000;
  constexpr std::array<float, 4> kProbabilities = {0.0f, 0.25f, 0.75f, 1.0f};
  std::vector<float> probabilities;
  probabilities.reserve(kSamplesPerProbability * kProbabilities.size());
  for (const float probability : kProbabilities) {
    probabilities.insert(
        probabilities.end(), kSamplesPerProbability, probability);
  }

  float* device_probabilities = nullptr;
  uint8_t* device_accepted = nullptr;
  ASSERT_CUDA_SUCCESS(cudaMalloc(
      &device_probabilities, probabilities.size() * sizeof(float)));
  ASSERT_CUDA_SUCCESS(
      cudaMalloc(&device_accepted, probabilities.size() * sizeof(uint8_t)));
  ASSERT_CUDA_SUCCESS(cudaMemcpy(
      device_probabilities,
      probabilities.data(),
      probabilities.size() * sizeof(float),
      cudaMemcpyHostToDevice));

  muse_glimmer::cuda::SamplingWorkspace workspace;
  ASSERT_CUDA_SUCCESS(workspace.reserve(1, 1, nullptr));
  ASSERT_CUDA_SUCCESS(workspace.set_seed(5678, nullptr));
  ASSERT_CUDA_SUCCESS(muse_glimmer::cuda::accept_with_probability(
      device_probabilities,
      probabilities.size(),
      device_accepted,
      workspace,
      nullptr));
  std::vector<uint8_t> accepted(probabilities.size());
  ASSERT_CUDA_SUCCESS(cudaMemcpy(
      accepted.data(),
      device_accepted,
      accepted.size() * sizeof(uint8_t),
      cudaMemcpyDeviceToHost));

  for (size_t probability_index = 0;
       probability_index < kProbabilities.size();
       ++probability_index) {
    int64_t accepted_count = 0;
    const size_t begin = probability_index * kSamplesPerProbability;
    for (size_t index = begin; index < begin + kSamplesPerProbability; ++index) {
      accepted_count += accepted[index];
    }
    const double frequency =
        static_cast<double>(accepted_count) / kSamplesPerProbability;
    EXPECT_NEAR(frequency, kProbabilities[probability_index], 0.02);
  }

  ASSERT_CUDA_SUCCESS(cudaFree(device_accepted));
  ASSERT_CUDA_SUCCESS(cudaFree(device_probabilities));
}

TEST(CudaSamplingTest, ExcludingTokenMatchesHostSemantics) {
  constexpr int64_t kRows = 12000;
  constexpr int64_t kRowSize = 3;
  const std::array<float, kRowSize> distribution = {0.2f, 0.5f, 0.3f};
  std::vector<float> probabilities(kRows * kRowSize);
  for (int64_t row = 0; row < kRows; ++row) {
    std::copy(
        distribution.begin(),
        distribution.end(),
        probabilities.begin() + row * kRowSize);
  }
  std::vector<uint64_t> excluded(kRows, 1);

  float* device_probabilities = nullptr;
  uint64_t* device_excluded = nullptr;
  uint64_t* device_tokens = nullptr;
  ASSERT_CUDA_SUCCESS(cudaMalloc(
      &device_probabilities, probabilities.size() * sizeof(float)));
  ASSERT_CUDA_SUCCESS(
      cudaMalloc(&device_excluded, excluded.size() * sizeof(uint64_t)));
  ASSERT_CUDA_SUCCESS(
      cudaMalloc(&device_tokens, kRows * sizeof(uint64_t)));
  ASSERT_CUDA_SUCCESS(cudaMemcpy(
      device_probabilities,
      probabilities.data(),
      probabilities.size() * sizeof(float),
      cudaMemcpyHostToDevice));
  ASSERT_CUDA_SUCCESS(cudaMemcpy(
      device_excluded,
      excluded.data(),
      excluded.size() * sizeof(uint64_t),
      cudaMemcpyHostToDevice));

  muse_glimmer::cuda::SamplingWorkspace workspace;
  ASSERT_CUDA_SUCCESS(workspace.reserve(kRows, kRowSize, nullptr));
  ASSERT_CUDA_SUCCESS(workspace.set_seed(9012, nullptr));
  ASSERT_CUDA_SUCCESS(muse_glimmer::cuda::sample_excluding_token_in_place(
      device_probabilities,
      kRows,
      kRowSize,
      device_excluded,
      device_tokens,
      workspace,
      nullptr));
  std::vector<uint64_t> tokens(kRows);
  ASSERT_CUDA_SUCCESS(cudaMemcpy(
      tokens.data(),
      device_tokens,
      tokens.size() * sizeof(uint64_t),
      cudaMemcpyDeviceToHost));
  ASSERT_CUDA_SUCCESS(cudaMemcpy(
      probabilities.data(),
      device_probabilities,
      probabilities.size() * sizeof(float),
      cudaMemcpyDeviceToHost));

  int64_t token_zero_count = 0;
  for (int64_t row = 0; row < kRows; ++row) {
    EXPECT_NEAR(probabilities[row * kRowSize], 0.4f, 1e-6f);
    EXPECT_EQ(probabilities[row * kRowSize + 1], 0.0f);
    EXPECT_NEAR(probabilities[row * kRowSize + 2], 0.6f, 1e-6f);
    ASSERT_NE(tokens[row], 1);
    token_zero_count += tokens[row] == 0;
  }
  EXPECT_NEAR(static_cast<double>(token_zero_count) / kRows, 0.4, 0.02);

  ASSERT_CUDA_SUCCESS(cudaFree(device_tokens));
  ASSERT_CUDA_SUCCESS(cudaFree(device_excluded));
  ASSERT_CUDA_SUCCESS(cudaFree(device_probabilities));
}

TEST(CudaSamplingTest, ResidualCorrectionMatchesHostSemantics) {
  constexpr int64_t kRowsPerCase = 8000;
  constexpr int64_t kRows = 2 * kRowsPerCase;
  constexpr int64_t kRowSize = 3;
  const std::array<float, kRowSize> target_a = {0.5f, 0.3f, 0.2f};
  const std::array<float, kRowSize> draft_a = {0.2f, 0.4f, 0.1f};
  const std::array<float, kRowSize> target_b = {0.1f, 0.6f, 0.3f};
  std::vector<float> target(kRows * kRowSize);
  std::vector<float> draft(kRows * kRowSize);
  for (int64_t row = 0; row < kRows; ++row) {
    const auto& row_target = row < kRowsPerCase ? target_a : target_b;
    const auto& row_draft = row < kRowsPerCase ? draft_a : target_b;
    std::copy(
        row_target.begin(),
        row_target.end(),
        target.begin() + row * kRowSize);
    std::copy(
        row_draft.begin(),
        row_draft.end(),
        draft.begin() + row * kRowSize);
  }

  float* device_target = nullptr;
  float* device_draft = nullptr;
  uint64_t* device_tokens = nullptr;
  const size_t probability_bytes = target.size() * sizeof(float);
  ASSERT_CUDA_SUCCESS(cudaMalloc(&device_target, probability_bytes));
  ASSERT_CUDA_SUCCESS(cudaMalloc(&device_draft, probability_bytes));
  ASSERT_CUDA_SUCCESS(cudaMalloc(&device_tokens, kRows * sizeof(uint64_t)));
  ASSERT_CUDA_SUCCESS(cudaMemcpy(
      device_target,
      target.data(),
      probability_bytes,
      cudaMemcpyHostToDevice));
  ASSERT_CUDA_SUCCESS(cudaMemcpy(
      device_draft,
      draft.data(),
      probability_bytes,
      cudaMemcpyHostToDevice));

  muse_glimmer::cuda::SamplingWorkspace workspace;
  ASSERT_CUDA_SUCCESS(workspace.reserve(kRows, kRowSize, nullptr));
  ASSERT_CUDA_SUCCESS(workspace.set_seed(3456, nullptr));
  ASSERT_CUDA_SUCCESS(muse_glimmer::cuda::sample_from_residual_in_place(
      device_target,
      device_draft,
      kRows,
      kRowSize,
      device_tokens,
      workspace,
      nullptr));
  std::vector<uint64_t> tokens(kRows);
  ASSERT_CUDA_SUCCESS(cudaMemcpy(
      tokens.data(),
      device_tokens,
      tokens.size() * sizeof(uint64_t),
      cudaMemcpyDeviceToHost));
  ASSERT_CUDA_SUCCESS(cudaMemcpy(
      target.data(),
      device_target,
      probability_bytes,
      cudaMemcpyDeviceToHost));

  std::array<int64_t, kRowSize> counts_a{};
  std::array<int64_t, kRowSize> counts_b{};
  for (int64_t row = 0; row < kRows; ++row) {
    ASSERT_LT(tokens[row], kRowSize);
    auto& counts = row < kRowsPerCase ? counts_a : counts_b;
    ++counts[tokens[row]];
  }
  const std::array<float, kRowSize> expected_a = {0.75f, 0.0f, 0.25f};
  for (int64_t token = 0; token < kRowSize; ++token) {
    EXPECT_NEAR(target[token], expected_a[token], 1e-6f);
    EXPECT_NEAR(
        static_cast<double>(counts_a[token]) / kRowsPerCase,
        expected_a[token],
        0.025);
    EXPECT_NEAR(target[kRowsPerCase * kRowSize + token], target_b[token], 1e-6f);
    EXPECT_NEAR(
        static_cast<double>(counts_b[token]) / kRowsPerCase,
        target_b[token],
        0.025);
  }

  ASSERT_CUDA_SUCCESS(cudaFree(device_tokens));
  ASSERT_CUDA_SUCCESS(cudaFree(device_draft));
  ASSERT_CUDA_SUCCESS(cudaFree(device_target));
}

TEST(CudaSamplingTest, SampleTokenMatchesHostModes) {
  constexpr int64_t kRows = 2;
  constexpr int64_t kRowSize = 5;
  const std::vector<float> logits = {
      -2.0f, 4.0f, 1.0f, 4.0f, 0.0f,
      3.0f, -1.0f, 2.0f, 0.5f, -4.0f,
  };
  float* device_logits = nullptr;
  float* device_probabilities = nullptr;
  uint64_t* device_tokens = nullptr;
  ASSERT_CUDA_SUCCESS(cudaMalloc(&device_logits, logits.size() * sizeof(float)));
  ASSERT_CUDA_SUCCESS(cudaMalloc(
      &device_probabilities, logits.size() * sizeof(float)));
  ASSERT_CUDA_SUCCESS(cudaMalloc(&device_tokens, kRows * sizeof(uint64_t)));
  ASSERT_CUDA_SUCCESS(cudaMemcpy(
      device_logits,
      logits.data(),
      logits.size() * sizeof(float),
      cudaMemcpyHostToDevice));

  muse_glimmer::cuda::SamplingWorkspace workspace;
  ASSERT_CUDA_SUCCESS(workspace.reserve(kRows, kRowSize, nullptr));
  ASSERT_CUDA_SUCCESS(muse_glimmer::cuda::sample_token(
      device_logits,
      kRows,
      kRowSize,
      0.0,
      0,
      1.0,
      device_tokens,
      nullptr,
      false,
      workspace,
      nullptr));
  std::vector<uint64_t> tokens(kRows);
  ASSERT_CUDA_SUCCESS(cudaMemcpy(
      tokens.data(),
      device_tokens,
      tokens.size() * sizeof(uint64_t),
      cudaMemcpyDeviceToHost));
  for (int64_t row = 0; row < kRows; ++row) {
    EXPECT_EQ(
        tokens[row],
        muse_glimmer::argmax_index(logits.data() + row * kRowSize, kRowSize));
  }

  ASSERT_CUDA_SUCCESS(muse_glimmer::cuda::sample_token(
      device_logits,
      kRows,
      kRowSize,
      0.8,
      3,
      0.7,
      device_tokens,
      device_probabilities,
      true,
      workspace,
      nullptr));
  std::vector<float> probabilities(logits.size());
  ASSERT_CUDA_SUCCESS(cudaMemcpy(
      probabilities.data(),
      device_probabilities,
      probabilities.size() * sizeof(float),
      cudaMemcpyDeviceToHost));
  for (int64_t row = 0; row < kRows; ++row) {
    const auto expected = muse_glimmer::sampling_probabilities(
        logits.data() + row * kRowSize, kRowSize, 0.8, 3, 0.7);
    for (int64_t token = 0; token < kRowSize; ++token) {
      EXPECT_NEAR(
          probabilities[row * kRowSize + token], expected[token], 2e-5f);
    }
  }

  ASSERT_CUDA_SUCCESS(cudaFree(device_tokens));
  ASSERT_CUDA_SUCCESS(cudaFree(device_probabilities));
  ASSERT_CUDA_SUCCESS(cudaFree(device_logits));
}

TEST(CudaSamplingTest, GreedySpeculativeSamplingMatchesHostFlow) {
  constexpr int64_t kVerifyLength = 4;
  const std::array<uint64_t, kVerifyLength> target_tokens = {11, 12, 13, 14};
  const std::array<uint64_t, kVerifyLength> rejected_candidates = {
      10, 11, 99, 13};
  const std::array<uint64_t, kVerifyLength> accepted_candidates = {
      10, 11, 12, 13};

  uint64_t* device_target_tokens = nullptr;
  uint64_t* device_candidates = nullptr;
  int64_t* device_accepted_count = nullptr;
  uint64_t* device_correction_token = nullptr;
  ASSERT_CUDA_SUCCESS(
      cudaMalloc(&device_target_tokens, target_tokens.size() * sizeof(uint64_t)));
  ASSERT_CUDA_SUCCESS(
      cudaMalloc(&device_candidates, rejected_candidates.size() * sizeof(uint64_t)));
  ASSERT_CUDA_SUCCESS(cudaMalloc(&device_accepted_count, sizeof(int64_t)));
  ASSERT_CUDA_SUCCESS(cudaMalloc(&device_correction_token, sizeof(uint64_t)));
  ASSERT_CUDA_SUCCESS(cudaMemcpy(
      device_target_tokens,
      target_tokens.data(),
      target_tokens.size() * sizeof(uint64_t),
      cudaMemcpyHostToDevice));

  auto verify = [&](const auto& candidates,
                    int64_t expected_count,
                    uint64_t expected_correction) {
    ASSERT_CUDA_SUCCESS(cudaMemcpy(
        device_candidates,
        candidates.data(),
        candidates.size() * sizeof(uint64_t),
        cudaMemcpyHostToDevice));
    ASSERT_CUDA_SUCCESS(muse_glimmer::cuda::greedy_speculative_sample(
        device_target_tokens,
        device_candidates,
        kVerifyLength,
        device_accepted_count,
        device_correction_token,
        nullptr));
    int64_t accepted_count = 0;
    uint64_t correction_token = 0;
    ASSERT_CUDA_SUCCESS(cudaMemcpy(
        &accepted_count,
        device_accepted_count,
        sizeof(int64_t),
        cudaMemcpyDeviceToHost));
    ASSERT_CUDA_SUCCESS(cudaMemcpy(
        &correction_token,
        device_correction_token,
        sizeof(uint64_t),
        cudaMemcpyDeviceToHost));
    EXPECT_EQ(accepted_count, expected_count);
    EXPECT_EQ(correction_token, expected_correction);
  };
  verify(rejected_candidates, 2, 12);
  verify(accepted_candidates, 4, 14);

  ASSERT_CUDA_SUCCESS(cudaFree(device_correction_token));
  ASSERT_CUDA_SUCCESS(cudaFree(device_accepted_count));
  ASSERT_CUDA_SUCCESS(cudaFree(device_candidates));
  ASSERT_CUDA_SUCCESS(cudaFree(device_target_tokens));
}

TEST(CudaSamplingTest, StochasticSpeculativeSamplingHandlesAcceptAndReject) {
  constexpr int64_t kVerifyLength = 4;
  constexpr int64_t kRowSize = 4;
  const std::array<uint64_t, kVerifyLength> candidates = {0, 1, 2, 3};
  const std::array<float, kVerifyLength * kRowSize> all_accepted_target = {
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
      0, 0, 0, 1,
  };
  const std::array<float, kVerifyLength * kRowSize> all_accepted_draft = {
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
  };
  auto rejected_target = all_accepted_target;
  rejected_target[0] = 1.0f;
  rejected_target[1] = 0.0f;

  float* device_target = nullptr;
  float* device_draft = nullptr;
  uint64_t* device_candidates = nullptr;
  int64_t* device_accepted_count = nullptr;
  uint64_t* device_correction_token = nullptr;
  const size_t probabilities_bytes = all_accepted_target.size() * sizeof(float);
  ASSERT_CUDA_SUCCESS(cudaMalloc(&device_target, probabilities_bytes));
  ASSERT_CUDA_SUCCESS(cudaMalloc(&device_draft, probabilities_bytes));
  ASSERT_CUDA_SUCCESS(
      cudaMalloc(&device_candidates, candidates.size() * sizeof(uint64_t)));
  ASSERT_CUDA_SUCCESS(cudaMalloc(&device_accepted_count, sizeof(int64_t)));
  ASSERT_CUDA_SUCCESS(cudaMalloc(&device_correction_token, sizeof(uint64_t)));
  ASSERT_CUDA_SUCCESS(cudaMemcpy(
      device_candidates,
      candidates.data(),
      candidates.size() * sizeof(uint64_t),
      cudaMemcpyHostToDevice));
  ASSERT_CUDA_SUCCESS(cudaMemcpy(
      device_draft,
      all_accepted_draft.data(),
      probabilities_bytes,
      cudaMemcpyHostToDevice));

  muse_glimmer::cuda::SamplingWorkspace workspace;
  ASSERT_CUDA_SUCCESS(workspace.reserve(kVerifyLength, kRowSize, nullptr));
  ASSERT_CUDA_SUCCESS(workspace.set_seed(7890, nullptr));
  auto verify = [&](const auto& target,
                    int64_t expected_count,
                    uint64_t expected_correction) {
    ASSERT_CUDA_SUCCESS(cudaMemcpy(
        device_target,
        target.data(),
        probabilities_bytes,
        cudaMemcpyHostToDevice));
    ASSERT_CUDA_SUCCESS(muse_glimmer::cuda::stochastic_speculative_sample(
        device_target,
        device_draft,
        device_candidates,
        kVerifyLength,
        kRowSize,
        false,
        device_accepted_count,
        device_correction_token,
        workspace,
        nullptr));
    int64_t accepted_count = 0;
    uint64_t correction_token = 0;
    ASSERT_CUDA_SUCCESS(cudaMemcpy(
        &accepted_count,
        device_accepted_count,
        sizeof(int64_t),
        cudaMemcpyDeviceToHost));
    ASSERT_CUDA_SUCCESS(cudaMemcpy(
        &correction_token,
        device_correction_token,
        sizeof(uint64_t),
        cudaMemcpyDeviceToHost));
    EXPECT_EQ(accepted_count, expected_count);
    EXPECT_EQ(correction_token, expected_correction);
  };
  verify(all_accepted_target, 4, 3);
  verify(rejected_target, 1, 0);

  ASSERT_CUDA_SUCCESS(cudaFree(device_correction_token));
  ASSERT_CUDA_SUCCESS(cudaFree(device_accepted_count));
  ASSERT_CUDA_SUCCESS(cudaFree(device_candidates));
  ASSERT_CUDA_SUCCESS(cudaFree(device_draft));
  ASSERT_CUDA_SUCCESS(cudaFree(device_target));
}

} // namespace
