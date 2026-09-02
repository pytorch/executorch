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

#include <cstdint>
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

} // namespace
