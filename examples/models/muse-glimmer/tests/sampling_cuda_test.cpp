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
#include <vector>

namespace {

#define ASSERT_CUDA_SUCCESS(expression)               \
  do {                                                \
    const cudaError_t error = (expression);           \
    ASSERT_EQ(error, cudaSuccess)                     \
        << cudaGetErrorString(error);                 \
  } while (false)

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

} // namespace
