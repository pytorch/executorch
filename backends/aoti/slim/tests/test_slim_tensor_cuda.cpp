/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cuda_runtime.h>

#include <executorch/backends/aoti/slim/core/SlimTensor.h>
#include <executorch/backends/aoti/slim/cuda/Guard.h>
#include <executorch/backends/aoti/slim/factory/Empty.h>
#include <executorch/backends/aoti/slim/factory/Factory.h>

namespace executorch::backends::aoti::slim {
namespace {

class SlimTensorCUDATest : public ::testing::Test {
 protected:
  void SetUp() override {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "CUDA device not available";
    }
  }
};

TEST_F(SlimTensorCUDATest, EmptyCUDATensorCreation) {
  auto tensor = empty(
      {2, 3, 4},
      executorch::backends::aoti::slim::c10::ScalarType::Float,
      DEFAULT_CUDA_DEVICE);
  EXPECT_EQ(tensor.dim(), 3);
  EXPECT_EQ(tensor.size(0), 2);
  EXPECT_EQ(tensor.size(1), 3);
  EXPECT_EQ(tensor.size(2), 4);
  EXPECT_EQ(tensor.numel(), 24);
  EXPECT_EQ(
      tensor.device().type(),
      executorch::backends::aoti::slim::c10::DeviceType::CUDA);
  EXPECT_TRUE(tensor.is_contiguous());
}

TEST_F(SlimTensorCUDATest, ZerosCUDATensor) {
  auto tensor = zeros(
      {3, 3},
      executorch::backends::aoti::slim::c10::ScalarType::Float,
      DEFAULT_CUDA_DEVICE);
  EXPECT_EQ(tensor.numel(), 9);
  EXPECT_EQ(
      tensor.device().type(),
      executorch::backends::aoti::slim::c10::DeviceType::CUDA);

  std::vector<float> host_data(9);
  cudaMemcpy(
      host_data.data(),
      tensor.data_ptr(),
      9 * sizeof(float),
      cudaMemcpyDeviceToHost);

  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(host_data[i], 0.0f);
  }
}

TEST_F(SlimTensorCUDATest, OnesCUDATensor) {
  auto tensor = ones(
      {2, 2},
      executorch::backends::aoti::slim::c10::ScalarType::Float,
      DEFAULT_CUDA_DEVICE);
  EXPECT_EQ(tensor.numel(), 4);

  std::vector<float> host_data(4);
  cudaMemcpy(
      host_data.data(),
      tensor.data_ptr(),
      4 * sizeof(float),
      cudaMemcpyDeviceToHost);

  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(host_data[i], 1.0f);
  }
}

TEST_F(SlimTensorCUDATest, FillCUDATensor) {
  auto tensor = empty(
      {2, 3},
      executorch::backends::aoti::slim::c10::ScalarType::Float,
      DEFAULT_CUDA_DEVICE);
  tensor.fill_(5.0f);

  std::vector<float> host_data(6);
  cudaMemcpy(
      host_data.data(),
      tensor.data_ptr(),
      6 * sizeof(float),
      cudaMemcpyDeviceToHost);

  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(host_data[i], 5.0f);
  }
}

TEST_F(SlimTensorCUDATest, CloneCUDATensor) {
  auto tensor = empty(
      {2, 3},
      executorch::backends::aoti::slim::c10::ScalarType::Float,
      DEFAULT_CUDA_DEVICE);
  tensor.fill_(3.14f);

  auto cloned = tensor.clone();
  EXPECT_NE(cloned.data_ptr(), tensor.data_ptr());
  EXPECT_EQ(cloned.sizes(), tensor.sizes());
  EXPECT_EQ(cloned.device(), tensor.device());

  std::vector<float> host_data(6);
  cudaMemcpy(
      host_data.data(),
      cloned.data_ptr(),
      6 * sizeof(float),
      cudaMemcpyDeviceToHost);

  for (int i = 0; i < 6; ++i) {
    EXPECT_FLOAT_EQ(host_data[i], 3.14f);
  }
}

TEST_F(SlimTensorCUDATest, CopyCUDAToCUDA) {
  auto src = empty(
      {2, 3},
      executorch::backends::aoti::slim::c10::ScalarType::Float,
      DEFAULT_CUDA_DEVICE);
  src.fill_(2.5f);

  auto dst = empty(
      {2, 3},
      executorch::backends::aoti::slim::c10::ScalarType::Float,
      DEFAULT_CUDA_DEVICE);
  dst.copy_(src);

  std::vector<float> host_data(6);
  cudaMemcpy(
      host_data.data(),
      dst.data_ptr(),
      6 * sizeof(float),
      cudaMemcpyDeviceToHost);

  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(host_data[i], 2.5f);
  }
}

TEST_F(SlimTensorCUDATest, CopyCPUToCUDA) {
  auto cpu_tensor = empty(
      {2, 3},
      executorch::backends::aoti::slim::c10::ScalarType::Float,
      CPU_DEVICE);
  cpu_tensor.fill_(1.5f);

  auto cuda_tensor = empty(
      {2, 3},
      executorch::backends::aoti::slim::c10::ScalarType::Float,
      DEFAULT_CUDA_DEVICE);
  cuda_tensor.copy_(cpu_tensor);

  std::vector<float> host_data(6);
  cudaMemcpy(
      host_data.data(),
      cuda_tensor.data_ptr(),
      6 * sizeof(float),
      cudaMemcpyDeviceToHost);

  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(host_data[i], 1.5f);
  }
}

TEST_F(SlimTensorCUDATest, CopyCUDAToCPU) {
  auto cuda_tensor = empty(
      {2, 3},
      executorch::backends::aoti::slim::c10::ScalarType::Float,
      DEFAULT_CUDA_DEVICE);
  cuda_tensor.fill_(4.5f);

  auto cpu_tensor = empty(
      {2, 3},
      executorch::backends::aoti::slim::c10::ScalarType::Float,
      CPU_DEVICE);
  cpu_tensor.copy_(cuda_tensor);

  float* data = static_cast<float*>(cpu_tensor.data_ptr());
  for (int i = 0; i < 6; ++i) {
    EXPECT_EQ(data[i], 4.5f);
  }
}

TEST_F(SlimTensorCUDATest, CUDAGuard) {
  cuda::CUDAGuard guard(0);
  auto tensor = empty(
      {2, 3},
      executorch::backends::aoti::slim::c10::ScalarType::Float,
      DEFAULT_CUDA_DEVICE);
  EXPECT_EQ(
      tensor.device().type(),
      executorch::backends::aoti::slim::c10::DeviceType::CUDA);
}

TEST_F(SlimTensorCUDATest, ReshapeCUDATensor) {
  auto tensor = empty(
      {2, 6},
      executorch::backends::aoti::slim::c10::ScalarType::Float,
      DEFAULT_CUDA_DEVICE);
  auto reshaped = tensor.reshape({3, 4});
  EXPECT_EQ(reshaped.dim(), 2);
  EXPECT_EQ(reshaped.size(0), 3);
  EXPECT_EQ(reshaped.size(1), 4);
  EXPECT_EQ(reshaped.device(), tensor.device());
}

TEST_F(SlimTensorCUDATest, TransposeCUDATensor) {
  auto tensor = empty(
      {2, 3},
      executorch::backends::aoti::slim::c10::ScalarType::Float,
      DEFAULT_CUDA_DEVICE);
  auto transposed = tensor.transpose(0, 1);
  EXPECT_EQ(transposed.size(0), 3);
  EXPECT_EQ(transposed.size(1), 2);
  EXPECT_EQ(transposed.device(), tensor.device());
}

TEST_F(SlimTensorCUDATest, PermuteCUDATensor) {
  auto tensor = empty(
      {2, 3, 4},
      executorch::backends::aoti::slim::c10::ScalarType::Float,
      DEFAULT_CUDA_DEVICE);
  auto permuted = tensor.permute({2, 0, 1});
  EXPECT_EQ(permuted.size(0), 4);
  EXPECT_EQ(permuted.size(1), 2);
  EXPECT_EQ(permuted.size(2), 3);
  EXPECT_EQ(permuted.device(), tensor.device());
}

} // namespace
} // namespace executorch::backends::aoti::slim
