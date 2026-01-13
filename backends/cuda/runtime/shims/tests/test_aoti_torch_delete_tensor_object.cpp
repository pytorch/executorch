/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

#include <executorch/backends/aoti/slim/c10/core/Device.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/cuda/runtime/shims/memory_slim.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/platform.h>

using namespace executorch::backends::cuda;
using executorch::runtime::Error;

namespace slim_c10 = executorch::backends::aoti::slim::c10;

namespace {

bool isCudaAvailable() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  return (err == cudaSuccess && device_count > 0);
}

std::vector<int64_t> calculateContiguousStrides(
    const std::vector<int64_t>& sizes) {
  std::vector<int64_t> strides(sizes.size());
  if (sizes.empty()) {
    return strides;
  }
  strides[sizes.size() - 1] = 1;
  for (int64_t i = static_cast<int64_t>(sizes.size()) - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * sizes[i + 1];
  }
  return strides;
}

} // namespace

class AOTITorchDeleteTensorObjectSlimTest : public ::testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();
  }

  void TearDown() override {
    // SlimTensor uses automatic reference counting - no manual cleanup needed
  }

  Tensor* createTestTensor(
      const std::vector<int64_t>& sizes,
      const std::vector<int64_t>& strides = {},
      int32_t dtype = static_cast<int32_t>(slim_c10::ScalarType::Float),
      int32_t device_type = static_cast<int32_t>(slim_c10::DeviceType::CPU),
      int32_t device_index = 0) {
    Tensor* tensor = nullptr;

    std::vector<int64_t> effective_strides = strides;
    if (strides.empty()) {
      effective_strides = calculateContiguousStrides(sizes);
    }

    AOTITorchError error = aoti_torch_empty_strided(
        sizes.size(),
        sizes.data(),
        effective_strides.data(),
        dtype,
        device_type,
        device_index,
        &tensor);

    return (error == Error::Ok) ? tensor : nullptr;
  }
};

// ============================================================================
// CPU Tests
// ============================================================================

TEST_F(AOTITorchDeleteTensorObjectSlimTest, DeleteCpuTensorBasic) {
  std::vector<int64_t> sizes = {2, 3};
  Tensor* tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(tensor, nullptr);

  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 2);
  EXPECT_EQ(tensor->size(1), 3);

  AOTITorchError error = aoti_torch_delete_tensor_object(tensor);
  EXPECT_EQ(error, Error::Ok);
}

TEST_F(AOTITorchDeleteTensorObjectSlimTest, DeleteNullTensor) {
  AOTITorchError error = aoti_torch_delete_tensor_object(nullptr);
  EXPECT_EQ(error, Error::InvalidArgument);
}

TEST_F(AOTITorchDeleteTensorObjectSlimTest, DeleteMultipleTensors_CPU) {
  std::vector<Tensor*> tensors;

  for (int i = 1; i <= 5; i++) {
    std::vector<int64_t> sizes = {i, i + 1};
    Tensor* tensor = createTestTensor(
        sizes,
        {},
        static_cast<int32_t>(slim_c10::ScalarType::Float),
        static_cast<int32_t>(slim_c10::DeviceType::CPU),
        0);
    ASSERT_NE(tensor, nullptr);
    tensors.push_back(tensor);
  }

  for (Tensor* tensor : tensors) {
    AOTITorchError error = aoti_torch_delete_tensor_object(tensor);
    EXPECT_EQ(error, Error::Ok);
  }
}

TEST_F(AOTITorchDeleteTensorObjectSlimTest, DeleteZeroSizedTensor_CPU) {
  std::vector<int64_t> sizes = {0, 5};
  Tensor* tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(tensor, nullptr);

  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 0);
  EXPECT_EQ(tensor->size(1), 5);
  EXPECT_EQ(tensor->numel(), 0);

  AOTITorchError error = aoti_torch_delete_tensor_object(tensor);
  EXPECT_EQ(error, Error::Ok);
}

TEST_F(AOTITorchDeleteTensorObjectSlimTest, DeleteScalarTensor_CPU) {
  std::vector<int64_t> sizes = {};
  Tensor* tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(tensor, nullptr);

  EXPECT_EQ(tensor->dim(), 0);
  EXPECT_EQ(tensor->numel(), 1);

  AOTITorchError error = aoti_torch_delete_tensor_object(tensor);
  EXPECT_EQ(error, Error::Ok);
}

TEST_F(AOTITorchDeleteTensorObjectSlimTest, DeleteLargeTensor_CPU) {
  std::vector<int64_t> sizes = {10, 20, 30};
  Tensor* tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(tensor, nullptr);

  EXPECT_EQ(tensor->dim(), 3);
  EXPECT_EQ(tensor->numel(), 6000);

  AOTITorchError error = aoti_torch_delete_tensor_object(tensor);
  EXPECT_EQ(error, Error::Ok);
}

TEST_F(AOTITorchDeleteTensorObjectSlimTest, DeleteTensorWithCustomStrides_CPU) {
  std::vector<int64_t> sizes = {3, 4};
  std::vector<int64_t> strides = {1, 3}; // Column-major
  Tensor* tensor = createTestTensor(
      sizes,
      strides,
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(tensor, nullptr);

  EXPECT_EQ(tensor->stride(0), 1);
  EXPECT_EQ(tensor->stride(1), 3);

  AOTITorchError error = aoti_torch_delete_tensor_object(tensor);
  EXPECT_EQ(error, Error::Ok);
}

TEST_F(AOTITorchDeleteTensorObjectSlimTest, DeleteDifferentDtypes_CPU) {
  std::vector<int64_t> sizes = {2, 3};

  // Float
  {
    Tensor* tensor = createTestTensor(
        sizes,
        {},
        static_cast<int32_t>(slim_c10::ScalarType::Float),
        static_cast<int32_t>(slim_c10::DeviceType::CPU),
        0);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(aoti_torch_delete_tensor_object(tensor), Error::Ok);
  }

  // BFloat16
  {
    Tensor* tensor = createTestTensor(
        sizes,
        {},
        static_cast<int32_t>(slim_c10::ScalarType::BFloat16),
        static_cast<int32_t>(slim_c10::DeviceType::CPU),
        0);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(aoti_torch_delete_tensor_object(tensor), Error::Ok);
  }

  // Long
  {
    Tensor* tensor = createTestTensor(
        sizes,
        {},
        static_cast<int32_t>(slim_c10::ScalarType::Long),
        static_cast<int32_t>(slim_c10::DeviceType::CPU),
        0);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(aoti_torch_delete_tensor_object(tensor), Error::Ok);
  }

  // Bool
  {
    Tensor* tensor = createTestTensor(
        sizes,
        {},
        static_cast<int32_t>(slim_c10::ScalarType::Bool),
        static_cast<int32_t>(slim_c10::DeviceType::CPU),
        0);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(aoti_torch_delete_tensor_object(tensor), Error::Ok);
  }
}

// ============================================================================
// CUDA Tests
// ============================================================================

TEST_F(AOTITorchDeleteTensorObjectSlimTest, DeleteCudaTensorBasic) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }

  std::vector<int64_t> sizes = {2, 3};
  Tensor* tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CUDA),
      0);
  ASSERT_NE(tensor, nullptr);

  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_TRUE(tensor->is_cuda());

  AOTITorchError error = aoti_torch_delete_tensor_object(tensor);
  EXPECT_EQ(error, Error::Ok);
}

TEST_F(AOTITorchDeleteTensorObjectSlimTest, DeleteMultipleTensors_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }

  std::vector<Tensor*> tensors;

  for (int i = 1; i <= 5; i++) {
    std::vector<int64_t> sizes = {i, i + 1};
    Tensor* tensor = createTestTensor(
        sizes,
        {},
        static_cast<int32_t>(slim_c10::ScalarType::Float),
        static_cast<int32_t>(slim_c10::DeviceType::CUDA),
        0);
    ASSERT_NE(tensor, nullptr);
    tensors.push_back(tensor);
  }

  for (Tensor* tensor : tensors) {
    AOTITorchError error = aoti_torch_delete_tensor_object(tensor);
    EXPECT_EQ(error, Error::Ok);
  }
}

TEST_F(AOTITorchDeleteTensorObjectSlimTest, DeleteLargeTensor_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }

  std::vector<int64_t> sizes = {100, 100};
  Tensor* tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CUDA),
      0);
  ASSERT_NE(tensor, nullptr);

  AOTITorchError error = aoti_torch_delete_tensor_object(tensor);
  EXPECT_EQ(error, Error::Ok);

  // Verify CUDA state is still good
  cudaError_t cuda_error = cudaGetLastError();
  EXPECT_EQ(cuda_error, cudaSuccess);
}

TEST_F(AOTITorchDeleteTensorObjectSlimTest, DeleteMixedDeviceTensors) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }

  std::vector<int64_t> sizes = {2, 3};

  // Create CUDA tensor
  Tensor* cuda_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CUDA),
      0);
  ASSERT_NE(cuda_tensor, nullptr);
  EXPECT_TRUE(cuda_tensor->is_cuda());

  // Create CPU tensor
  Tensor* cpu_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(cpu_tensor, nullptr);
  EXPECT_TRUE(cpu_tensor->is_cpu());

  // Delete both tensors
  EXPECT_EQ(aoti_torch_delete_tensor_object(cuda_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(cpu_tensor), Error::Ok);
}

// ============================================================================
// Stress Tests
// ============================================================================

TEST_F(
    AOTITorchDeleteTensorObjectSlimTest,
    StressDeletionManySmallTensors_CPU) {
  const int num_tensors = 100;
  std::vector<Tensor*> tensors;

  for (int i = 0; i < num_tensors; i++) {
    std::vector<int64_t> sizes = {1, 1};
    Tensor* tensor = createTestTensor(
        sizes,
        {},
        static_cast<int32_t>(slim_c10::ScalarType::Float),
        static_cast<int32_t>(slim_c10::DeviceType::CPU),
        0);
    if (tensor != nullptr) {
      tensors.push_back(tensor);
    }
  }

  for (Tensor* tensor : tensors) {
    AOTITorchError error = aoti_torch_delete_tensor_object(tensor);
    EXPECT_EQ(error, Error::Ok);
  }
}
