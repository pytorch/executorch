/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_runtime.h>
#include <executorch/backends/aoti/common_shims.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/backends/cuda/runtime/shims/tensor_attribute.h>
#include <executorch/backends/cuda/runtime/utils.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/platform.h>
#include <gtest/gtest.h>
#include <vector>

using namespace executorch::backends::aoti;
using namespace executorch::backends::cuda;
using namespace executorch::runtime;
using executorch::runtime::etensor::Tensor;

// Test fixture for aoti_torch_item_bool tests
class AOTITorchItemBoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize ExecuTorch Platform Abstraction Layer
    et_pal_init();

    // Check if CUDA is available
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "CUDA not available, skipping CUDA tests";
    }

    // Clean up any existing cached metadata before each test
    cleanup_tensor_metadata();

    // Clear any remaining tensors from previous tests
    clear_all_tensors();
  }

  void TearDown() override {
    // Clean up metadata
    cleanup_tensor_metadata();

    // Clear the global tensor storage using the provided function
    clear_all_tensors();
  }

  // Helper to create a bool tensor on CUDA with a specific value
  Tensor* create_cuda_bool_tensor(bool value) {
    // Create a 0D (scalar) bool tensor
    std::vector<int64_t> sizes = {}; // 0D tensor
    std::vector<int64_t> strides = {}; // Empty strides for scalar
    Tensor* tensor;

    AOTITorchError error = aoti_torch_empty_strided(
        sizes.size(),
        sizes.data(),
        strides.data(),
        static_cast<int32_t>(SupportedDTypes::BOOL),
        static_cast<int32_t>(SupportedDevices::CUDA),
        0,
        &tensor);

    if (error != Error::Ok || tensor == nullptr) {
      return nullptr;
    }

    // Set the value
    bool host_value = value;
    cudaError_t cuda_err = cudaMemcpy(
        tensor->mutable_data_ptr(),
        &host_value,
        sizeof(bool),
        cudaMemcpyHostToDevice);

    if (cuda_err != cudaSuccess) {
      aoti_torch_delete_tensor_object(tensor);
      return nullptr;
    }

    return tensor;
  }

  // Helper to create a bool tensor on CPU with a specific value
  Tensor* create_cpu_bool_tensor(bool value) {
    // Create a 0D (scalar) bool tensor
    std::vector<int64_t> sizes = {}; // 0D tensor
    std::vector<int64_t> strides = {}; // Empty strides for scalar
    Tensor* tensor;

    AOTITorchError error = aoti_torch_empty_strided(
        sizes.size(),
        sizes.data(),
        strides.data(),
        static_cast<int32_t>(SupportedDTypes::BOOL),
        static_cast<int32_t>(SupportedDevices::CPU),
        0,
        &tensor);

    if (error != Error::Ok || tensor == nullptr) {
      return nullptr;
    }

    // Set the value directly
    bool* data_ptr = static_cast<bool*>(tensor->mutable_data_ptr());
    *data_ptr = value;

    return tensor;
  }
};

// Test extracting true value from CUDA bool tensor
TEST_F(AOTITorchItemBoolTest, CUDATensorTrueValue) {
  Tensor* tensor = create_cuda_bool_tensor(true);
  ASSERT_NE(tensor, nullptr);

  bool result = false;
  AOTITorchError error = aoti_torch_item_bool(tensor, &result);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_TRUE(result);
}

// Test extracting false value from CUDA bool tensor
TEST_F(AOTITorchItemBoolTest, CUDATensorFalseValue) {
  Tensor* tensor = create_cuda_bool_tensor(false);
  ASSERT_NE(tensor, nullptr);

  bool result = true;
  AOTITorchError error = aoti_torch_item_bool(tensor, &result);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_FALSE(result);
}

// Test extracting true value from CPU bool tensor
TEST_F(AOTITorchItemBoolTest, CPUTensorTrueValue) {
  Tensor* tensor = create_cpu_bool_tensor(true);
  ASSERT_NE(tensor, nullptr);

  bool result = false;
  AOTITorchError error = aoti_torch_item_bool(tensor, &result);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_TRUE(result);
}

// Test extracting false value from CPU bool tensor
TEST_F(AOTITorchItemBoolTest, CPUTensorFalseValue) {
  Tensor* tensor = create_cpu_bool_tensor(false);
  ASSERT_NE(tensor, nullptr);

  bool result = true;
  AOTITorchError error = aoti_torch_item_bool(tensor, &result);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_FALSE(result);
}

// Test with null tensor pointer
TEST_F(AOTITorchItemBoolTest, NullTensorPointer) {
  bool result;
  AOTITorchError error = aoti_torch_item_bool(nullptr, &result);
  EXPECT_EQ(error, Error::InvalidArgument);
}

// Test with null result pointer
TEST_F(AOTITorchItemBoolTest, NullResultPointer) {
  Tensor* tensor = create_cuda_bool_tensor(true);
  ASSERT_NE(tensor, nullptr);

  AOTITorchError error = aoti_torch_item_bool(tensor, nullptr);
  EXPECT_EQ(error, Error::InvalidArgument);
}

// Test with non-bool dtype (should fail)
TEST_F(AOTITorchItemBoolTest, NonBoolDtype) {
  // Create a float tensor
  std::vector<int64_t> sizes = {};
  std::vector<int64_t> strides = {};
  Tensor* tensor;

  AOTITorchError error = aoti_torch_empty_strided(
      sizes.size(),
      sizes.data(),
      strides.data(),
      static_cast<int32_t>(SupportedDTypes::FLOAT32), // Not bool
      static_cast<int32_t>(SupportedDevices::CUDA),
      0,
      &tensor);

  ASSERT_EQ(error, Error::Ok);
  ASSERT_NE(tensor, nullptr);

  bool result;
  error = aoti_torch_item_bool(tensor, &result);
  EXPECT_EQ(error, Error::InvalidArgument);
}
