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

// Test fixture for aoti_torch_assign_tensors_out tests
class AOTITorchAssignTensorsOutTest : public ::testing::Test {
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

  // Helper to create a test tensor
  Tensor* create_test_tensor(
      const std::vector<int64_t>& sizes,
      int32_t dtype = static_cast<int32_t>(SupportedDTypes::FLOAT32),
      int32_t device_type = static_cast<int32_t>(SupportedDevices::CUDA)) {
    std::vector<int64_t> strides;
    // Calculate contiguous strides
    if (!sizes.empty()) {
      strides.resize(sizes.size());
      strides[sizes.size() - 1] = 1;
      for (int64_t i = static_cast<int64_t>(sizes.size()) - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * sizes[i + 1];
      }
    }

    Tensor* tensor;
    const int64_t* strides_ptr = strides.empty() ? nullptr : strides.data();

    AOTITorchError error = aoti_torch_empty_strided(
        sizes.size(),
        sizes.data(),
        strides_ptr,
        dtype,
        device_type,
        0,
        &tensor);

    return (error == Error::Ok) ? tensor : nullptr;
  }
};

// Test basic functionality
TEST_F(AOTITorchAssignTensorsOutTest, BasicFunctionality) {
  // Create a source tensor
  std::vector<int64_t> sizes = {2, 3};
  Tensor* src = create_test_tensor(sizes);
  ASSERT_NE(src, nullptr);

  // Create output tensor handle
  Tensor* dst = nullptr;
  AOTITorchError error = aoti_torch_assign_tensors_out(src, &dst);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(dst, nullptr);

  // Verify the output tensor has the same properties as source
  EXPECT_EQ(dst->dim(), src->dim());
  EXPECT_EQ(dst->size(0), src->size(0));
  EXPECT_EQ(dst->size(1), src->size(1));
  EXPECT_EQ(dst->numel(), src->numel());

  // Verify they share the same memory
  EXPECT_EQ(dst->mutable_data_ptr(), src->mutable_data_ptr());
}

// Test with 1D tensor
TEST_F(AOTITorchAssignTensorsOutTest, OneDimensionalTensor) {
  std::vector<int64_t> sizes = {10};
  Tensor* src = create_test_tensor(sizes);
  ASSERT_NE(src, nullptr);

  Tensor* dst = nullptr;
  AOTITorchError error = aoti_torch_assign_tensors_out(src, &dst);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(dst, nullptr);
  EXPECT_EQ(dst->dim(), 1);
  EXPECT_EQ(dst->size(0), 10);
  EXPECT_EQ(dst->mutable_data_ptr(), src->mutable_data_ptr());
}

// Test with 3D tensor
TEST_F(AOTITorchAssignTensorsOutTest, ThreeDimensionalTensor) {
  std::vector<int64_t> sizes = {2, 3, 4};
  Tensor* src = create_test_tensor(sizes);
  ASSERT_NE(src, nullptr);

  Tensor* dst = nullptr;
  AOTITorchError error = aoti_torch_assign_tensors_out(src, &dst);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(dst, nullptr);
  EXPECT_EQ(dst->dim(), 3);
  EXPECT_EQ(dst->size(0), 2);
  EXPECT_EQ(dst->size(1), 3);
  EXPECT_EQ(dst->size(2), 4);
  EXPECT_EQ(dst->mutable_data_ptr(), src->mutable_data_ptr());
}

// Test with scalar (0D) tensor
TEST_F(AOTITorchAssignTensorsOutTest, ScalarTensor) {
  std::vector<int64_t> sizes = {};
  Tensor* src = create_test_tensor(sizes);
  ASSERT_NE(src, nullptr);

  Tensor* dst = nullptr;
  AOTITorchError error = aoti_torch_assign_tensors_out(src, &dst);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(dst, nullptr);
  EXPECT_EQ(dst->dim(), 0);
  EXPECT_EQ(dst->mutable_data_ptr(), src->mutable_data_ptr());
}

// Test with null source pointer
TEST_F(AOTITorchAssignTensorsOutTest, NullSourcePointer) {
  Tensor* dst = nullptr;
  AOTITorchError error = aoti_torch_assign_tensors_out(nullptr, &dst);
  EXPECT_EQ(error, Error::InvalidArgument);
}

// Test with null destination pointer
TEST_F(AOTITorchAssignTensorsOutTest, NullDestinationPointer) {
  std::vector<int64_t> sizes = {2, 3};
  Tensor* src = create_test_tensor(sizes);
  ASSERT_NE(src, nullptr);

  AOTITorchError error = aoti_torch_assign_tensors_out(src, nullptr);
  EXPECT_EQ(error, Error::InvalidArgument);
}

// Test that strides are preserved
TEST_F(AOTITorchAssignTensorsOutTest, StridesPreserved) {
  std::vector<int64_t> sizes = {2, 3};
  Tensor* src = create_test_tensor(sizes);
  ASSERT_NE(src, nullptr);

  Tensor* dst = nullptr;
  AOTITorchError error = aoti_torch_assign_tensors_out(src, &dst);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(dst, nullptr);

  // Get strides from both tensors
  int64_t* src_strides;
  int64_t* dst_strides;
  aoti_torch_get_strides(src, &src_strides);
  aoti_torch_get_strides(dst, &dst_strides);

  // Verify strides match
  for (int64_t i = 0; i < src->dim(); i++) {
    EXPECT_EQ(src_strides[i], dst_strides[i]);
  }
}

// Test with CPU tensor
TEST_F(AOTITorchAssignTensorsOutTest, CPUTensor) {
  std::vector<int64_t> sizes = {2, 3};
  Tensor* src = create_test_tensor(
      sizes,
      static_cast<int32_t>(SupportedDTypes::FLOAT32),
      static_cast<int32_t>(SupportedDevices::CPU));
  ASSERT_NE(src, nullptr);

  Tensor* dst = nullptr;
  AOTITorchError error = aoti_torch_assign_tensors_out(src, &dst);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(dst, nullptr);
  EXPECT_EQ(dst->mutable_data_ptr(), src->mutable_data_ptr());
}

// Test dtype is preserved
TEST_F(AOTITorchAssignTensorsOutTest, DtypePreserved) {
  // Test with different dtypes
  std::vector<int32_t> dtypes = {
      static_cast<int32_t>(SupportedDTypes::FLOAT32),
      static_cast<int32_t>(SupportedDTypes::INT32),
      static_cast<int32_t>(SupportedDTypes::INT64),
  };

  for (int32_t dtype : dtypes) {
    cleanup_tensor_metadata();
    clear_all_tensors();

    std::vector<int64_t> sizes = {2, 3};
    Tensor* src = create_test_tensor(sizes, dtype);
    ASSERT_NE(src, nullptr);

    Tensor* dst = nullptr;
    AOTITorchError error = aoti_torch_assign_tensors_out(src, &dst);

    EXPECT_EQ(error, Error::Ok);
    EXPECT_NE(dst, nullptr);

    // Verify dtype is preserved
    int32_t src_dtype, dst_dtype;
    aoti_torch_get_dtype(src, &src_dtype);
    aoti_torch_get_dtype(dst, &dst_dtype);
    EXPECT_EQ(src_dtype, dst_dtype)
        << "Dtype mismatch for dtype code: " << dtype;
  }
}
