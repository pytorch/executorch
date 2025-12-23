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

using namespace executorch::backends::cuda;
using namespace executorch::backends::aoti;
using namespace executorch::runtime;
using executorch::runtime::etensor::Tensor;

// Test fixture for aoti_torch_empty_strided tests
class AOTITorchEmptyStridedTest : public ::testing::Test {
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

  // Helper to create test tensors
  Tensor* create_tracked_tensor(
      const std::vector<int64_t>& sizes,
      const std::vector<int64_t>& strides = {},
      int32_t dtype = static_cast<int32_t>(SupportedDTypes::FLOAT32),
      int32_t device_type = static_cast<int32_t>(SupportedDevices::CUDA),
      int32_t device_index = 0) {
    Tensor* tensor;

    const int64_t* strides_ptr = strides.empty() ? nullptr : strides.data();

    AOTITorchError error = aoti_torch_empty_strided(
        sizes.size(),
        sizes.data(),
        strides_ptr,
        dtype,
        device_type,
        device_index,
        &tensor);

    return (error == Error::Ok) ? tensor : nullptr;
  }
};

// Test aoti_torch_empty_strided basic functionality
TEST_F(AOTITorchEmptyStridedTest, BasicFunctionality) {
  // Test 1D tensor
  std::vector<int64_t> sizes_1d = {5};
  Tensor* tensor_1d;
  AOTITorchError error = aoti_torch_empty_strided(
      sizes_1d.size(),
      sizes_1d.data(),
      nullptr, // Let function compute strides
      static_cast<int32_t>(SupportedDTypes::FLOAT32),
      static_cast<int32_t>(SupportedDevices::CUDA),
      0, // device index
      &tensor_1d);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(tensor_1d, nullptr);

  // CRITICAL: Verify the tensor is actually float32
  int32_t actual_dtype;
  EXPECT_EQ(aoti_torch_get_dtype(tensor_1d, &actual_dtype), Error::Ok);
  EXPECT_EQ(actual_dtype, static_cast<int32_t>(SupportedDTypes::FLOAT32))
      << "Expected float32 dtype ("
      << static_cast<int32_t>(SupportedDTypes::FLOAT32) << "), got "
      << actual_dtype;

  // Verify element size (float32 should be 4 bytes per element)
  size_t element_size = tensor_1d->element_size();
  EXPECT_EQ(element_size, 4)
      << "Expected float32 element size to be 4 bytes, got " << element_size;

  // Verify total number of elements and memory usage
  int64_t expected_numel = 5; // 5 elements
  EXPECT_EQ(tensor_1d->numel(), expected_numel)
      << "Expected " << expected_numel << " elements, got "
      << tensor_1d->numel();

  // Verify total memory size (numel * element_size)
  size_t expected_memory_size = expected_numel * 4; // 5 * 4 = 20 bytes
  size_t actual_memory_size = tensor_1d->numel() * tensor_1d->element_size();
  EXPECT_EQ(actual_memory_size, expected_memory_size)
      << "Expected " << expected_memory_size << " bytes, got "
      << actual_memory_size;

  // Check tensor properties
  EXPECT_EQ(tensor_1d->dim(), 1);
  EXPECT_EQ(tensor_1d->size(0), 5);

  // Test 2D tensor with explicit strides
  std::vector<int64_t> sizes_2d = {3, 4};
  std::vector<int64_t> strides_2d = {4, 1};
  Tensor* tensor_2d;
  error = aoti_torch_empty_strided(
      sizes_2d.size(),
      sizes_2d.data(),
      strides_2d.data(),
      static_cast<int32_t>(SupportedDTypes::FLOAT32),
      static_cast<int32_t>(SupportedDevices::CUDA),
      0, // device index
      &tensor_2d);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(tensor_2d, nullptr);

  // Verify 2D tensor is also float32
  int32_t dtype_2d;
  EXPECT_EQ(aoti_torch_get_dtype(tensor_2d, &dtype_2d), Error::Ok);
  EXPECT_EQ(dtype_2d, static_cast<int32_t>(SupportedDTypes::FLOAT32))
      << "Expected float32 dtype ("
      << static_cast<int32_t>(SupportedDTypes::FLOAT32) << "), got "
      << dtype_2d;

  // Verify element size for 2D tensor
  EXPECT_EQ(tensor_2d->element_size(), 4);

  // Check tensor properties
  EXPECT_EQ(tensor_2d->dim(), 2);
  EXPECT_EQ(tensor_2d->size(0), 3);
  EXPECT_EQ(tensor_2d->size(1), 4);

  // Verify memory size for 2D tensor
  int64_t expected_numel_2d = 3 * 4; // 12 elements
  size_t expected_memory_2d = expected_numel_2d * 4; // 12 * 4 = 48 bytes
  EXPECT_EQ(tensor_2d->numel() * tensor_2d->element_size(), expected_memory_2d);
}

// Test aoti_torch_empty_strided with CPU device
TEST_F(AOTITorchEmptyStridedTest, CPUDevice) {
  std::vector<int64_t> sizes = {2, 3};
  Tensor* tensor;
  AOTITorchError error = aoti_torch_empty_strided(
      sizes.size(),
      sizes.data(),
      nullptr, // Let function compute strides
      static_cast<int32_t>(SupportedDTypes::FLOAT32),
      static_cast<int32_t>(SupportedDevices::CPU),
      0, // device index
      &tensor);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(tensor, nullptr);

  // Check tensor properties
  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 2);
  EXPECT_EQ(tensor->size(1), 3);
}

// Test aoti_torch_empty_strided with invalid dtype
TEST_F(AOTITorchEmptyStridedTest, InvalidDtype) {
  std::vector<int64_t> sizes = {2, 3};
  Tensor* tensor;
  AOTITorchError error = aoti_torch_empty_strided(
      sizes.size(),
      sizes.data(),
      nullptr,
      999, // invalid dtype
      1, // CUDA device
      0, // device index
      &tensor);

  EXPECT_EQ(error, Error::InvalidArgument);
}

// Test aoti_torch_empty_strided with unsupported device
TEST_F(AOTITorchEmptyStridedTest, UnsupportedDevice) {
  std::vector<int64_t> sizes = {2, 3};
  Tensor* tensor;
  AOTITorchError error = aoti_torch_empty_strided(
      sizes.size(),
      sizes.data(),
      nullptr,
      6, // float32
      2, // unsupported device type
      0, // device index
      &tensor);

  EXPECT_EQ(error, Error::NotImplemented);
}

// Test aoti_torch_empty_strided with zero-sized tensor
TEST_F(AOTITorchEmptyStridedTest, ZeroSized) {
  std::vector<int64_t> sizes = {0, 5};
  Tensor* tensor;
  AOTITorchError error = aoti_torch_empty_strided(
      sizes.size(),
      sizes.data(),
      nullptr,
      6, // float32
      1, // CUDA device
      0, // device index
      &tensor);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(tensor, nullptr);

  // Check tensor properties
  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 0);
  EXPECT_EQ(tensor->size(1), 5);
}

// Test aoti_torch_empty_strided scalar tensor (0D)
TEST_F(AOTITorchEmptyStridedTest, Scalar) {
  std::vector<int64_t> sizes = {};
  Tensor* tensor;
  AOTITorchError error = aoti_torch_empty_strided(
      sizes.size(),
      sizes.data(),
      nullptr,
      6, // float32
      1, // CUDA device
      0, // device index
      &tensor);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(tensor, nullptr);

  // Check tensor properties
  EXPECT_EQ(tensor->dim(), 0);
}

// Test aoti_torch_empty_strided with large tensor
TEST_F(AOTITorchEmptyStridedTest, LargeTensor) {
  std::vector<int64_t> sizes = {100, 200, 50};
  Tensor* tensor;
  AOTITorchError error = aoti_torch_empty_strided(
      sizes.size(),
      sizes.data(),
      nullptr,
      6, // float32
      1, // CUDA device
      0, // device index
      &tensor);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(tensor, nullptr);

  // Check tensor properties
  EXPECT_EQ(tensor->dim(), 3);
  EXPECT_EQ(tensor->size(0), 100);
  EXPECT_EQ(tensor->size(1), 200);
  EXPECT_EQ(tensor->size(2), 50);
}

// Test aoti_torch_empty_strided with bfloat16 dtype
TEST_F(AOTITorchEmptyStridedTest, BFloat16Tensor) {
  // Test creating bfloat16 tensor on CUDA
  std::vector<int64_t> sizes = {2, 3, 4};
  Tensor* tensor_bf16;
  AOTITorchError error = aoti_torch_empty_strided(
      sizes.size(),
      sizes.data(),
      nullptr, // Let function compute strides
      static_cast<int32_t>(SupportedDTypes::BFLOAT16),
      static_cast<int32_t>(SupportedDevices::CUDA),
      0, // device index
      &tensor_bf16);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(tensor_bf16, nullptr);

  // CRITICAL: Verify the tensor is actually bfloat16
  int32_t actual_dtype;
  EXPECT_EQ(aoti_torch_get_dtype(tensor_bf16, &actual_dtype), Error::Ok);
  EXPECT_EQ(actual_dtype, static_cast<int32_t>(SupportedDTypes::BFLOAT16))
      << "Expected bfloat16 dtype ("
      << static_cast<int32_t>(SupportedDTypes::BFLOAT16) << "), got "
      << actual_dtype;

  // Verify element size (bfloat16 should be 2 bytes per element)
  size_t element_size = tensor_bf16->element_size();
  EXPECT_EQ(element_size, 2)
      << "Expected bfloat16 element size to be 2 bytes, got " << element_size;

  // Verify total number of elements and memory usage
  int64_t expected_numel = 2 * 3 * 4; // 24 elements
  EXPECT_EQ(tensor_bf16->numel(), expected_numel)
      << "Expected " << expected_numel << " elements, got "
      << tensor_bf16->numel();

  // Verify total memory size (numel * element_size)
  size_t expected_memory_size = expected_numel * 2; // 24 * 2 = 48 bytes
  size_t actual_memory_size =
      tensor_bf16->numel() * tensor_bf16->element_size();
  EXPECT_EQ(actual_memory_size, expected_memory_size)
      << "Expected " << expected_memory_size << " bytes, got "
      << actual_memory_size;

  // Check tensor properties
  EXPECT_EQ(tensor_bf16->dim(), 3);
  EXPECT_EQ(tensor_bf16->size(0), 2);
  EXPECT_EQ(tensor_bf16->size(1), 3);
  EXPECT_EQ(tensor_bf16->size(2), 4);

  // Verify we can get tensor metadata
  int64_t* sizes_ptr;
  int64_t* strides_ptr;
  EXPECT_EQ(aoti_torch_get_sizes(tensor_bf16, &sizes_ptr), Error::Ok);
  EXPECT_EQ(aoti_torch_get_strides(tensor_bf16, &strides_ptr), Error::Ok);

  // Check sizes match
  EXPECT_EQ(sizes_ptr[0], 2);
  EXPECT_EQ(sizes_ptr[1], 3);
  EXPECT_EQ(sizes_ptr[2], 4);

  // Check that strides are computed correctly (row-major order)
  EXPECT_EQ(strides_ptr[0], 12); // 3 * 4
  EXPECT_EQ(strides_ptr[1], 4); // 4
  EXPECT_EQ(strides_ptr[2], 1); // 1

  // Test bfloat16 tensor with custom strides
  std::vector<int64_t> sizes_2d = {3, 2};
  std::vector<int64_t> strides_2d = {2, 1}; // Row-major strides
  Tensor* tensor_bf16_custom;
  error = aoti_torch_empty_strided(
      sizes_2d.size(),
      sizes_2d.data(),
      strides_2d.data(),
      static_cast<int32_t>(SupportedDTypes::BFLOAT16),
      static_cast<int32_t>(SupportedDevices::CUDA),
      0, // device index
      &tensor_bf16_custom);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(tensor_bf16_custom, nullptr);

  // Verify custom stride tensor is also bfloat16
  int32_t custom_dtype;
  EXPECT_EQ(aoti_torch_get_dtype(tensor_bf16_custom, &custom_dtype), Error::Ok);
  EXPECT_EQ(custom_dtype, static_cast<int32_t>(SupportedDTypes::BFLOAT16))
      << "Expected bfloat16 dtype ("
      << static_cast<int32_t>(SupportedDTypes::BFLOAT16) << "), got "
      << custom_dtype;

  // Verify element size for custom stride tensor
  EXPECT_EQ(tensor_bf16_custom->element_size(), 2);

  // Check tensor properties
  EXPECT_EQ(tensor_bf16_custom->dim(), 2);
  EXPECT_EQ(tensor_bf16_custom->size(0), 3);
  EXPECT_EQ(tensor_bf16_custom->size(1), 2);

  // Verify memory size for custom stride tensor
  int64_t custom_expected_numel = 3 * 2; // 6 elements
  size_t custom_expected_memory = custom_expected_numel * 2; // 6 * 2 = 12 bytes
  EXPECT_EQ(
      tensor_bf16_custom->numel() * tensor_bf16_custom->element_size(),
      custom_expected_memory);

  // Check custom strides
  int64_t* custom_strides_ptr;
  EXPECT_EQ(
      aoti_torch_get_strides(tensor_bf16_custom, &custom_strides_ptr),
      Error::Ok);
  EXPECT_EQ(custom_strides_ptr[0], 2);
  EXPECT_EQ(custom_strides_ptr[1], 1);

  // Test bfloat16 scalar tensor (0D)
  std::vector<int64_t> scalar_sizes = {};
  Tensor* tensor_bf16_scalar;
  error = aoti_torch_empty_strided(
      scalar_sizes.size(),
      scalar_sizes.data(),
      nullptr,
      static_cast<int32_t>(SupportedDTypes::BFLOAT16),
      static_cast<int32_t>(SupportedDevices::CUDA),
      0, // device index
      &tensor_bf16_scalar);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(tensor_bf16_scalar, nullptr);
  EXPECT_EQ(tensor_bf16_scalar->dim(), 0);

  // Verify scalar tensor is also bfloat16
  int32_t scalar_dtype;
  EXPECT_EQ(aoti_torch_get_dtype(tensor_bf16_scalar, &scalar_dtype), Error::Ok);
  EXPECT_EQ(scalar_dtype, static_cast<int32_t>(SupportedDTypes::BFLOAT16))
      << "Expected bfloat16 dtype ("
      << static_cast<int32_t>(SupportedDTypes::BFLOAT16) << "), got "
      << scalar_dtype;

  // Verify scalar tensor properties
  EXPECT_EQ(tensor_bf16_scalar->element_size(), 2);
  EXPECT_EQ(tensor_bf16_scalar->numel(), 1); // Scalar tensor has 1 element
  EXPECT_EQ(
      tensor_bf16_scalar->numel() * tensor_bf16_scalar->element_size(),
      2); // 1 * 2 = 2 bytes
}

// Test custom strides functionality
TEST_F(AOTITorchEmptyStridedTest, CustomStrides) {
  // Create tensor with valid custom strides (contiguous layout)
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = {3, 1}; // Standard row-major strides

  Tensor* tensor = create_tracked_tensor(sizes, strides);
  EXPECT_NE(tensor, nullptr);

  // Verify the tensor was created correctly
  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 2);
  EXPECT_EQ(tensor->size(1), 3);

  // Check strides through AOTI interface
  int64_t* strides_ptr;
  EXPECT_EQ(aoti_torch_get_strides(tensor, &strides_ptr), Error::Ok);
  EXPECT_EQ(strides_ptr[0], 3);
  EXPECT_EQ(strides_ptr[1], 1);

  // Test another valid stride pattern - transpose-like
  std::vector<int64_t> sizes_2 = {3, 2};
  std::vector<int64_t> strides_2 = {1, 3}; // Column-major strides

  Tensor* tensor_2 = create_tracked_tensor(sizes_2, strides_2);
  EXPECT_NE(tensor_2, nullptr);

  // Verify the tensor properties
  EXPECT_EQ(tensor_2->dim(), 2);
  EXPECT_EQ(tensor_2->size(0), 3);
  EXPECT_EQ(tensor_2->size(1), 2);

  // Check strides
  int64_t* strides_ptr_2;
  EXPECT_EQ(aoti_torch_get_strides(tensor_2, &strides_ptr_2), Error::Ok);
  EXPECT_EQ(strides_ptr_2[0], 1);
  EXPECT_EQ(strides_ptr_2[1], 3);
}

// Test edge case: zero-element tensor with non-zero dimensions
TEST_F(AOTITorchEmptyStridedTest, ZeroElementTensor) {
  std::vector<int64_t> sizes = {2, 0, 3}; // Total elements = 0
  Tensor* tensor = create_tracked_tensor(sizes);
  EXPECT_NE(tensor, nullptr);

  // Verify the tensor properties
  EXPECT_EQ(tensor->dim(), 3);
  EXPECT_EQ(tensor->size(0), 2);
  EXPECT_EQ(tensor->size(1), 0);
  EXPECT_EQ(tensor->size(2), 3);

  // Should be able to get metadata
  int64_t* sizes_ptr;
  int64_t* strides_ptr;
  EXPECT_EQ(aoti_torch_get_sizes(tensor, &sizes_ptr), Error::Ok);
  EXPECT_EQ(aoti_torch_get_strides(tensor, &strides_ptr), Error::Ok);

  EXPECT_EQ(sizes_ptr[0], 2);
  EXPECT_EQ(sizes_ptr[1], 0);
  EXPECT_EQ(sizes_ptr[2], 3);
}

// Test different data types (currently we support bf16, fp32 and int32)
TEST_F(AOTITorchEmptyStridedTest, DifferentDataTypes) {
  std::vector<int64_t> sizes = {2, 3};

  // Test float32 (dtype 6) - one of the supported types
  Tensor* tensor_float32;
  AOTITorchError error = aoti_torch_empty_strided(
      sizes.size(),
      sizes.data(),
      nullptr,
      6, // float32
      1, // CUDA device
      0, // device index
      &tensor_float32);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(tensor_float32, nullptr);

  // Test int32 (dtype 3) - one of the supported types
  Tensor* tensor_int32;
  error = aoti_torch_empty_strided(
      sizes.size(),
      sizes.data(),
      nullptr,
      3, // int32 - unsupported
      1, // CUDA device
      0, // device index
      &tensor_int32);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(tensor_int32, nullptr);

  // Test another unsupported data type
  Tensor* tensor_float64;
  error = aoti_torch_empty_strided(
      sizes.size(),
      sizes.data(),
      nullptr,
      7, // float64 - unsupported
      1, // CUDA device
      0, // device index
      &tensor_float64);

  EXPECT_EQ(error, Error::InvalidArgument); // Should fail for unsupported dtype
}

// Test multi-dimensional tensors with various shapes
TEST_F(AOTITorchEmptyStridedTest, MultiDimensionalTensors) {
  // Test 3D tensor
  std::vector<int64_t> sizes_3d = {2, 3, 4};
  Tensor* tensor_3d = create_tracked_tensor(sizes_3d);
  EXPECT_NE(tensor_3d, nullptr);
  EXPECT_EQ(tensor_3d->dim(), 3);
  EXPECT_EQ(tensor_3d->size(0), 2);
  EXPECT_EQ(tensor_3d->size(1), 3);
  EXPECT_EQ(tensor_3d->size(2), 4);

  // Test 4D tensor
  std::vector<int64_t> sizes_4d = {2, 3, 4, 5};
  Tensor* tensor_4d = create_tracked_tensor(sizes_4d);
  EXPECT_NE(tensor_4d, nullptr);
  EXPECT_EQ(tensor_4d->dim(), 4);
  EXPECT_EQ(tensor_4d->size(0), 2);
  EXPECT_EQ(tensor_4d->size(1), 3);
  EXPECT_EQ(tensor_4d->size(2), 4);
  EXPECT_EQ(tensor_4d->size(3), 5);

  // Test 5D tensor
  std::vector<int64_t> sizes_5d = {1, 2, 3, 4, 5};
  Tensor* tensor_5d = create_tracked_tensor(sizes_5d);
  EXPECT_NE(tensor_5d, nullptr);
  EXPECT_EQ(tensor_5d->dim(), 5);
  EXPECT_EQ(tensor_5d->size(0), 1);
  EXPECT_EQ(tensor_5d->size(1), 2);
  EXPECT_EQ(tensor_5d->size(2), 3);
  EXPECT_EQ(tensor_5d->size(3), 4);
  EXPECT_EQ(tensor_5d->size(4), 5);
}

// Test incontiguous tensor creation - transpose-like layout
TEST_F(AOTITorchEmptyStridedTest, IncontiguousTransposeLayout) {
  // Create a tensor with transpose-like strides (column-major)
  // For a 3x4 tensor in column-major order, strides should be [1, 3]
  // This means each row step is 1, and each column step is 3
  std::vector<int64_t> sizes = {3, 4};
  std::vector<int64_t> strides = {1, 3}; // Column-major (incontiguous)

  Tensor* tensor;
  AOTITorchError error = aoti_torch_empty_strided(
      sizes.size(),
      sizes.data(),
      strides.data(),
      static_cast<int32_t>(SupportedDTypes::FLOAT32),
      static_cast<int32_t>(SupportedDevices::CUDA),
      0, // device index
      &tensor);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(tensor, nullptr);

  // Verify tensor properties
  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 3);
  EXPECT_EQ(tensor->size(1), 4);

  // Verify the strides are what we specified
  int64_t* strides_ptr;
  EXPECT_EQ(aoti_torch_get_strides(tensor, &strides_ptr), Error::Ok);
  EXPECT_EQ(strides_ptr[0], 1); // Column-major stride for dimension 0
  EXPECT_EQ(strides_ptr[1], 3); // Column-major stride for dimension 1

  // Verify that memory was allocated correctly for incontiguous layout
  // Storage size should be: stride[0] * (size[0] - 1) + stride[1] * (size[1] -
  // 1) + 1 = 1 * (3 - 1) + 3 * (4 - 1) + 1 = 1 * 2 + 3 * 3 + 1 = 2 + 9 + 1 = 12
  // elements Total bytes = 12 * 4 = 48 bytes (for float32)
  EXPECT_EQ(tensor->numel(), 12); // numel is still 3*4=12 for logical shape

  // The tensor should be accessible and writable
  void* data_ptr = tensor->mutable_data_ptr();
  EXPECT_NE(data_ptr, nullptr);

  // Verify we can use CUDA to write to the memory
  std::vector<float> test_data(12, 1.0f);
  cudaError_t cuda_err = cudaMemcpy(
      data_ptr, test_data.data(), 12 * sizeof(float), cudaMemcpyHostToDevice);
  EXPECT_EQ(cuda_err, cudaSuccess);
}

// Test incontiguous tensor creation - expanded/broadcasted stride pattern
TEST_F(AOTITorchEmptyStridedTest, IncontiguousExpandedStrides) {
  // Create a tensor with expanded strides (simulating broadcasting)
  // A 2x3x4 tensor where the first dimension has stride 0 (expanded)
  // This creates a tensor where the first dimension is "broadcasted"
  std::vector<int64_t> sizes = {2, 3, 4};
  std::vector<int64_t> strides = {0, 4, 1}; // First dimension has stride 0

  Tensor* tensor;
  AOTITorchError error = aoti_torch_empty_strided(
      sizes.size(),
      sizes.data(),
      strides.data(),
      static_cast<int32_t>(SupportedDTypes::FLOAT32),
      static_cast<int32_t>(SupportedDevices::CUDA),
      0, // device index
      &tensor);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(tensor, nullptr);

  // Verify tensor properties
  EXPECT_EQ(tensor->dim(), 3);
  EXPECT_EQ(tensor->size(0), 2);
  EXPECT_EQ(tensor->size(1), 3);
  EXPECT_EQ(tensor->size(2), 4);

  // Verify the strides are what we specified
  int64_t* strides_ptr;
  EXPECT_EQ(aoti_torch_get_strides(tensor, &strides_ptr), Error::Ok);
  EXPECT_EQ(strides_ptr[0], 0); // Expanded dimension stride
  EXPECT_EQ(strides_ptr[1], 4);
  EXPECT_EQ(strides_ptr[2], 1);

  // Verify that memory was allocated correctly for this incontiguous layout
  // Storage size should be: stride[0] * (size[0] - 1) + stride[1] * (size[1] -
  // 1) + stride[2] * (size[2] - 1) + 1 = 0 * (2 - 1) + 4 * (3 - 1) + 1 * (4 -
  // 1) + 1 = 0 + 8 + 3 + 1 = 12 elements Note: numel() returns logical number
  // of elements (2*3*4=24), not storage size
  EXPECT_EQ(tensor->numel(), 24); // Logical numel is 2*3*4=24

  // The tensor should be accessible and writable
  void* data_ptr = tensor->mutable_data_ptr();
  EXPECT_NE(data_ptr, nullptr);

  // Verify we can use CUDA to write to the allocated memory
  // We only need to allocate 12 elements (storage size), not 24
  std::vector<float> test_data(12, 2.0f);
  cudaError_t cuda_err = cudaMemcpy(
      data_ptr, test_data.data(), 12 * sizeof(float), cudaMemcpyHostToDevice);
  EXPECT_EQ(cuda_err, cudaSuccess);
}
