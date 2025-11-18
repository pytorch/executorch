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

// Test fixture for aoti_torch_new_tensor_handle tests
class AOTITorchNewTensorHandleTest : public ::testing::Test {
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
  Tensor* create_test_tensor(
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

// Test basic functionality of creating a new tensor handle
TEST_F(AOTITorchNewTensorHandleTest, BasicFunctionality) {
  // Create an original tensor
  std::vector<int64_t> sizes = {2, 3};
  Tensor* orig_tensor = create_test_tensor(sizes);
  ASSERT_NE(orig_tensor, nullptr);

  // Create a new handle from the original tensor
  Tensor* new_tensor;
  AOTITorchError error = aoti_torch_new_tensor_handle(orig_tensor, &new_tensor);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(new_tensor, nullptr);

  // Verify the new tensor has the same properties
  EXPECT_EQ(new_tensor->dim(), orig_tensor->dim());
  EXPECT_EQ(new_tensor->size(0), orig_tensor->size(0));
  EXPECT_EQ(new_tensor->size(1), orig_tensor->size(1));
  EXPECT_EQ(new_tensor->numel(), orig_tensor->numel());

  // Verify they share the same memory
  EXPECT_EQ(new_tensor->mutable_data_ptr(), orig_tensor->mutable_data_ptr());

  // Clean up
  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(new_tensor), Error::Ok);
}

// Test creating new handle from null tensor
TEST_F(AOTITorchNewTensorHandleTest, NullOriginalTensor) {
  Tensor* new_tensor;
  AOTITorchError error = aoti_torch_new_tensor_handle(nullptr, &new_tensor);

  EXPECT_EQ(error, Error::InvalidArgument);
}

// Test passing null pointer for new handle
TEST_F(AOTITorchNewTensorHandleTest, NullNewHandle) {
  std::vector<int64_t> sizes = {2, 3};
  Tensor* orig_tensor = create_test_tensor(sizes);
  ASSERT_NE(orig_tensor, nullptr);

  AOTITorchError error = aoti_torch_new_tensor_handle(orig_tensor, nullptr);

  EXPECT_EQ(error, Error::InvalidArgument);

  // Clean up
  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
}

// Test memory sharing between original and new tensor handle
TEST_F(AOTITorchNewTensorHandleTest, MemorySharing) {
  // Create an original tensor
  std::vector<int64_t> sizes = {3, 4};
  Tensor* orig_tensor = create_test_tensor(sizes);
  ASSERT_NE(orig_tensor, nullptr);

  // Get original memory pointer
  void* orig_ptr = orig_tensor->mutable_data_ptr();
  ASSERT_NE(orig_ptr, nullptr);

  // Create a new handle
  Tensor* new_tensor;
  AOTITorchError error = aoti_torch_new_tensor_handle(orig_tensor, &new_tensor);
  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(new_tensor, nullptr);

  // Verify both tensors point to the same memory
  void* new_ptr = new_tensor->mutable_data_ptr();
  EXPECT_EQ(orig_ptr, new_ptr);

  // Clean up - deleting one should not affect the other's validity
  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);

  // New tensor should still be valid and accessible
  void* still_valid_ptr = new_tensor->mutable_data_ptr();
  EXPECT_EQ(still_valid_ptr, new_ptr);

  EXPECT_EQ(aoti_torch_delete_tensor_object(new_tensor), Error::Ok);
}

// Test creating multiple handles from the same tensor
TEST_F(AOTITorchNewTensorHandleTest, MultipleHandles) {
  // Create an original tensor
  std::vector<int64_t> sizes = {2, 3};
  Tensor* orig_tensor = create_test_tensor(sizes);
  ASSERT_NE(orig_tensor, nullptr);

  void* orig_ptr = orig_tensor->mutable_data_ptr();

  // Create multiple handles
  std::vector<Tensor*> handles;
  const int num_handles = 5;

  for (int i = 0; i < num_handles; i++) {
    Tensor* new_tensor;
    AOTITorchError error =
        aoti_torch_new_tensor_handle(orig_tensor, &new_tensor);
    EXPECT_EQ(error, Error::Ok);
    ASSERT_NE(new_tensor, nullptr);
    EXPECT_EQ(new_tensor->mutable_data_ptr(), orig_ptr);
    handles.push_back(new_tensor);
  }

  // Delete original tensor
  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);

  // All handles should still be valid
  for (Tensor* handle : handles) {
    EXPECT_EQ(handle->mutable_data_ptr(), orig_ptr);
    EXPECT_EQ(handle->dim(), 2);
    EXPECT_EQ(handle->size(0), 2);
    EXPECT_EQ(handle->size(1), 3);
  }

  // Delete all handles
  for (Tensor* handle : handles) {
    EXPECT_EQ(aoti_torch_delete_tensor_object(handle), Error::Ok);
  }
}

// Test creating handle from tensor with custom strides
TEST_F(AOTITorchNewTensorHandleTest, CustomStrides) {
  std::vector<int64_t> sizes = {3, 4};
  std::vector<int64_t> strides = {4, 1}; // Row-major strides
  Tensor* orig_tensor = create_test_tensor(sizes, strides);
  ASSERT_NE(orig_tensor, nullptr);

  // Create new handle
  Tensor* new_tensor;
  AOTITorchError error = aoti_torch_new_tensor_handle(orig_tensor, &new_tensor);
  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(new_tensor, nullptr);

  // Verify strides are preserved
  int64_t* orig_strides_ptr;
  int64_t* new_strides_ptr;
  EXPECT_EQ(aoti_torch_get_strides(orig_tensor, &orig_strides_ptr), Error::Ok);
  EXPECT_EQ(aoti_torch_get_strides(new_tensor, &new_strides_ptr), Error::Ok);

  EXPECT_EQ(orig_strides_ptr[0], new_strides_ptr[0]);
  EXPECT_EQ(orig_strides_ptr[1], new_strides_ptr[1]);

  // Clean up
  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(new_tensor), Error::Ok);
}

// Test creating handle from bfloat16 tensor
TEST_F(AOTITorchNewTensorHandleTest, BFloat16Tensor) {
  std::vector<int64_t> sizes = {2, 3, 4};
  Tensor* orig_tensor = create_test_tensor(
      sizes,
      {},
      static_cast<int32_t>(SupportedDTypes::BFLOAT16),
      static_cast<int32_t>(SupportedDevices::CUDA));
  ASSERT_NE(orig_tensor, nullptr);

  // Verify original is bfloat16
  int32_t orig_dtype;
  EXPECT_EQ(aoti_torch_get_dtype(orig_tensor, &orig_dtype), Error::Ok);
  EXPECT_EQ(orig_dtype, static_cast<int32_t>(SupportedDTypes::BFLOAT16));

  // Create new handle
  Tensor* new_tensor;
  AOTITorchError error = aoti_torch_new_tensor_handle(orig_tensor, &new_tensor);
  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(new_tensor, nullptr);

  // Verify new tensor is also bfloat16
  int32_t new_dtype;
  EXPECT_EQ(aoti_torch_get_dtype(new_tensor, &new_dtype), Error::Ok);
  EXPECT_EQ(new_dtype, static_cast<int32_t>(SupportedDTypes::BFLOAT16));

  // Verify element size (bfloat16 should be 2 bytes)
  EXPECT_EQ(new_tensor->element_size(), 2);

  // Clean up
  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(new_tensor), Error::Ok);
}

// Test creating handle from scalar (0D) tensor
TEST_F(AOTITorchNewTensorHandleTest, ScalarTensor) {
  std::vector<int64_t> sizes = {};
  Tensor* orig_tensor = create_test_tensor(sizes);
  ASSERT_NE(orig_tensor, nullptr);
  EXPECT_EQ(orig_tensor->dim(), 0);

  // Create new handle
  Tensor* new_tensor;
  AOTITorchError error = aoti_torch_new_tensor_handle(orig_tensor, &new_tensor);
  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(new_tensor, nullptr);

  // Verify scalar properties
  EXPECT_EQ(new_tensor->dim(), 0);
  EXPECT_EQ(new_tensor->numel(), 1);

  // Clean up
  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(new_tensor), Error::Ok);
}

// Test creating handle from zero-sized tensor
TEST_F(AOTITorchNewTensorHandleTest, ZeroSizedTensor) {
  std::vector<int64_t> sizes = {0, 5};
  Tensor* orig_tensor = create_test_tensor(sizes);
  ASSERT_NE(orig_tensor, nullptr);
  EXPECT_EQ(orig_tensor->numel(), 0);

  // Attempt to create new handle - should fail because zero-sized tensors have
  // null data pointers
  Tensor* new_tensor = nullptr;
  AOTITorchError error = aoti_torch_new_tensor_handle(orig_tensor, &new_tensor);

  // Zero-sized tensors are not currently supported
  EXPECT_EQ(error, Error::InvalidArgument);
  EXPECT_EQ(new_tensor, nullptr);

  // Clean up original tensor
  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
}

// Test creating handle from large multi-dimensional tensor
TEST_F(AOTITorchNewTensorHandleTest, LargeMultiDimensionalTensor) {
  std::vector<int64_t> sizes = {10, 20, 30};
  Tensor* orig_tensor = create_test_tensor(sizes);
  ASSERT_NE(orig_tensor, nullptr);

  // Create new handle
  Tensor* new_tensor;
  AOTITorchError error = aoti_torch_new_tensor_handle(orig_tensor, &new_tensor);
  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(new_tensor, nullptr);

  // Verify dimensions
  EXPECT_EQ(new_tensor->dim(), 3);
  EXPECT_EQ(new_tensor->size(0), 10);
  EXPECT_EQ(new_tensor->size(1), 20);
  EXPECT_EQ(new_tensor->size(2), 30);
  EXPECT_EQ(new_tensor->numel(), 6000);

  // Clean up
  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(new_tensor), Error::Ok);
}

// Test creating handle preserves tensor metadata
TEST_F(AOTITorchNewTensorHandleTest, MetadataPreservation) {
  std::vector<int64_t> sizes = {2, 3, 4};
  std::vector<int64_t> strides = {12, 4, 1};
  Tensor* orig_tensor = create_test_tensor(
      sizes,
      strides,
      static_cast<int32_t>(SupportedDTypes::FLOAT32),
      static_cast<int32_t>(SupportedDevices::CUDA));
  ASSERT_NE(orig_tensor, nullptr);

  // Create new handle
  Tensor* new_tensor;
  AOTITorchError error = aoti_torch_new_tensor_handle(orig_tensor, &new_tensor);
  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(new_tensor, nullptr);

  // Get and compare all metadata
  int64_t* orig_sizes_ptr;
  int64_t* new_sizes_ptr;
  int64_t* orig_strides_ptr;
  int64_t* new_strides_ptr;
  int32_t orig_dtype, new_dtype;
  int32_t orig_device_type, new_device_type;
  int32_t orig_device_index, new_device_index;

  EXPECT_EQ(aoti_torch_get_sizes(orig_tensor, &orig_sizes_ptr), Error::Ok);
  EXPECT_EQ(aoti_torch_get_sizes(new_tensor, &new_sizes_ptr), Error::Ok);
  EXPECT_EQ(aoti_torch_get_strides(orig_tensor, &orig_strides_ptr), Error::Ok);
  EXPECT_EQ(aoti_torch_get_strides(new_tensor, &new_strides_ptr), Error::Ok);
  EXPECT_EQ(aoti_torch_get_dtype(orig_tensor, &orig_dtype), Error::Ok);
  EXPECT_EQ(aoti_torch_get_dtype(new_tensor, &new_dtype), Error::Ok);
  EXPECT_EQ(
      aoti_torch_get_device_type(orig_tensor, &orig_device_type), Error::Ok);
  EXPECT_EQ(
      aoti_torch_get_device_type(new_tensor, &new_device_type), Error::Ok);
  EXPECT_EQ(
      aoti_torch_get_device_index(orig_tensor, &orig_device_index), Error::Ok);
  EXPECT_EQ(
      aoti_torch_get_device_index(new_tensor, &new_device_index), Error::Ok);

  // Verify all metadata matches
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(orig_sizes_ptr[i], new_sizes_ptr[i]);
    EXPECT_EQ(orig_strides_ptr[i], new_strides_ptr[i]);
  }
  EXPECT_EQ(orig_dtype, new_dtype);
  EXPECT_EQ(orig_device_type, new_device_type);
  EXPECT_EQ(orig_device_index, new_device_index);

  // Clean up
  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(new_tensor), Error::Ok);
}

// Test creating handle chain: orig -> handle1 -> handle2
TEST_F(AOTITorchNewTensorHandleTest, HandleChain) {
  std::vector<int64_t> sizes = {2, 3};
  Tensor* orig_tensor = create_test_tensor(sizes);
  ASSERT_NE(orig_tensor, nullptr);

  void* orig_ptr = orig_tensor->mutable_data_ptr();

  // Create first handle
  Tensor* handle1;
  AOTITorchError error = aoti_torch_new_tensor_handle(orig_tensor, &handle1);
  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(handle1, nullptr);
  EXPECT_EQ(handle1->mutable_data_ptr(), orig_ptr);

  // Create second handle from the first handle
  Tensor* handle2;
  error = aoti_torch_new_tensor_handle(handle1, &handle2);
  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(handle2, nullptr);
  EXPECT_EQ(handle2->mutable_data_ptr(), orig_ptr);

  // Delete in reverse order
  EXPECT_EQ(aoti_torch_delete_tensor_object(handle2), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(handle1), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
}

// Test creating handle and verifying reference counting
TEST_F(AOTITorchNewTensorHandleTest, ReferenceCountingTest) {
  std::vector<int64_t> sizes = {2, 3};
  Tensor* orig_tensor = create_test_tensor(sizes);
  ASSERT_NE(orig_tensor, nullptr);

  void* orig_ptr = orig_tensor->mutable_data_ptr();

  // Create multiple handles
  Tensor* handle1;
  Tensor* handle2;
  Tensor* handle3;

  EXPECT_EQ(aoti_torch_new_tensor_handle(orig_tensor, &handle1), Error::Ok);
  EXPECT_EQ(aoti_torch_new_tensor_handle(orig_tensor, &handle2), Error::Ok);
  EXPECT_EQ(aoti_torch_new_tensor_handle(orig_tensor, &handle3), Error::Ok);

  // Delete original
  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);

  // All handles should still be valid
  EXPECT_EQ(handle1->mutable_data_ptr(), orig_ptr);
  EXPECT_EQ(handle2->mutable_data_ptr(), orig_ptr);
  EXPECT_EQ(handle3->mutable_data_ptr(), orig_ptr);

  // Delete handles one by one
  EXPECT_EQ(aoti_torch_delete_tensor_object(handle1), Error::Ok);

  // Remaining handles should still be valid
  EXPECT_EQ(handle2->mutable_data_ptr(), orig_ptr);
  EXPECT_EQ(handle3->mutable_data_ptr(), orig_ptr);

  EXPECT_EQ(aoti_torch_delete_tensor_object(handle2), Error::Ok);

  // Last handle should still be valid
  EXPECT_EQ(handle3->mutable_data_ptr(), orig_ptr);

  EXPECT_EQ(aoti_torch_delete_tensor_object(handle3), Error::Ok);
}

// Test creating handle from int32 tensor
TEST_F(AOTITorchNewTensorHandleTest, Int32Tensor) {
  std::vector<int64_t> sizes = {2, 3};
  Tensor* orig_tensor = create_test_tensor(
      sizes,
      {},
      3, // int32
      static_cast<int32_t>(SupportedDevices::CUDA));
  ASSERT_NE(orig_tensor, nullptr);

  // Create new handle
  Tensor* new_tensor;
  AOTITorchError error = aoti_torch_new_tensor_handle(orig_tensor, &new_tensor);
  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(new_tensor, nullptr);

  // Verify dtype
  int32_t new_dtype;
  EXPECT_EQ(aoti_torch_get_dtype(new_tensor, &new_dtype), Error::Ok);
  EXPECT_EQ(new_dtype, 3); // int32

  // Clean up
  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(new_tensor), Error::Ok);
}

// Test creating handle with incontiguous tensor (transpose-like layout)
TEST_F(AOTITorchNewTensorHandleTest, IncontiguousTransposeLayout) {
  std::vector<int64_t> sizes = {3, 4};
  std::vector<int64_t> strides = {1, 3}; // Column-major (incontiguous)
  Tensor* orig_tensor = create_test_tensor(sizes, strides);
  ASSERT_NE(orig_tensor, nullptr);

  // Create new handle
  Tensor* new_tensor;
  AOTITorchError error = aoti_torch_new_tensor_handle(orig_tensor, &new_tensor);
  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(new_tensor, nullptr);

  // Verify strides are preserved
  int64_t* new_strides_ptr;
  EXPECT_EQ(aoti_torch_get_strides(new_tensor, &new_strides_ptr), Error::Ok);
  EXPECT_EQ(new_strides_ptr[0], 1);
  EXPECT_EQ(new_strides_ptr[1], 3);

  // Verify both tensors share the same memory
  EXPECT_EQ(new_tensor->mutable_data_ptr(), orig_tensor->mutable_data_ptr());

  // Clean up
  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(new_tensor), Error::Ok);
}

// Test creating handle with expanded strides (broadcasted dimension)
TEST_F(AOTITorchNewTensorHandleTest, ExpandedStrides) {
  std::vector<int64_t> sizes = {2, 3, 4};
  std::vector<int64_t> strides = {0, 4, 1}; // First dimension has stride 0
  Tensor* orig_tensor = create_test_tensor(sizes, strides);
  ASSERT_NE(orig_tensor, nullptr);

  // Create new handle
  Tensor* new_tensor;
  AOTITorchError error = aoti_torch_new_tensor_handle(orig_tensor, &new_tensor);
  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(new_tensor, nullptr);

  // Verify expanded strides are preserved
  int64_t* new_strides_ptr;
  EXPECT_EQ(aoti_torch_get_strides(new_tensor, &new_strides_ptr), Error::Ok);
  EXPECT_EQ(new_strides_ptr[0], 0);
  EXPECT_EQ(new_strides_ptr[1], 4);
  EXPECT_EQ(new_strides_ptr[2], 1);

  // Clean up
  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(new_tensor), Error::Ok);
}

// Stress test: create many handles
TEST_F(AOTITorchNewTensorHandleTest, StressTestManyHandles) {
  std::vector<int64_t> sizes = {2, 3};
  Tensor* orig_tensor = create_test_tensor(sizes);
  ASSERT_NE(orig_tensor, nullptr);

  void* orig_ptr = orig_tensor->mutable_data_ptr();

  // Create many handles
  const int num_handles = 100;
  std::vector<Tensor*> handles;

  for (int i = 0; i < num_handles; i++) {
    Tensor* new_tensor;
    AOTITorchError error =
        aoti_torch_new_tensor_handle(orig_tensor, &new_tensor);
    EXPECT_EQ(error, Error::Ok);
    ASSERT_NE(new_tensor, nullptr);
    EXPECT_EQ(new_tensor->mutable_data_ptr(), orig_ptr);
    handles.push_back(new_tensor);
  }

  // Delete original
  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);

  // All handles should still be valid
  for (Tensor* handle : handles) {
    EXPECT_EQ(handle->mutable_data_ptr(), orig_ptr);
  }

  // Delete all handles
  for (Tensor* handle : handles) {
    EXPECT_EQ(aoti_torch_delete_tensor_object(handle), Error::Ok);
  }
}
