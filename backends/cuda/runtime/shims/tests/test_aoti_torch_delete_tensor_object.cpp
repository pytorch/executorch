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
#include <executorch/backends/cuda/runtime/shims/utils.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/platform.h>
#include <gtest/gtest.h>
#include <vector>

using namespace executorch::backends::aoti;
using namespace executorch::backends::cuda;
using namespace executorch::runtime;
using executorch::runtime::etensor::Tensor;

// Test fixture for aoti_torch_delete_tensor_object tests
class AOTITorchDeleteTensorObjectTest : public ::testing::Test {
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
      int32_t dtype = 6, // float32
      int32_t device_type = 1, // CUDA
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

// Test basic deletion of CUDA tensor
TEST_F(AOTITorchDeleteTensorObjectTest, DeleteCudaTensorBasic) {
  // Create a CUDA tensor
  std::vector<int64_t> sizes = {2, 3};
  Tensor* tensor = create_test_tensor(sizes, {}, 6, 1, 0); // CUDA device
  ASSERT_NE(tensor, nullptr);

  // Verify tensor properties before deletion
  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 2);
  EXPECT_EQ(tensor->size(1), 3);

  // Delete the tensor
  AOTITorchError error = aoti_torch_delete_tensor_object(tensor);
  EXPECT_EQ(error, Error::Ok);
}

// Test basic deletion of CPU tensor
TEST_F(AOTITorchDeleteTensorObjectTest, DeleteCpuTensorBasic) {
  // Create a CPU tensor
  std::vector<int64_t> sizes = {3, 4};
  Tensor* tensor = create_test_tensor(sizes, {}, 6, 0, 0); // CPU device
  ASSERT_NE(tensor, nullptr);

  // Verify tensor properties before deletion
  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 3);
  EXPECT_EQ(tensor->size(1), 4);

  // Delete the tensor
  AOTITorchError error = aoti_torch_delete_tensor_object(tensor);
  EXPECT_EQ(error, Error::Ok);
}

// Test deletion of null tensor pointer
TEST_F(AOTITorchDeleteTensorObjectTest, DeleteNullTensor) {
  AOTITorchError error = aoti_torch_delete_tensor_object(nullptr);
  EXPECT_EQ(error, Error::InvalidArgument);
}

// Test deletion of tensor not in tracking system
TEST_F(AOTITorchDeleteTensorObjectTest, DeleteUntrackedTensor) {
  // Create a tensor and then clear the tracking system
  std::vector<int64_t> sizes = {2, 3};
  Tensor* tensor = create_test_tensor(sizes);
  ASSERT_NE(tensor, nullptr);

  // Clear the tracking system (simulating an untracked tensor)
  clear_all_tensors();

  // Try to delete the tensor - should fail
  AOTITorchError error = aoti_torch_delete_tensor_object(tensor);
  EXPECT_EQ(error, Error::InvalidArgument);
}

// Test deletion of multiple tensors
TEST_F(AOTITorchDeleteTensorObjectTest, DeleteMultipleTensors) {
  // Create multiple tensors
  std::vector<Tensor*> tensors;

  for (int i = 1; i <= 5; i++) {
    std::vector<int64_t> sizes = {i, i + 1};
    Tensor* tensor = create_test_tensor(sizes);
    ASSERT_NE(tensor, nullptr);
    tensors.push_back(tensor);
  }

  // Delete all tensors
  for (Tensor* tensor : tensors) {
    AOTITorchError error = aoti_torch_delete_tensor_object(tensor);
    EXPECT_EQ(error, Error::Ok);
  }
}

// Test deletion of zero-sized tensors
TEST_F(AOTITorchDeleteTensorObjectTest, DeleteZeroSizedTensor) {
  // Create a zero-sized tensor
  std::vector<int64_t> sizes = {0, 5};
  Tensor* tensor = create_test_tensor(sizes);
  ASSERT_NE(tensor, nullptr);

  // Verify tensor properties
  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 0);
  EXPECT_EQ(tensor->size(1), 5);

  // Delete the tensor
  AOTITorchError error = aoti_torch_delete_tensor_object(tensor);
  EXPECT_EQ(error, Error::Ok);
}

// Test deletion of scalar (0D) tensors
TEST_F(AOTITorchDeleteTensorObjectTest, DeleteScalarTensor) {
  // Create a scalar tensor
  std::vector<int64_t> sizes = {};
  Tensor* tensor = create_test_tensor(sizes);
  ASSERT_NE(tensor, nullptr);

  // Verify tensor properties
  EXPECT_EQ(tensor->dim(), 0);

  // Delete the tensor
  AOTITorchError error = aoti_torch_delete_tensor_object(tensor);
  EXPECT_EQ(error, Error::Ok);
}

// Test deletion of large multi-dimensional tensors
TEST_F(AOTITorchDeleteTensorObjectTest, DeleteLargeTensor) {
  // Create a large multi-dimensional tensor
  std::vector<int64_t> sizes = {10, 20, 30};
  Tensor* tensor = create_test_tensor(sizes);
  ASSERT_NE(tensor, nullptr);

  // Verify tensor properties
  EXPECT_EQ(tensor->dim(), 3);
  EXPECT_EQ(tensor->size(0), 10);
  EXPECT_EQ(tensor->size(1), 20);
  EXPECT_EQ(tensor->size(2), 30);

  // Delete the tensor
  AOTITorchError error = aoti_torch_delete_tensor_object(tensor);
  EXPECT_EQ(error, Error::Ok);
}

// Test deletion of tensors with custom strides
TEST_F(AOTITorchDeleteTensorObjectTest, DeleteTensorWithCustomStrides) {
  // Create tensor with custom strides
  std::vector<int64_t> sizes = {3, 4};
  std::vector<int64_t> strides = {4, 1}; // Row-major strides
  Tensor* tensor = create_test_tensor(sizes, strides);
  ASSERT_NE(tensor, nullptr);

  // Verify tensor properties
  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 3);
  EXPECT_EQ(tensor->size(1), 4);

  // Delete the tensor
  AOTITorchError error = aoti_torch_delete_tensor_object(tensor);
  EXPECT_EQ(error, Error::Ok);
}

// Test deletion after accessing tensor data
TEST_F(AOTITorchDeleteTensorObjectTest, DeleteAfterDataAccess) {
  // Create a tensor
  std::vector<int64_t> sizes = {2, 3};
  Tensor* tensor = create_test_tensor(sizes);
  ASSERT_NE(tensor, nullptr);

  // Access tensor data (this should not prevent deletion)
  void* data_ptr = tensor->mutable_data_ptr();
  EXPECT_NE(data_ptr, nullptr);

  // Delete the tensor
  AOTITorchError error = aoti_torch_delete_tensor_object(tensor);
  EXPECT_EQ(error, Error::Ok);
}

// Test double deletion (should fail on second attempt)
TEST_F(AOTITorchDeleteTensorObjectTest, DoubleDeletion) {
  // Create a tensor
  std::vector<int64_t> sizes = {2, 3};
  Tensor* tensor = create_test_tensor(sizes);
  ASSERT_NE(tensor, nullptr);

  // First deletion should succeed
  AOTITorchError error1 = aoti_torch_delete_tensor_object(tensor);
  EXPECT_EQ(error1, Error::Ok);

  // Second deletion should fail (tensor no longer tracked)
  AOTITorchError error2 = aoti_torch_delete_tensor_object(tensor);
  EXPECT_EQ(error2, Error::InvalidArgument);
}

// Test deletion of tensors on both CUDA and CPU devices
TEST_F(AOTITorchDeleteTensorObjectTest, DeleteMixedDeviceTensors) {
  // Create CUDA tensor
  std::vector<int64_t> sizes = {2, 3};
  Tensor* cuda_tensor = create_test_tensor(sizes, {}, 6, 1, 0);
  ASSERT_NE(cuda_tensor, nullptr);

  // Create CPU tensor
  Tensor* cpu_tensor = create_test_tensor(sizes, {}, 6, 0, 0);
  ASSERT_NE(cpu_tensor, nullptr);

  // Delete both tensors
  AOTITorchError cuda_error = aoti_torch_delete_tensor_object(cuda_tensor);
  EXPECT_EQ(cuda_error, Error::Ok);

  AOTITorchError cpu_error = aoti_torch_delete_tensor_object(cpu_tensor);
  EXPECT_EQ(cpu_error, Error::Ok);
}

// Test memory consistency after deletion
TEST_F(AOTITorchDeleteTensorObjectTest, MemoryConsistencyAfterDeletion) {
  // Create multiple tensors
  std::vector<Tensor*> tensors;
  const int num_tensors = 10;

  for (int i = 0; i < num_tensors; i++) {
    std::vector<int64_t> sizes = {i + 1, i + 2};
    Tensor* tensor = create_test_tensor(sizes);
    ASSERT_NE(tensor, nullptr);
    tensors.push_back(tensor);
  }

  // Delete every other tensor
  for (int i = 0; i < num_tensors; i += 2) {
    AOTITorchError error = aoti_torch_delete_tensor_object(tensors[i]);
    EXPECT_EQ(error, Error::Ok);
  }

  // Delete remaining tensors
  for (int i = 1; i < num_tensors; i += 2) {
    AOTITorchError error = aoti_torch_delete_tensor_object(tensors[i]);
    EXPECT_EQ(error, Error::Ok);
  }
}

// Test stress deletion with many small tensors
TEST_F(AOTITorchDeleteTensorObjectTest, StressDeletionManySmallTensors) {
  const int num_tensors = 100;
  std::vector<Tensor*> tensors;

  // Create many small tensors
  for (int i = 0; i < num_tensors; i++) {
    std::vector<int64_t> sizes = {1, 1}; // Minimal size
    Tensor* tensor = create_test_tensor(sizes);
    if (tensor != nullptr) {
      tensors.push_back(tensor);
    }
  }

  // Delete all created tensors
  for (Tensor* tensor : tensors) {
    AOTITorchError error = aoti_torch_delete_tensor_object(tensor);
    EXPECT_EQ(error, Error::Ok);
  }
}

// Test CUDA synchronization during deletion
TEST_F(AOTITorchDeleteTensorObjectTest, CudaSynchronizationDuringDeletion) {
  // Create a larger CUDA tensor to ensure memory allocation
  std::vector<int64_t> sizes = {100, 100};
  Tensor* tensor = create_test_tensor(sizes, {}, 6, 1, 0); // CUDA device
  ASSERT_NE(tensor, nullptr);

  // Delete the tensor (should handle synchronization internally)
  AOTITorchError error = aoti_torch_delete_tensor_object(tensor);
  EXPECT_EQ(error, Error::Ok);

  // Verify CUDA state is still good
  cudaError_t cuda_error = cudaGetLastError();
  EXPECT_EQ(cuda_error, cudaSuccess);
}

// Test specific deletion of bfloat16 tensors
TEST_F(AOTITorchDeleteTensorObjectTest, DeleteBFloat16Tensor) {
  // Test 1D bfloat16 tensor deletion
  std::vector<int64_t> sizes_1d = {10};
  Tensor* tensor_bf16_1d = create_test_tensor(
      sizes_1d,
      {},
      static_cast<int32_t>(SupportedDTypes::BFLOAT16),
      1, // CUDA device
      0);
  ASSERT_NE(tensor_bf16_1d, nullptr);

  // Verify it's bfloat16 before deletion
  int32_t actual_dtype;
  EXPECT_EQ(aoti_torch_get_dtype(tensor_bf16_1d, &actual_dtype), Error::Ok);
  EXPECT_EQ(actual_dtype, static_cast<int32_t>(SupportedDTypes::BFLOAT16))
      << "Expected bfloat16 dtype ("
      << static_cast<int32_t>(SupportedDTypes::BFLOAT16) << "), got "
      << actual_dtype;

  // Verify element size (bfloat16 should be 2 bytes per element)
  EXPECT_EQ(tensor_bf16_1d->element_size(), 2);

  // Delete the bfloat16 tensor
  AOTITorchError error = aoti_torch_delete_tensor_object(tensor_bf16_1d);
  EXPECT_EQ(error, Error::Ok);

  // Test 2D bfloat16 tensor deletion with custom strides
  std::vector<int64_t> sizes_2d = {4, 6};
  std::vector<int64_t> strides_2d = {6, 1}; // Row-major strides
  Tensor* tensor_bf16_2d = create_test_tensor(
      sizes_2d,
      strides_2d,
      static_cast<int32_t>(SupportedDTypes::BFLOAT16),
      1, // CUDA device
      0);
  ASSERT_NE(tensor_bf16_2d, nullptr);

  // Verify tensor properties
  EXPECT_EQ(tensor_bf16_2d->dim(), 2);
  EXPECT_EQ(tensor_bf16_2d->size(0), 4);
  EXPECT_EQ(tensor_bf16_2d->size(1), 6);
  EXPECT_EQ(tensor_bf16_2d->element_size(), 2);

  // Verify it's bfloat16
  int32_t dtype_2d;
  EXPECT_EQ(aoti_torch_get_dtype(tensor_bf16_2d, &dtype_2d), Error::Ok);
  EXPECT_EQ(dtype_2d, static_cast<int32_t>(SupportedDTypes::BFLOAT16));

  // Delete the 2D bfloat16 tensor
  error = aoti_torch_delete_tensor_object(tensor_bf16_2d);
  EXPECT_EQ(error, Error::Ok);

  // Test 3D bfloat16 tensor deletion
  std::vector<int64_t> sizes_3d = {2, 3, 4};
  Tensor* tensor_bf16_3d = create_test_tensor(
      sizes_3d,
      {},
      static_cast<int32_t>(SupportedDTypes::BFLOAT16),
      1, // CUDA device
      0);
  ASSERT_NE(tensor_bf16_3d, nullptr);

  // Verify tensor properties
  EXPECT_EQ(tensor_bf16_3d->dim(), 3);
  EXPECT_EQ(tensor_bf16_3d->size(0), 2);
  EXPECT_EQ(tensor_bf16_3d->size(1), 3);
  EXPECT_EQ(tensor_bf16_3d->size(2), 4);
  EXPECT_EQ(tensor_bf16_3d->element_size(), 2);

  // Verify memory size (2 * 3 * 4 * 2 bytes = 48 bytes)
  size_t expected_memory = 2 * 3 * 4 * 2;
  size_t actual_memory =
      tensor_bf16_3d->numel() * tensor_bf16_3d->element_size();
  EXPECT_EQ(actual_memory, expected_memory);

  // Delete the 3D bfloat16 tensor
  error = aoti_torch_delete_tensor_object(tensor_bf16_3d);
  EXPECT_EQ(error, Error::Ok);

  // Test bfloat16 scalar tensor (0D) deletion
  std::vector<int64_t> scalar_sizes = {};
  Tensor* tensor_bf16_scalar = create_test_tensor(
      scalar_sizes,
      {},
      static_cast<int32_t>(SupportedDTypes::BFLOAT16),
      1, // CUDA device
      0);
  ASSERT_NE(tensor_bf16_scalar, nullptr);

  // Verify scalar tensor properties
  EXPECT_EQ(tensor_bf16_scalar->dim(), 0);
  EXPECT_EQ(tensor_bf16_scalar->numel(), 1);
  EXPECT_EQ(tensor_bf16_scalar->element_size(), 2);

  // Delete the scalar bfloat16 tensor
  error = aoti_torch_delete_tensor_object(tensor_bf16_scalar);
  EXPECT_EQ(error, Error::Ok);

  // Test zero-element bfloat16 tensor deletion
  std::vector<int64_t> zero_sizes = {0, 5};
  Tensor* tensor_bf16_zero = create_test_tensor(
      zero_sizes,
      {},
      static_cast<int32_t>(SupportedDTypes::BFLOAT16),
      1, // CUDA device
      0);
  ASSERT_NE(tensor_bf16_zero, nullptr);

  // Verify zero-element tensor properties
  EXPECT_EQ(tensor_bf16_zero->dim(), 2);
  EXPECT_EQ(tensor_bf16_zero->size(0), 0);
  EXPECT_EQ(tensor_bf16_zero->size(1), 5);
  EXPECT_EQ(tensor_bf16_zero->numel(), 0);
  EXPECT_EQ(tensor_bf16_zero->element_size(), 2);

  // Delete the zero-element bfloat16 tensor
  error = aoti_torch_delete_tensor_object(tensor_bf16_zero);
  EXPECT_EQ(error, Error::Ok);
}

// Test deletion of mixed dtype tensors (float32 and bfloat16)
