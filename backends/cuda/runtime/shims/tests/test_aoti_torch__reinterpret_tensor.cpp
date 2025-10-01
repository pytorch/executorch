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

// Test fixture for aoti_torch__reinterpret_tensor tests
class AOTITorchReinterpretTensorTest : public ::testing::Test {
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

  // Helper to calculate number of elements from sizes
  int64_t calculate_numel(const std::vector<int64_t>& sizes) {
    int64_t numel = 1;
    for (int64_t size : sizes) {
      numel *= size;
    }
    return numel;
  }

  // Helper to calculate contiguous strides from sizes
  std::vector<int64_t> calculate_contiguous_strides(
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

  // Helper to create a source tensor using empty_strided (which allocates new
  // memory)
  Tensor* create_source_tensor(
      const std::vector<int64_t>& sizes,
      int32_t dtype = 6, // float32
      int32_t device_type = 1, // CUDA
      int32_t device_index = 0) {
    std::vector<int64_t> strides = calculate_contiguous_strides(sizes);

    Tensor* tensor;
    AOTITorchError error = aoti_torch_empty_strided(
        sizes.size(),
        sizes.data(),
        strides.data(),
        dtype,
        device_type,
        device_index,
        &tensor);

    if (error != Error::Ok) {
      return nullptr;
    }

    return tensor;
  }

 private:
  std::vector<void*> cuda_memory_buffers_;
  std::vector<void*> cpu_memory_buffers_;
};

// Test basic functionality: reinterpret tensor with different shapes
TEST_F(AOTITorchReinterpretTensorTest, BasicReinterpretation) {
  // Create a source tensor with shape [12] (1D with 12 elements)
  std::vector<int64_t> source_sizes = {12};
  Tensor* source_tensor = create_source_tensor(source_sizes);
  ASSERT_NE(source_tensor, nullptr);

  // Store the original data pointer
  void* original_data_ptr = source_tensor->mutable_data_ptr();
  ASSERT_NE(original_data_ptr, nullptr);

  // Reinterpret as [3, 4] (2D with same number of elements)
  std::vector<int64_t> new_sizes = {3, 4};
  std::vector<int64_t> new_strides = calculate_contiguous_strides(new_sizes);

  Tensor* reinterpreted_tensor;
  AOTITorchError error = aoti_torch__reinterpret_tensor(
      source_tensor,
      new_sizes.size(),
      new_sizes.data(),
      new_strides.data(),
      0, // storage_offset
      &reinterpreted_tensor);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(reinterpreted_tensor, nullptr);

  // Check that the reinterpreted tensor has the new shape
  EXPECT_EQ(reinterpreted_tensor->dim(), 2);
  EXPECT_EQ(reinterpreted_tensor->size(0), 3);
  EXPECT_EQ(reinterpreted_tensor->size(1), 4);

  // CRITICAL: Check that the reinterpreted tensor uses the SAME memory
  void* reinterpreted_data_ptr = reinterpreted_tensor->mutable_data_ptr();
  EXPECT_EQ(reinterpreted_data_ptr, original_data_ptr)
      << "Reinterpreted tensor should use the same memory as the source tensor";

  // Write data through the original tensor and verify it's visible through the
  // reinterpreted tensor
  std::vector<float> test_data = {
      1.0f,
      2.0f,
      3.0f,
      4.0f,
      5.0f,
      6.0f,
      7.0f,
      8.0f,
      9.0f,
      10.0f,
      11.0f,
      12.0f};
  cudaError_t cuda_err = cudaMemcpy(
      original_data_ptr,
      test_data.data(),
      test_data.size() * sizeof(float),
      cudaMemcpyHostToDevice);
  EXPECT_EQ(cuda_err, cudaSuccess);

  // Read back through the reinterpreted tensor
  std::vector<float> readback_data(12);
  cuda_err = cudaMemcpy(
      readback_data.data(),
      reinterpreted_data_ptr,
      readback_data.size() * sizeof(float),
      cudaMemcpyDeviceToHost);
  EXPECT_EQ(cuda_err, cudaSuccess);

  // Verify the data matches
  for (size_t i = 0; i < test_data.size(); i++) {
    EXPECT_EQ(readback_data[i], test_data[i])
        << "Data should be the same through both tensors at index " << i;
  }
}

// Test reinterpreting with different strides
TEST_F(AOTITorchReinterpretTensorTest, ReinterpretWithCustomStrides) {
  // Create a source tensor with shape [2, 6] (contiguous)
  std::vector<int64_t> source_sizes = {2, 6};
  Tensor* source_tensor = create_source_tensor(source_sizes);
  ASSERT_NE(source_tensor, nullptr);

  void* original_data_ptr = source_tensor->mutable_data_ptr();
  ASSERT_NE(original_data_ptr, nullptr);

  // Reinterpret as [3, 4] with custom strides (still valid for the same memory)
  std::vector<int64_t> new_sizes = {3, 4};
  std::vector<int64_t> new_strides = {4, 1}; // Row-major strides for [3, 4]

  Tensor* reinterpreted_tensor;
  AOTITorchError error = aoti_torch__reinterpret_tensor(
      source_tensor,
      new_sizes.size(),
      new_sizes.data(),
      new_strides.data(),
      0, // storage_offset
      &reinterpreted_tensor);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(reinterpreted_tensor, nullptr);

  // Check shape
  EXPECT_EQ(reinterpreted_tensor->dim(), 2);
  EXPECT_EQ(reinterpreted_tensor->size(0), 3);
  EXPECT_EQ(reinterpreted_tensor->size(1), 4);

  // CRITICAL: Check that the reinterpreted tensor uses the SAME memory
  void* reinterpreted_data_ptr = reinterpreted_tensor->mutable_data_ptr();
  EXPECT_EQ(reinterpreted_data_ptr, original_data_ptr)
      << "Reinterpreted tensor should use the same memory as the source tensor";

  // Verify strides were set correctly
  int64_t* tensor_strides;
  error = aoti_torch_get_strides(reinterpreted_tensor, &tensor_strides);
  EXPECT_EQ(error, Error::Ok);
  EXPECT_EQ(tensor_strides[0], 4);
  EXPECT_EQ(tensor_strides[1], 1);
}

// Test error cases: null input tensor
TEST_F(AOTITorchReinterpretTensorTest, NullInputTensor) {
  std::vector<int64_t> new_sizes = {2, 3};
  std::vector<int64_t> new_strides = calculate_contiguous_strides(new_sizes);

  Tensor* reinterpreted_tensor;
  AOTITorchError error = aoti_torch__reinterpret_tensor(
      nullptr, // null input tensor
      new_sizes.size(),
      new_sizes.data(),
      new_strides.data(),
      0, // storage_offset
      &reinterpreted_tensor);

  EXPECT_EQ(error, Error::InvalidArgument);
}

// Test error cases: null sizes pointer
TEST_F(AOTITorchReinterpretTensorTest, NullSizesPointer) {
  std::vector<int64_t> source_sizes = {6};
  Tensor* source_tensor = create_source_tensor(source_sizes);
  ASSERT_NE(source_tensor, nullptr);

  std::vector<int64_t> new_strides = {2, 1};

  Tensor* reinterpreted_tensor;
  AOTITorchError error = aoti_torch__reinterpret_tensor(
      source_tensor,
      2, // ndim > 0
      nullptr, // null sizes pointer
      new_strides.data(),
      0, // storage_offset
      &reinterpreted_tensor);

  EXPECT_EQ(error, Error::InvalidArgument);
}

// Test error cases: null return tensor pointer
TEST_F(AOTITorchReinterpretTensorTest, NullReturnTensorPointer) {
  std::vector<int64_t> source_sizes = {6};
  Tensor* source_tensor = create_source_tensor(source_sizes);
  ASSERT_NE(source_tensor, nullptr);

  std::vector<int64_t> new_sizes = {2, 3};
  std::vector<int64_t> new_strides = calculate_contiguous_strides(new_sizes);

  AOTITorchError error = aoti_torch__reinterpret_tensor(
      source_tensor,
      new_sizes.size(),
      new_sizes.data(),
      new_strides.data(),
      0, // storage_offset
      nullptr); // null return tensor pointer

  EXPECT_EQ(error, Error::InvalidArgument);
}

// Test error cases: non-zero storage offset (should fail)
TEST_F(AOTITorchReinterpretTensorTest, NonZeroStorageOffset) {
  std::vector<int64_t> source_sizes = {6};
  Tensor* source_tensor = create_source_tensor(source_sizes);
  ASSERT_NE(source_tensor, nullptr);

  std::vector<int64_t> new_sizes = {2, 3};
  std::vector<int64_t> new_strides = calculate_contiguous_strides(new_sizes);

  Tensor* reinterpreted_tensor;
  AOTITorchError error = aoti_torch__reinterpret_tensor(
      source_tensor,
      new_sizes.size(),
      new_sizes.data(),
      new_strides.data(),
      1, // non-zero storage_offset (should fail)
      &reinterpreted_tensor);

  EXPECT_EQ(error, Error::InvalidArgument);
}

// Test reinterpreting CPU tensor
TEST_F(AOTITorchReinterpretTensorTest, ReinterpretCPUTensor) {
  // Create a CPU tensor with shape [8]
  std::vector<int64_t> source_sizes = {8};
  Tensor* source_tensor = create_source_tensor(
      source_sizes,
      6, // float32
      0, // CPU device
      0);
  ASSERT_NE(source_tensor, nullptr);

  void* original_data_ptr = source_tensor->mutable_data_ptr();
  ASSERT_NE(original_data_ptr, nullptr);

  // Reinterpret as [2, 4]
  std::vector<int64_t> new_sizes = {2, 4};
  std::vector<int64_t> new_strides = calculate_contiguous_strides(new_sizes);

  Tensor* reinterpreted_tensor;
  AOTITorchError error = aoti_torch__reinterpret_tensor(
      source_tensor,
      new_sizes.size(),
      new_sizes.data(),
      new_strides.data(),
      0, // storage_offset
      &reinterpreted_tensor);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(reinterpreted_tensor, nullptr);

  // Check that the reinterpreted tensor uses the SAME memory
  void* reinterpreted_data_ptr = reinterpreted_tensor->mutable_data_ptr();
  EXPECT_EQ(reinterpreted_data_ptr, original_data_ptr)
      << "Reinterpreted CPU tensor should use the same memory as the source tensor";

  // Test direct memory access for CPU tensors
  float* original_float_ptr = reinterpret_cast<float*>(original_data_ptr);
  float* reinterpreted_float_ptr =
      reinterpret_cast<float*>(reinterpreted_data_ptr);

  // Write through original and read through reinterpreted
  original_float_ptr[0] = 42.0f;
  EXPECT_EQ(reinterpreted_float_ptr[0], 42.0f)
      << "Changes through original tensor should be visible through reinterpreted tensor";
}

// Test that deleting source tensor doesn't affect reinterpreted tensor (they
// share memory)
TEST_F(AOTITorchReinterpretTensorTest, DeletionBehavior) {
  std::vector<int64_t> source_sizes = {6};
  Tensor* source_tensor = create_source_tensor(source_sizes);
  ASSERT_NE(source_tensor, nullptr);

  void* shared_data_ptr = source_tensor->mutable_data_ptr();

  // Reinterpret as [2, 3]
  std::vector<int64_t> new_sizes = {2, 3};
  std::vector<int64_t> new_strides = calculate_contiguous_strides(new_sizes);

  Tensor* reinterpreted_tensor;
  AOTITorchError error = aoti_torch__reinterpret_tensor(
      source_tensor,
      new_sizes.size(),
      new_sizes.data(),
      new_strides.data(),
      0,
      &reinterpreted_tensor);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(reinterpreted_tensor, nullptr);

  // Verify they share the same memory
  EXPECT_EQ(reinterpreted_tensor->mutable_data_ptr(), shared_data_ptr);

  // Delete the source tensor (which owns the memory)
  error = aoti_torch_delete_tensor_object(source_tensor);
  EXPECT_EQ(error, Error::Ok);

  // The reinterpreted tensor should still be valid but the memory might be
  // freed Since the source tensor owned the memory, the reinterpreted tensor
  // becomes invalid This is expected behavior - the user needs to manage the
  // lifecycle properly

  // Clean up the reinterpreted tensor
  error = aoti_torch_delete_tensor_object(reinterpreted_tensor);
  EXPECT_EQ(error, Error::Ok);
}

// Test scalar tensor reinterpretation
TEST_F(AOTITorchReinterpretTensorTest, ReinterpretScalarTensor) {
  // Create a scalar tensor (0D)
  std::vector<int64_t> source_sizes = {};
  Tensor* source_tensor = create_source_tensor(source_sizes);
  ASSERT_NE(source_tensor, nullptr);

  void* original_data_ptr = source_tensor->mutable_data_ptr();

  // Try to reinterpret scalar as [1] (1D with 1 element)
  std::vector<int64_t> new_sizes = {1};
  std::vector<int64_t> new_strides = {1};

  Tensor* reinterpreted_tensor;
  AOTITorchError error = aoti_torch__reinterpret_tensor(
      source_tensor,
      new_sizes.size(),
      new_sizes.data(),
      new_strides.data(),
      0,
      &reinterpreted_tensor);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(reinterpreted_tensor, nullptr);

  // Check that the reinterpreted tensor uses the SAME memory
  EXPECT_EQ(reinterpreted_tensor->mutable_data_ptr(), original_data_ptr);

  // Check new shape
  EXPECT_EQ(reinterpreted_tensor->dim(), 1);
  EXPECT_EQ(reinterpreted_tensor->size(0), 1);
}

// Test reinterpreting tensor with zero-sized dimension
// TODO: This test is disabled because zero-sized tensors have complex stride
// validation requirements that need further investigation
TEST_F(AOTITorchReinterpretTensorTest, DISABLED_ReinterpretZeroSizedTensor) {
  // Create a tensor with shape [0, 5] (zero elements)
  std::vector<int64_t> source_sizes = {0, 5};
  Tensor* source_tensor = create_source_tensor(source_sizes);
  ASSERT_NE(source_tensor, nullptr);

  void* original_data_ptr = source_tensor->mutable_data_ptr();

  // Reinterpret as [5, 0] (still zero elements)
  std::vector<int64_t> new_sizes = {5, 0};
  std::vector<int64_t> new_strides = calculate_contiguous_strides(new_sizes);

  Tensor* reinterpreted_tensor;
  AOTITorchError error = aoti_torch__reinterpret_tensor(
      source_tensor,
      new_sizes.size(),
      new_sizes.data(),
      new_strides.data(),
      0,
      &reinterpreted_tensor);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(reinterpreted_tensor, nullptr);

  // Check that the reinterpreted tensor uses the SAME memory
  EXPECT_EQ(reinterpreted_tensor->mutable_data_ptr(), original_data_ptr);

  // Check new shape
  EXPECT_EQ(reinterpreted_tensor->dim(), 2);
  EXPECT_EQ(reinterpreted_tensor->size(0), 5);
  EXPECT_EQ(reinterpreted_tensor->size(1), 0);
}

// Test with nullptr strides (should use contiguous strides)
TEST_F(AOTITorchReinterpretTensorTest, NullStridesPointer) {
  std::vector<int64_t> source_sizes = {12};
  Tensor* source_tensor = create_source_tensor(source_sizes);
  ASSERT_NE(source_tensor, nullptr);

  void* original_data_ptr = source_tensor->mutable_data_ptr();

  // Reinterpret as [3, 4] with null strides (should calculate contiguous
  // strides)
  std::vector<int64_t> new_sizes = {3, 4};

  Tensor* reinterpreted_tensor;
  AOTITorchError error = aoti_torch__reinterpret_tensor(
      source_tensor,
      new_sizes.size(),
      new_sizes.data(),
      nullptr, // null strides - should calculate contiguous strides
      0,
      &reinterpreted_tensor);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(reinterpreted_tensor, nullptr);

  // Check that the reinterpreted tensor uses the SAME memory
  EXPECT_EQ(reinterpreted_tensor->mutable_data_ptr(), original_data_ptr);

  // Check that contiguous strides were calculated correctly
  int64_t* tensor_strides;
  error = aoti_torch_get_strides(reinterpreted_tensor, &tensor_strides);
  EXPECT_EQ(error, Error::Ok);
  EXPECT_EQ(tensor_strides[0], 4); // stride for dimension 0 should be 4
  EXPECT_EQ(tensor_strides[1], 1); // stride for dimension 1 should be 1
}

// Test bf16 tensor reinterpretation
TEST_F(AOTITorchReinterpretTensorTest, ReinterpretBF16Tensor) {
  // Create a bf16 source tensor with shape [6]
  std::vector<int64_t> source_sizes = {6};
  Tensor* source_tensor = create_source_tensor(
      source_sizes,
      static_cast<int32_t>(
          SupportedDTypes::BFLOAT16), // bf16 dtype from SupportedDTypes
      static_cast<int32_t>(
          SupportedDevices::CUDA), // CUDA device from SupportedDevices
      0); // device_index must be 0
  ASSERT_NE(source_tensor, nullptr);

  void* original_data_ptr = source_tensor->mutable_data_ptr();
  ASSERT_NE(original_data_ptr, nullptr);

  // Verify the tensor is actually bf16
  int32_t actual_dtype = 0;
  AOTITorchError dtype_check_error =
      aoti_torch_get_dtype(source_tensor, &actual_dtype);
  EXPECT_EQ(dtype_check_error, Error::Ok);
  EXPECT_EQ(actual_dtype, static_cast<int32_t>(SupportedDTypes::BFLOAT16))
      << "Source tensor should have bfloat16 dtype";

  // Reinterpret as [2, 3] (same number of elements)
  std::vector<int64_t> new_sizes = {2, 3};
  std::vector<int64_t> new_strides = calculate_contiguous_strides(new_sizes);

  Tensor* reinterpreted_tensor;
  AOTITorchError error = aoti_torch__reinterpret_tensor(
      source_tensor,
      new_sizes.size(),
      new_sizes.data(),
      new_strides.data(),
      0, // storage_offset
      &reinterpreted_tensor);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(reinterpreted_tensor, nullptr);

  // Check that the reinterpreted tensor has the new shape
  EXPECT_EQ(reinterpreted_tensor->dim(), 2);
  EXPECT_EQ(reinterpreted_tensor->size(0), 2);
  EXPECT_EQ(reinterpreted_tensor->size(1), 3);

  // Verify the dtype is preserved as bf16
  int32_t reinterpreted_dtype = 0;
  dtype_check_error =
      aoti_torch_get_dtype(reinterpreted_tensor, &reinterpreted_dtype);
  EXPECT_EQ(dtype_check_error, Error::Ok);
  EXPECT_EQ(
      reinterpreted_dtype, static_cast<int32_t>(SupportedDTypes::BFLOAT16))
      << "Reinterpreted tensor should preserve bfloat16 dtype";

  // CRITICAL: Check that the reinterpreted tensor uses the SAME memory
  void* reinterpreted_data_ptr = reinterpreted_tensor->mutable_data_ptr();
  EXPECT_EQ(reinterpreted_data_ptr, original_data_ptr)
      << "Reinterpreted tensor should use the same memory as the source tensor";

  // Test memory sharing by writing data through the original tensor
  // and verifying it's visible through the reinterpreted tensor
  // Note: bf16 has 2 bytes per element
  std::vector<uint16_t> test_data_bf16 = {
      0x3F80, 0x4000, 0x4040, 0x4080, 0x40A0, 0x40C0}; // bf16 values
  cudaError_t cuda_err = cudaMemcpy(
      original_data_ptr,
      test_data_bf16.data(),
      test_data_bf16.size() * sizeof(uint16_t),
      cudaMemcpyHostToDevice);
  EXPECT_EQ(cuda_err, cudaSuccess);

  // Read back through the reinterpreted tensor
  std::vector<uint16_t> readback_data_bf16(6);
  cuda_err = cudaMemcpy(
      readback_data_bf16.data(),
      reinterpreted_data_ptr,
      readback_data_bf16.size() * sizeof(uint16_t),
      cudaMemcpyDeviceToHost);
  EXPECT_EQ(cuda_err, cudaSuccess);

  // Verify the data matches
  for (size_t i = 0; i < test_data_bf16.size(); i++) {
    EXPECT_EQ(readback_data_bf16[i], test_data_bf16[i])
        << "BF16 data should be the same through both tensors at index " << i;
  }
}

// Test reference counting behavior - memory not in map should fail
TEST_F(AOTITorchReinterpretTensorTest, MemoryNotInMapShouldFail) {
  // Create a tensor directly without using our allocation functions
  // This should NOT be in the reference counting map
  void* external_memory;
  ASSERT_EQ(
      cudaMallocManaged(&external_memory, 12 * sizeof(float)), cudaSuccess);

  // Create a tensor by manually wrapping this memory without going through our
  // APIs
  std::vector<int64_t> sizes = {12};
  std::vector<int64_t> strides = calculate_contiguous_strides(sizes);

  // Create the tensor directly using ExecutorTorch extension
  auto tensor_shared = executorch::extension::from_blob(
      external_memory,
      convert_sizes_to_vector(sizes.size(), sizes.data()),
      convert_strides_to_vector(sizes.size(), sizes.data(), strides.data()),
      executorch::runtime::etensor::ScalarType::Float);

  ASSERT_TRUE(tensor_shared);
  Tensor* external_tensor = tensor_shared.get();

  // Try to reinterpret this tensor - should fail because memory is not in map
  std::vector<int64_t> new_sizes = {3, 4};
  std::vector<int64_t> new_strides = calculate_contiguous_strides(new_sizes);

  Tensor* reinterpreted_tensor;
  AOTITorchError error = aoti_torch__reinterpret_tensor(
      external_tensor,
      new_sizes.size(),
      new_sizes.data(),
      new_strides.data(),
      0, // storage_offset
      &reinterpreted_tensor);

  // Should fail because memory is not being tracked by reference counting
  // system
  EXPECT_EQ(error, Error::InvalidArgument);

  // Clean up the external memory
  ASSERT_EQ(cudaFree(external_memory), cudaSuccess);
}

// Test reference counting behavior - creating view increments reference count
TEST_F(AOTITorchReinterpretTensorTest, ViewCreationIncrementsReferenceCount) {
  // Create a source tensor that owns memory (reference count = 1)
  std::vector<int64_t> source_sizes = {12};
  Tensor* source_tensor = create_source_tensor(source_sizes);
  ASSERT_NE(source_tensor, nullptr);

  void* shared_data_ptr = source_tensor->mutable_data_ptr();
  ASSERT_NE(shared_data_ptr, nullptr);

  // Create first view - should increment reference count to 2
  std::vector<int64_t> view1_sizes = {3, 4};
  std::vector<int64_t> view1_strides =
      calculate_contiguous_strides(view1_sizes);

  Tensor* view1_tensor;
  AOTITorchError error = aoti_torch__reinterpret_tensor(
      source_tensor,
      view1_sizes.size(),
      view1_sizes.data(),
      view1_strides.data(),
      0,
      &view1_tensor);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(view1_tensor, nullptr);
  EXPECT_EQ(view1_tensor->mutable_data_ptr(), shared_data_ptr);

  // Create second view - should increment reference count to 3
  std::vector<int64_t> view2_sizes = {2, 6};
  std::vector<int64_t> view2_strides =
      calculate_contiguous_strides(view2_sizes);

  Tensor* view2_tensor;
  error = aoti_torch__reinterpret_tensor(
      source_tensor,
      view2_sizes.size(),
      view2_sizes.data(),
      view2_strides.data(),
      0,
      &view2_tensor);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(view2_tensor, nullptr);
  EXPECT_EQ(view2_tensor->mutable_data_ptr(), shared_data_ptr);

  // Now delete the source tensor - memory should NOT be freed (reference count
  // = 2)
  error = aoti_torch_delete_tensor_object(source_tensor);
  EXPECT_EQ(error, Error::Ok);

  // Both views should still be valid - test by accessing memory
  float test_value = 42.0f;
  cudaError_t cuda_err = cudaMemcpy(
      shared_data_ptr, &test_value, sizeof(float), cudaMemcpyHostToDevice);
  EXPECT_EQ(cuda_err, cudaSuccess);

  float readback_value = 0.0f;
  cuda_err = cudaMemcpy(
      &readback_value,
      view1_tensor->mutable_data_ptr(),
      sizeof(float),
      cudaMemcpyDeviceToHost);
  EXPECT_EQ(cuda_err, cudaSuccess);
  EXPECT_EQ(readback_value, test_value);

  // Delete first view - memory should still NOT be freed (reference count = 1)
  error = aoti_torch_delete_tensor_object(view1_tensor);
  EXPECT_EQ(error, Error::Ok);

  // Second view should still be valid
  readback_value = 0.0f;
  cuda_err = cudaMemcpy(
      &readback_value,
      view2_tensor->mutable_data_ptr(),
      sizeof(float),
      cudaMemcpyDeviceToHost);
  EXPECT_EQ(cuda_err, cudaSuccess);
  EXPECT_EQ(readback_value, test_value);

  // Delete second view - NOW memory should be freed (reference count = 0)
  error = aoti_torch_delete_tensor_object(view2_tensor);
  EXPECT_EQ(error, Error::Ok);
}

// Test reference counting behavior with NOT_OWN memory (from blob) - should
// SUCCEED and keep NOT_OWN
TEST_F(AOTITorchReinterpretTensorTest, ViewOfNotOwnMemoryKeepsNotOwnStatus) {
  // Allocate external memory
  void* external_memory;
  cudaError_t cuda_err =
      cudaMallocManaged(&external_memory, 12 * sizeof(float));
  ASSERT_EQ(cuda_err, cudaSuccess);

  // Create tensor from blob (which marks memory as NOT_OWN)
  std::vector<int64_t> blob_sizes = {12};
  std::vector<int64_t> blob_strides = calculate_contiguous_strides(blob_sizes);

  Tensor* blob_tensor;
  AOTITorchError error = aoti_torch_create_tensor_from_blob_v2(
      external_memory,
      blob_sizes.size(),
      blob_sizes.data(),
      blob_strides.data(),
      0, // storage_offset
      static_cast<int32_t>(SupportedDTypes::FLOAT32),
      static_cast<int32_t>(SupportedDevices::CUDA),
      0, // device_index
      &blob_tensor,
      0, // layout
      nullptr, // opaque_metadata
      0); // opaque_metadata_size

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(blob_tensor, nullptr);

  // Create view of NOT_OWN memory - should SUCCEED and keep NOT_OWN status
  std::vector<int64_t> view_sizes = {3, 4};
  std::vector<int64_t> view_strides = calculate_contiguous_strides(view_sizes);

  Tensor* view_tensor;
  error = aoti_torch__reinterpret_tensor(
      blob_tensor,
      view_sizes.size(),
      view_sizes.data(),
      view_strides.data(),
      0,
      &view_tensor);

  // Should succeed - NOT_OWN memory can be reinterpreted but stays NOT_OWN
  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(view_tensor, nullptr);
  EXPECT_EQ(view_tensor->mutable_data_ptr(), external_memory);

  // Verify both tensors share the same memory
  EXPECT_EQ(blob_tensor->mutable_data_ptr(), view_tensor->mutable_data_ptr());

  // Test memory sharing by writing data through one tensor and reading through
  // the other
  float test_value = 42.0f;
  cuda_err = cudaMemcpy(
      external_memory, &test_value, sizeof(float), cudaMemcpyHostToDevice);
  EXPECT_EQ(cuda_err, cudaSuccess);

  float readback_value = 0.0f;
  cuda_err = cudaMemcpy(
      &readback_value,
      view_tensor->mutable_data_ptr(),
      sizeof(float),
      cudaMemcpyDeviceToHost);
  EXPECT_EQ(cuda_err, cudaSuccess);
  EXPECT_EQ(readback_value, test_value);

  // Delete the blob tensor - external memory should NOT be freed (NOT_OWN
  // behavior)
  error = aoti_torch_delete_tensor_object(blob_tensor);
  EXPECT_EQ(error, Error::Ok);

  // View tensor should still be valid - test by accessing memory
  readback_value = 0.0f;
  cuda_err = cudaMemcpy(
      &readback_value,
      view_tensor->mutable_data_ptr(),
      sizeof(float),
      cudaMemcpyDeviceToHost);
  EXPECT_EQ(cuda_err, cudaSuccess);
  EXPECT_EQ(readback_value, test_value);

  // Delete view tensor - external memory should still NOT be freed (NOT_OWN
  // behavior)
  error = aoti_torch_delete_tensor_object(view_tensor);
  EXPECT_EQ(error, Error::Ok);

  // External memory should still be accessible (proves neither tensor freed it)
  readback_value = 0.0f;
  cuda_err = cudaMemcpy(
      &readback_value, external_memory, sizeof(float), cudaMemcpyDeviceToHost);
  EXPECT_EQ(cuda_err, cudaSuccess);
  EXPECT_EQ(readback_value, test_value);

  // Clean up external memory manually (as expected for NOT_OWN memory)
  ASSERT_EQ(cudaFree(external_memory), cudaSuccess);
}
