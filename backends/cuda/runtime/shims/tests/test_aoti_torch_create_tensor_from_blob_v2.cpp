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

// Test fixture for aoti_torch_create_tensor_from_blob_v2 tests
class AOTITorchCreateTensorFromBlobV2Test : public ::testing::Test {
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

    // Clean up any allocated memory buffers
    for (void* ptr : cuda_memory_buffers_) {
      if (ptr) {
        cudaError_t cuda_err = cudaFree(ptr);
        EXPECT_EQ(cuda_err, cudaSuccess)
            << "Failed to free CUDA memory: " << cudaGetErrorString(cuda_err);
      }
    }
    cuda_memory_buffers_.clear();

    for (void* ptr : cpu_memory_buffers_) {
      if (ptr) {
        free(ptr);
      }
    }
    cpu_memory_buffers_.clear();
  }

  // Helper to allocate CUDA memory and track it for cleanup
  void* allocate_cuda_memory(size_t bytes) {
    void* ptr;
    cudaError_t err = cudaMallocManaged(&ptr, bytes);
    if (err == cudaSuccess) {
      cuda_memory_buffers_.push_back(ptr);
      return ptr;
    }
    return nullptr;
  }

  // Helper to allocate CPU memory and track it for cleanup
  void* allocate_cpu_memory(size_t bytes) {
    void* ptr;
    int result = posix_memalign(&ptr, 16, bytes); // 16-byte aligned
    if (result == 0 && ptr != nullptr) {
      cpu_memory_buffers_.push_back(ptr);
      return ptr;
    }
    return nullptr;
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
    // Use int64_t and check for underflow to avoid unsigned integer wraparound
    for (int64_t i = static_cast<int64_t>(sizes.size()) - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * sizes[i + 1];
    }
    return strides;
  }

 private:
  std::vector<void*> cuda_memory_buffers_;
  std::vector<void*> cpu_memory_buffers_;
};

// Test basic functionality with CUDA memory
TEST_F(AOTITorchCreateTensorFromBlobV2Test, BasicFunctionalityCUDA) {
  // Test 1D tensor
  std::vector<int64_t> sizes_1d = {5};
  std::vector<int64_t> strides_1d = calculate_contiguous_strides(sizes_1d);

  // Allocate CUDA memory
  size_t bytes = calculate_numel(sizes_1d) * sizeof(float);
  void* cuda_data = allocate_cuda_memory(bytes);
  ASSERT_NE(cuda_data, nullptr);

  Tensor* tensor_1d;
  AOTITorchError error = aoti_torch_create_tensor_from_blob_v2(
      cuda_data,
      sizes_1d.size(),
      sizes_1d.data(),
      strides_1d.data(),
      0, // storage_offset
      static_cast<int32_t>(SupportedDTypes::FLOAT32),
      static_cast<int32_t>(SupportedDevices::CUDA),
      0, // device index
      &tensor_1d,
      0, // layout (strided)
      nullptr, // opaque_metadata
      0); // opaque_metadata_size

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(tensor_1d, nullptr);

  // Check tensor properties
  EXPECT_EQ(tensor_1d->dim(), 1);
  EXPECT_EQ(tensor_1d->size(0), 5);

  // Verify the tensor uses the same data pointer
  void* tensor_data = tensor_1d->mutable_data_ptr();
  EXPECT_EQ(tensor_data, cuda_data);

  // Delete the tensor - this should NOT free the original memory
  error = aoti_torch_delete_tensor_object(tensor_1d);
  EXPECT_EQ(error, Error::Ok);

  // Test that the original memory is still accessible (proves tensor didn't own
  // it) For CUDA memory, check that we can still access it (synchronously)
  // after tensor deletion
  float pattern_value = 42.0f;
  cudaError_t cuda_err = cudaMemcpy(
      cuda_data, &pattern_value, sizeof(float), cudaMemcpyHostToDevice);
  EXPECT_EQ(cuda_err, cudaSuccess)
      << "Should be able to write to original CUDA memory after tensor deletion";

  float readback_value = 0.0f;
  cuda_err = cudaMemcpy(
      &readback_value, cuda_data, sizeof(float), cudaMemcpyDeviceToHost);
  EXPECT_EQ(cuda_err, cudaSuccess)
      << "Should be able to read from original CUDA memory after tensor deletion";
  EXPECT_EQ(readback_value, pattern_value)
      << "Original CUDA memory should still contain our test pattern";
}

// Test basic functionality with CPU memory
TEST_F(AOTITorchCreateTensorFromBlobV2Test, BasicFunctionalityCPU) {
  // Test 2D tensor
  std::vector<int64_t> sizes_2d = {3, 4};
  std::vector<int64_t> strides_2d = calculate_contiguous_strides(sizes_2d);

  // Allocate CPU memory
  size_t bytes = calculate_numel(sizes_2d) * sizeof(float);
  void* cpu_data = allocate_cpu_memory(bytes);
  ASSERT_NE(cpu_data, nullptr);

  Tensor* tensor_2d;
  AOTITorchError error = aoti_torch_create_tensor_from_blob_v2(
      cpu_data,
      sizes_2d.size(),
      sizes_2d.data(),
      strides_2d.data(),
      0, // storage_offset
      static_cast<int32_t>(SupportedDTypes::FLOAT32),
      static_cast<int32_t>(SupportedDevices::CPU),
      0, // device index
      &tensor_2d,
      0, // layout (strided)
      nullptr, // opaque_metadata
      0); // opaque_metadata_size

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(tensor_2d, nullptr);

  // Check tensor properties
  EXPECT_EQ(tensor_2d->dim(), 2);
  EXPECT_EQ(tensor_2d->size(0), 3);
  EXPECT_EQ(tensor_2d->size(1), 4);

  // Verify the tensor uses the same data pointer
  void* tensor_data = tensor_2d->mutable_data_ptr();
  EXPECT_EQ(tensor_data, cpu_data);

  // Delete the tensor - this should NOT free the original memory
  error = aoti_torch_delete_tensor_object(tensor_2d);
  EXPECT_EQ(error, Error::Ok);

  // Test that the original memory is still accessible (proves tensor didn't own
  // it) For CPU memory, directly write and read to verify accessibility
  float* float_ptr = reinterpret_cast<float*>(cpu_data);
  float pattern_value = 42.0f;
  *float_ptr = pattern_value;
  EXPECT_EQ(*float_ptr, pattern_value)
      << "Original CPU memory should still be accessible after tensor deletion";
}

// Test with invalid dtype
TEST_F(AOTITorchCreateTensorFromBlobV2Test, InvalidDtype) {
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = calculate_contiguous_strides(sizes);

  size_t bytes = calculate_numel(sizes) * sizeof(float);
  void* data = allocate_cuda_memory(bytes);
  ASSERT_NE(data, nullptr);

  Tensor* tensor;
  AOTITorchError error = aoti_torch_create_tensor_from_blob_v2(
      data,
      sizes.size(),
      sizes.data(),
      strides.data(),
      0, // storage_offset
      999, // invalid dtype
      static_cast<int32_t>(SupportedDevices::CUDA),
      0, // device index
      &tensor,
      0, // layout
      nullptr, // opaque_metadata
      0); // opaque_metadata_size

  EXPECT_EQ(error, Error::InvalidArgument);
}

// Test with non-zero storage offset (should fail since from_blob cannot handle
// offsets)
TEST_F(AOTITorchCreateTensorFromBlobV2Test, NonZeroStorageOffset) {
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = calculate_contiguous_strides(sizes);

  size_t bytes = calculate_numel(sizes) * sizeof(float);
  void* data = allocate_cuda_memory(bytes);
  ASSERT_NE(data, nullptr);

  Tensor* tensor;
  AOTITorchError error = aoti_torch_create_tensor_from_blob_v2(
      data,
      sizes.size(),
      sizes.data(),
      strides.data(),
      1, // non-zero storage_offset (should fail since from_blob cannot handle
         // offsets)
      static_cast<int32_t>(SupportedDTypes::FLOAT32),
      static_cast<int32_t>(SupportedDevices::CUDA),
      0, // device index
      &tensor,
      0, // layout
      nullptr, // opaque_metadata
      0); // opaque_metadata_size

  EXPECT_EQ(error, Error::InvalidArgument);
}

// Test with custom strides (using stride parameter but still contiguous)
TEST_F(AOTITorchCreateTensorFromBlobV2Test, CustomContiguousStrides) {
  std::vector<int64_t> sizes = {2, 3};
  // Use the correct contiguous strides but pass them explicitly
  std::vector<int64_t> contiguous_strides = {3, 1}; // Proper contiguous strides

  size_t bytes = calculate_numel(sizes) * sizeof(float);
  void* data = allocate_cuda_memory(bytes);
  ASSERT_NE(data, nullptr);

  Tensor* tensor;
  AOTITorchError error = aoti_torch_create_tensor_from_blob_v2(
      data,
      sizes.size(),
      sizes.data(),
      contiguous_strides.data(), // Explicitly pass contiguous strides
      0, // storage_offset
      static_cast<int32_t>(SupportedDTypes::FLOAT32),
      static_cast<int32_t>(SupportedDevices::CUDA),
      0, // device index
      &tensor,
      0, // layout
      nullptr, // opaque_metadata
      0); // opaque_metadata_size

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(tensor, nullptr);

  // Check tensor properties
  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 2);
  EXPECT_EQ(tensor->size(1), 3);

  // Verify the tensor uses the same data pointer
  void* tensor_data = tensor->mutable_data_ptr();
  EXPECT_EQ(tensor_data, data);

  // Verify strides were properly set (we can check via aoti_torch_get_strides)
  int64_t* tensor_strides;
  error = aoti_torch_get_strides(tensor, &tensor_strides);
  EXPECT_EQ(error, Error::Ok);
  EXPECT_EQ(tensor_strides[0], 3);
  EXPECT_EQ(tensor_strides[1], 1);

  // Delete the tensor - this should NOT free the original memory
  error = aoti_torch_delete_tensor_object(tensor);
  EXPECT_EQ(error, Error::Ok);

  // Test that the original memory is still accessible (proves tensor didn't own
  // it)
  float pattern_value = 42.0f;
  cudaError_t cuda_err =
      cudaMemcpy(data, &pattern_value, sizeof(float), cudaMemcpyHostToDevice);
  EXPECT_EQ(cuda_err, cudaSuccess)
      << "Should be able to write to original CUDA memory after tensor deletion";

  float readback_value = 0.0f;
  cuda_err =
      cudaMemcpy(&readback_value, data, sizeof(float), cudaMemcpyDeviceToHost);
  EXPECT_EQ(cuda_err, cudaSuccess)
      << "Should be able to read from original CUDA memory after tensor deletion";
  EXPECT_EQ(readback_value, pattern_value)
      << "Original CUDA memory should still contain our test pattern";
}

// Test with null data pointer
TEST_F(AOTITorchCreateTensorFromBlobV2Test, NullDataPointer) {
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = calculate_contiguous_strides(sizes);

  Tensor* tensor;
  AOTITorchError error = aoti_torch_create_tensor_from_blob_v2(
      nullptr, // null data pointer
      sizes.size(),
      sizes.data(),
      strides.data(),
      0, // storage_offset
      static_cast<int32_t>(SupportedDTypes::FLOAT32),
      static_cast<int32_t>(SupportedDevices::CUDA),
      0, // device index
      &tensor,
      0, // layout
      nullptr, // opaque_metadata
      0); // opaque_metadata_size

  EXPECT_EQ(error, Error::InvalidArgument);
}

// Test scalar tensor (0D)
TEST_F(AOTITorchCreateTensorFromBlobV2Test, ScalarTensor) {
  std::vector<int64_t> sizes = {}; // 0D tensor
  std::vector<int64_t> strides = {}; // Empty strides for scalar

  size_t bytes = sizeof(float); // Single element
  void* data = allocate_cuda_memory(bytes);
  ASSERT_NE(data, nullptr);

  Tensor* tensor = nullptr;
  AOTITorchError error = aoti_torch_create_tensor_from_blob_v2(
      data,
      sizes.size(),
      sizes.data(),
      strides.data(),
      0, // storage_offset
      static_cast<int32_t>(SupportedDTypes::FLOAT32),
      static_cast<int32_t>(SupportedDevices::CUDA),
      0, // device index
      &tensor,
      0, // layout
      nullptr, // opaque_metadata
      0); // opaque_metadata_size

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(tensor, nullptr);

  // Check tensor properties
  EXPECT_EQ(tensor->dim(), 0);

  // Verify the tensor uses the same data pointer
  void* tensor_data = tensor->mutable_data_ptr();
  EXPECT_EQ(tensor_data, data);

  // Delete the tensor - this should NOT free the original memory
  error = aoti_torch_delete_tensor_object(tensor);
  EXPECT_EQ(error, Error::Ok);

  // Test that the original memory is still accessible (proves tensor didn't own
  // it)
  float pattern_value = 42.0f;
  cudaError_t cuda_err =
      cudaMemcpy(data, &pattern_value, sizeof(float), cudaMemcpyHostToDevice);
  EXPECT_EQ(cuda_err, cudaSuccess)
      << "Should be able to write to original CUDA memory after tensor deletion";

  float readback_value = 0.0f;
  cuda_err =
      cudaMemcpy(&readback_value, data, sizeof(float), cudaMemcpyDeviceToHost);
  EXPECT_EQ(cuda_err, cudaSuccess)
      << "Should be able to read from original CUDA memory after tensor deletion";
  EXPECT_EQ(readback_value, pattern_value)
      << "Original CUDA memory should still contain our test pattern";
}

// Test zero-sized tensor
TEST_F(AOTITorchCreateTensorFromBlobV2Test, ZeroSizedTensor) {
  std::vector<int64_t> sizes = {0, 5}; // Zero elements
  std::vector<int64_t> strides = calculate_contiguous_strides(sizes);

  // Even for zero-sized tensor, we need some memory allocated
  size_t bytes = sizeof(float); // Minimum allocation
  void* data = allocate_cuda_memory(bytes);
  ASSERT_NE(data, nullptr);

  Tensor* tensor;
  AOTITorchError error = aoti_torch_create_tensor_from_blob_v2(
      data,
      sizes.size(),
      sizes.data(),
      strides.data(),
      0, // storage_offset
      static_cast<int32_t>(SupportedDTypes::FLOAT32),
      static_cast<int32_t>(SupportedDevices::CUDA),
      0, // device index
      &tensor,
      0, // layout
      nullptr, // opaque_metadata
      0); // opaque_metadata_size

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(tensor, nullptr);

  // Check tensor properties
  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 0);
  EXPECT_EQ(tensor->size(1), 5);

  // Verify the tensor uses the same data pointer
  void* tensor_data = tensor->mutable_data_ptr();
  EXPECT_EQ(tensor_data, data);

  // Delete the tensor - this should NOT free the original memory
  error = aoti_torch_delete_tensor_object(tensor);
  EXPECT_EQ(error, Error::Ok);

  // Test that the original memory is still accessible (proves tensor didn't own
  // it)
  float pattern_value = 42.0f;
  cudaError_t cuda_err =
      cudaMemcpy(data, &pattern_value, sizeof(float), cudaMemcpyHostToDevice);
  EXPECT_EQ(cuda_err, cudaSuccess)
      << "Should be able to write to original CUDA memory after tensor deletion";

  float readback_value = 0.0f;
  cuda_err =
      cudaMemcpy(&readback_value, data, sizeof(float), cudaMemcpyDeviceToHost);
  EXPECT_EQ(cuda_err, cudaSuccess)
      << "Should be able to read from original CUDA memory after tensor deletion";
  EXPECT_EQ(readback_value, pattern_value)
      << "Original CUDA memory should still contain our test pattern";
}

// Test multi-dimensional tensors
TEST_F(AOTITorchCreateTensorFromBlobV2Test, MultiDimensionalTensors) {
  // Test 3D tensor
  std::vector<int64_t> sizes_3d = {2, 3, 4};
  std::vector<int64_t> strides_3d = calculate_contiguous_strides(sizes_3d);

  size_t bytes_3d = calculate_numel(sizes_3d) * sizeof(float);
  void* data_3d = allocate_cuda_memory(bytes_3d);
  ASSERT_NE(data_3d, nullptr);

  Tensor* tensor_3d;
  AOTITorchError error = aoti_torch_create_tensor_from_blob_v2(
      data_3d,
      sizes_3d.size(),
      sizes_3d.data(),
      strides_3d.data(),
      0, // storage_offset
      static_cast<int32_t>(SupportedDTypes::FLOAT32),
      static_cast<int32_t>(SupportedDevices::CUDA),
      0, // device index
      &tensor_3d,
      0, // layout
      nullptr, // opaque_metadata
      0); // opaque_metadata_size

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(tensor_3d, nullptr);
  EXPECT_EQ(tensor_3d->dim(), 3);
  EXPECT_EQ(tensor_3d->size(0), 2);
  EXPECT_EQ(tensor_3d->size(1), 3);
  EXPECT_EQ(tensor_3d->size(2), 4);

  // Test 4D tensor
  std::vector<int64_t> sizes_4d = {2, 3, 4, 5};
  std::vector<int64_t> strides_4d = calculate_contiguous_strides(sizes_4d);

  size_t bytes_4d = calculate_numel(sizes_4d) * sizeof(float);
  void* data_4d = allocate_cuda_memory(bytes_4d);
  ASSERT_NE(data_4d, nullptr);

  Tensor* tensor_4d;
  error = aoti_torch_create_tensor_from_blob_v2(
      data_4d,
      sizes_4d.size(),
      sizes_4d.data(),
      strides_4d.data(),
      0, // storage_offset
      static_cast<int32_t>(SupportedDTypes::FLOAT32),
      static_cast<int32_t>(SupportedDevices::CUDA),
      0, // device index
      &tensor_4d,
      0, // layout
      nullptr, // opaque_metadata
      0); // opaque_metadata_size

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(tensor_4d, nullptr);
  EXPECT_EQ(tensor_4d->dim(), 4);
  EXPECT_EQ(tensor_4d->size(0), 2);
  EXPECT_EQ(tensor_4d->size(1), 3);
  EXPECT_EQ(tensor_4d->size(2), 4);
  EXPECT_EQ(tensor_4d->size(3), 5);
}

// Test tensor data pointer consistency
TEST_F(AOTITorchCreateTensorFromBlobV2Test, DataPointerConsistency) {
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = calculate_contiguous_strides(sizes);

  size_t bytes = calculate_numel(sizes) * sizeof(float);
  void* original_data = allocate_cuda_memory(bytes);
  ASSERT_NE(original_data, nullptr);

  Tensor* tensor;
  AOTITorchError error = aoti_torch_create_tensor_from_blob_v2(
      original_data,
      sizes.size(),
      sizes.data(),
      strides.data(),
      0, // storage_offset
      static_cast<int32_t>(SupportedDTypes::FLOAT32),
      static_cast<int32_t>(SupportedDevices::CUDA),
      0, // device index
      &tensor,
      0, // layout
      nullptr, // opaque_metadata
      0); // opaque_metadata_size

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(tensor, nullptr);

  // Check that the tensor uses the same data pointer
  void* tensor_data = tensor->mutable_data_ptr();
  EXPECT_EQ(tensor_data, original_data);
}

// Test creating multiple tensors from different blobs
TEST_F(AOTITorchCreateTensorFromBlobV2Test, MultipleTensorsFromBlobs) {
  const int num_tensors = 5;
  std::vector<Tensor*> tensors;
  std::vector<void*> data_ptrs;

  for (int i = 0; i < num_tensors; i++) {
    std::vector<int64_t> sizes = {i + 1, i + 2};
    std::vector<int64_t> strides = calculate_contiguous_strides(sizes);

    size_t bytes = calculate_numel(sizes) * sizeof(float);
    void* data = allocate_cuda_memory(bytes);
    ASSERT_NE(data, nullptr);
    data_ptrs.push_back(data);

    Tensor* tensor;
    AOTITorchError error = aoti_torch_create_tensor_from_blob_v2(
        data,
        sizes.size(),
        sizes.data(),
        strides.data(),
        0, // storage_offset
        static_cast<int32_t>(SupportedDTypes::FLOAT32),
        static_cast<int32_t>(SupportedDevices::CUDA),
        0, // device index
        &tensor,
        0, // layout
        nullptr, // opaque_metadata
        0); // opaque_metadata_size

    EXPECT_EQ(error, Error::Ok);
    EXPECT_NE(tensor, nullptr);
    tensors.push_back(tensor);

    // Verify dimensions
    EXPECT_EQ(tensor->dim(), 2);
    EXPECT_EQ(tensor->size(0), i + 1);
    EXPECT_EQ(tensor->size(1), i + 2);

    // Verify the tensor uses the correct data pointer
    EXPECT_EQ(tensor->mutable_data_ptr(), data);
  }

  // Verify all tensors have different data pointers
  for (int i = 0; i < num_tensors; i++) {
    EXPECT_EQ(tensors[i]->mutable_data_ptr(), data_ptrs[i]);
    for (int j = i + 1; j < num_tensors; j++) {
      EXPECT_NE(tensors[i]->mutable_data_ptr(), tensors[j]->mutable_data_ptr());
    }
  }
}

// Test deletion of tensor created from blob (should not free the original
// memory)
TEST_F(AOTITorchCreateTensorFromBlobV2Test, DeletionDoesNotFreeOriginalMemory) {
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = calculate_contiguous_strides(sizes);

  size_t bytes = calculate_numel(sizes) * sizeof(float);
  void* data = allocate_cuda_memory(bytes);
  ASSERT_NE(data, nullptr);

  Tensor* tensor;
  AOTITorchError error = aoti_torch_create_tensor_from_blob_v2(
      data,
      sizes.size(),
      sizes.data(),
      strides.data(),
      0, // storage_offset
      static_cast<int32_t>(SupportedDTypes::FLOAT32),
      static_cast<int32_t>(SupportedDevices::CUDA),
      0, // device index
      &tensor,
      0, // layout
      nullptr, // opaque_metadata
      0); // opaque_metadata_size

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(tensor, nullptr);

  // Delete the tensor - this should NOT free the original memory
  error = aoti_torch_delete_tensor_object(tensor);
  EXPECT_EQ(error, Error::Ok);

  // The original memory should still be valid (we'll free it in teardown)
  // We can't easily test if the memory is still valid without risking crashes,
  // but the test should pass without issues if memory management is correct
}

// Test with opaque metadata
TEST_F(AOTITorchCreateTensorFromBlobV2Test, WithOpaqueMetadata) {
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = calculate_contiguous_strides(sizes);

  size_t bytes = calculate_numel(sizes) * sizeof(float);
  void* data = allocate_cuda_memory(bytes);
  ASSERT_NE(data, nullptr);

  // Create some opaque metadata
  std::vector<uint8_t> metadata = {0x01, 0x02, 0x03, 0x04};

  Tensor* tensor;
  AOTITorchError error = aoti_torch_create_tensor_from_blob_v2(
      data,
      sizes.size(),
      sizes.data(),
      strides.data(),
      0, // storage_offset
      static_cast<int32_t>(SupportedDTypes::FLOAT32),
      static_cast<int32_t>(SupportedDevices::CUDA),
      0, // device index
      &tensor,
      0, // layout
      metadata.data(), // opaque_metadata
      metadata.size()); // opaque_metadata_size

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(tensor, nullptr);

  // Check tensor properties
  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 2);
  EXPECT_EQ(tensor->size(1), 3);
}

// Test stress test with many small tensors from blobs
TEST_F(AOTITorchCreateTensorFromBlobV2Test, StressTestManySmallTensors) {
  const int num_tensors = 50; // Reduced for reasonable test time
  std::vector<Tensor*> tensors;

  for (int i = 0; i < num_tensors; i++) {
    std::vector<int64_t> sizes = {1, 1}; // Minimal size
    std::vector<int64_t> strides = calculate_contiguous_strides(sizes);

    size_t bytes = calculate_numel(sizes) * sizeof(float);
    void* data = allocate_cuda_memory(bytes);
    if (data == nullptr) {
      // Skip if we run out of memory
      continue;
    }

    Tensor* tensor;
    AOTITorchError error = aoti_torch_create_tensor_from_blob_v2(
        data,
        sizes.size(),
        sizes.data(),
        strides.data(),
        0, // storage_offset
        static_cast<int32_t>(SupportedDTypes::FLOAT32),
        static_cast<int32_t>(SupportedDevices::CUDA),
        0, // device index
        &tensor,
        0, // layout
        nullptr, // opaque_metadata
        0); // opaque_metadata_size

    if (error == Error::Ok && tensor != nullptr) {
      tensors.push_back(tensor);

      // Verify the tensor uses the correct data pointer
      EXPECT_EQ(tensor->mutable_data_ptr(), data);
    }
  }

  // Delete all created tensors
  for (Tensor* tensor : tensors) {
    AOTITorchError error = aoti_torch_delete_tensor_object(tensor);
    EXPECT_EQ(error, Error::Ok);
  }
}
