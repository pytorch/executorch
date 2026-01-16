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

// Helper to check if CUDA is available
bool isCudaAvailable() {
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  return (err == cudaSuccess && device_count > 0);
}

// Helper to calculate contiguous strides from sizes
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

// Helper to calculate numel from sizes
int64_t calculateNumel(const std::vector<int64_t>& sizes) {
  int64_t numel = 1;
  for (int64_t size : sizes) {
    numel *= size;
  }
  return numel;
}

} // namespace

// Test fixture for SlimTensor-based aoti_torch_create_tensor_from_blob_v2 tests
class AOTITorchCreateTensorFromBlobV2SlimTest : public ::testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();
  }

  void TearDown() override {
    // Clean up tensors
    for (Tensor* t : tensors_) {
      delete t;
    }
    tensors_.clear();

    // Clean up CUDA memory
    for (void* ptr : cuda_memory_) {
      if (ptr != nullptr) {
        cudaFree(ptr);
      }
    }
    cuda_memory_.clear();

    // Clean up CPU memory
    for (void* ptr : cpu_memory_) {
      if (ptr != nullptr) {
        free(ptr);
      }
    }
    cpu_memory_.clear();
  }

  void* allocateCudaMemory(size_t bytes) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err == cudaSuccess && ptr != nullptr) {
      cuda_memory_.push_back(ptr);
    }
    return ptr;
  }

  void* allocateCpuMemory(size_t bytes) {
    void* ptr = nullptr;
    int result = posix_memalign(&ptr, 16, bytes);
    if (result == 0 && ptr != nullptr) {
      cpu_memory_.push_back(ptr);
    }
    return ptr;
  }

  void trackTensor(Tensor* t) {
    if (t != nullptr) {
      tensors_.push_back(t);
    }
  }

 private:
  std::vector<Tensor*> tensors_;
  std::vector<void*> cuda_memory_;
  std::vector<void*> cpu_memory_;
};

// ============================================================================
// Common test body - parameterized by device type
// ============================================================================

void runBasicFromBlobTest(
    AOTITorchCreateTensorFromBlobV2SlimTest* fixture,
    void* data,
    int32_t device_type,
    int32_t device_index) {
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = calculateContiguousStrides(sizes);

  Tensor* tensor = nullptr;
  AOTITorchError error = aoti_torch_create_tensor_from_blob_v2(
      data,
      sizes.size(),
      sizes.data(),
      strides.data(),
      0, // storage_offset
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      device_type,
      device_index,
      &tensor,
      0, // layout
      nullptr, // opaque_metadata
      0); // opaque_metadata_size

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(tensor, nullptr);

  // Check tensor properties
  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 2);
  EXPECT_EQ(tensor->size(1), 3);
  EXPECT_EQ(tensor->numel(), 6);
  EXPECT_EQ(
      static_cast<int32_t>(tensor->dtype()),
      static_cast<int32_t>(slim_c10::ScalarType::Float));

  // Verify the tensor uses the same data pointer (non-owning)
  EXPECT_EQ(tensor->data_ptr(), data);

  // Cleanup - tensor should NOT free the original memory
  delete tensor;
}

void runScalarFromBlobTest(
    AOTITorchCreateTensorFromBlobV2SlimTest* fixture,
    void* data,
    int32_t device_type,
    int32_t device_index) {
  std::vector<int64_t> sizes = {}; // 0D tensor
  std::vector<int64_t> strides = {};

  Tensor* tensor = nullptr;
  AOTITorchError error = aoti_torch_create_tensor_from_blob_v2(
      data,
      sizes.size(),
      sizes.data(),
      strides.data(),
      0, // storage_offset
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      device_type,
      device_index,
      &tensor,
      0, // layout
      nullptr, // opaque_metadata
      0); // opaque_metadata_size

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(tensor, nullptr);

  EXPECT_EQ(tensor->dim(), 0);
  EXPECT_EQ(tensor->numel(), 1);
  EXPECT_EQ(tensor->data_ptr(), data);

  delete tensor;
}

void runMultiDimensionalFromBlobTest(
    AOTITorchCreateTensorFromBlobV2SlimTest* fixture,
    void* data,
    int32_t device_type,
    int32_t device_index) {
  std::vector<int64_t> sizes = {2, 3, 4};
  std::vector<int64_t> strides = calculateContiguousStrides(sizes);

  Tensor* tensor = nullptr;
  AOTITorchError error = aoti_torch_create_tensor_from_blob_v2(
      data,
      sizes.size(),
      sizes.data(),
      strides.data(),
      0, // storage_offset
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      device_type,
      device_index,
      &tensor,
      0, // layout
      nullptr, // opaque_metadata
      0); // opaque_metadata_size

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(tensor, nullptr);

  EXPECT_EQ(tensor->dim(), 3);
  EXPECT_EQ(tensor->size(0), 2);
  EXPECT_EQ(tensor->size(1), 3);
  EXPECT_EQ(tensor->size(2), 4);
  EXPECT_EQ(tensor->numel(), 24);
  EXPECT_EQ(tensor->data_ptr(), data);

  delete tensor;
}

void runCustomStridesFromBlobTest(
    AOTITorchCreateTensorFromBlobV2SlimTest* fixture,
    void* data,
    int32_t device_type,
    int32_t device_index) {
  std::vector<int64_t> sizes = {3, 4};
  std::vector<int64_t> strides = {1, 3}; // Column-major

  Tensor* tensor = nullptr;
  AOTITorchError error = aoti_torch_create_tensor_from_blob_v2(
      data,
      sizes.size(),
      sizes.data(),
      strides.data(),
      0, // storage_offset
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      device_type,
      device_index,
      &tensor,
      0, // layout
      nullptr, // opaque_metadata
      0); // opaque_metadata_size

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(tensor, nullptr);

  EXPECT_EQ(tensor->stride(0), 1);
  EXPECT_EQ(tensor->stride(1), 3);
  EXPECT_FALSE(tensor->is_contiguous());
  EXPECT_EQ(tensor->data_ptr(), data);

  delete tensor;
}

void runStorageOffsetFromBlobTest(
    AOTITorchCreateTensorFromBlobV2SlimTest* fixture,
    void* data,
    int32_t device_type,
    int32_t device_index) {
  std::vector<int64_t> sizes = {2, 2};
  std::vector<int64_t> strides = calculateContiguousStrides(sizes);

  Tensor* tensor = nullptr;
  AOTITorchError error = aoti_torch_create_tensor_from_blob_v2(
      data,
      sizes.size(),
      sizes.data(),
      strides.data(),
      2, // storage_offset = 2 elements
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      device_type,
      device_index,
      &tensor,
      0, // layout
      nullptr, // opaque_metadata
      0); // opaque_metadata_size

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(tensor, nullptr);

  EXPECT_EQ(tensor->storage_offset(), 2);
  // data_ptr should point to base + offset * itemsize
  char* expected_ptr = static_cast<char*>(data) + 2 * sizeof(float);
  EXPECT_EQ(tensor->data_ptr(), expected_ptr);

  delete tensor;
}

// ============================================================================
// CPU Tests
// ============================================================================

TEST_F(AOTITorchCreateTensorFromBlobV2SlimTest, BasicFunctionality_CPU) {
  size_t bytes = 6 * sizeof(float);
  void* data = allocateCpuMemory(bytes);
  ASSERT_NE(data, nullptr);

  runBasicFromBlobTest(
      this, data, static_cast<int32_t>(slim_c10::DeviceType::CPU), 0);
}

TEST_F(AOTITorchCreateTensorFromBlobV2SlimTest, ScalarTensor_CPU) {
  size_t bytes = sizeof(float);
  void* data = allocateCpuMemory(bytes);
  ASSERT_NE(data, nullptr);

  runScalarFromBlobTest(
      this, data, static_cast<int32_t>(slim_c10::DeviceType::CPU), 0);
}

TEST_F(AOTITorchCreateTensorFromBlobV2SlimTest, MultiDimensional_CPU) {
  size_t bytes = 24 * sizeof(float);
  void* data = allocateCpuMemory(bytes);
  ASSERT_NE(data, nullptr);

  runMultiDimensionalFromBlobTest(
      this, data, static_cast<int32_t>(slim_c10::DeviceType::CPU), 0);
}

TEST_F(AOTITorchCreateTensorFromBlobV2SlimTest, CustomStrides_CPU) {
  size_t bytes = 12 * sizeof(float);
  void* data = allocateCpuMemory(bytes);
  ASSERT_NE(data, nullptr);

  runCustomStridesFromBlobTest(
      this, data, static_cast<int32_t>(slim_c10::DeviceType::CPU), 0);
}

TEST_F(AOTITorchCreateTensorFromBlobV2SlimTest, StorageOffset_CPU) {
  // Allocate extra space for offset
  size_t bytes = 6 * sizeof(float); // 2 for offset + 4 for tensor
  void* data = allocateCpuMemory(bytes);
  ASSERT_NE(data, nullptr);

  runStorageOffsetFromBlobTest(
      this, data, static_cast<int32_t>(slim_c10::DeviceType::CPU), 0);
}

// ============================================================================
// CUDA Tests
// ============================================================================

TEST_F(AOTITorchCreateTensorFromBlobV2SlimTest, BasicFunctionality_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }

  size_t bytes = 6 * sizeof(float);
  void* data = allocateCudaMemory(bytes);
  ASSERT_NE(data, nullptr);

  runBasicFromBlobTest(
      this, data, static_cast<int32_t>(slim_c10::DeviceType::CUDA), 0);
}

TEST_F(AOTITorchCreateTensorFromBlobV2SlimTest, ScalarTensor_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }

  size_t bytes = sizeof(float);
  void* data = allocateCudaMemory(bytes);
  ASSERT_NE(data, nullptr);

  runScalarFromBlobTest(
      this, data, static_cast<int32_t>(slim_c10::DeviceType::CUDA), 0);
}

TEST_F(AOTITorchCreateTensorFromBlobV2SlimTest, MultiDimensional_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }

  size_t bytes = 24 * sizeof(float);
  void* data = allocateCudaMemory(bytes);
  ASSERT_NE(data, nullptr);

  runMultiDimensionalFromBlobTest(
      this, data, static_cast<int32_t>(slim_c10::DeviceType::CUDA), 0);
}

TEST_F(AOTITorchCreateTensorFromBlobV2SlimTest, CustomStrides_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }

  size_t bytes = 12 * sizeof(float);
  void* data = allocateCudaMemory(bytes);
  ASSERT_NE(data, nullptr);

  runCustomStridesFromBlobTest(
      this, data, static_cast<int32_t>(slim_c10::DeviceType::CUDA), 0);
}

TEST_F(AOTITorchCreateTensorFromBlobV2SlimTest, StorageOffset_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }

  // Allocate extra space for offset
  size_t bytes = 6 * sizeof(float);
  void* data = allocateCudaMemory(bytes);
  ASSERT_NE(data, nullptr);

  runStorageOffsetFromBlobTest(
      this, data, static_cast<int32_t>(slim_c10::DeviceType::CUDA), 0);
}

// ============================================================================
// Verify Non-Owning Behavior
// ============================================================================

TEST_F(AOTITorchCreateTensorFromBlobV2SlimTest, NonOwningBehavior_CPU) {
  size_t bytes = 6 * sizeof(float);
  void* data = allocateCpuMemory(bytes);
  ASSERT_NE(data, nullptr);

  // Write a pattern
  float* float_data = static_cast<float*>(data);
  float_data[0] = 42.0f;

  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = calculateContiguousStrides(sizes);

  Tensor* tensor = nullptr;
  AOTITorchError error = aoti_torch_create_tensor_from_blob_v2(
      data,
      sizes.size(),
      sizes.data(),
      strides.data(),
      0,
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0,
      &tensor,
      0,
      nullptr,
      0);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(tensor, nullptr);

  // Delete tensor - memory should NOT be freed
  delete tensor;
  tensor = nullptr;

  // Memory should still be accessible
  EXPECT_FLOAT_EQ(float_data[0], 42.0f);
}

TEST_F(AOTITorchCreateTensorFromBlobV2SlimTest, NonOwningBehavior_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }

  size_t bytes = 6 * sizeof(float);
  void* data = allocateCudaMemory(bytes);
  ASSERT_NE(data, nullptr);

  // Write a pattern
  float pattern = 42.0f;
  cudaMemcpy(data, &pattern, sizeof(float), cudaMemcpyHostToDevice);

  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = calculateContiguousStrides(sizes);

  Tensor* tensor = nullptr;
  AOTITorchError error = aoti_torch_create_tensor_from_blob_v2(
      data,
      sizes.size(),
      sizes.data(),
      strides.data(),
      0,
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CUDA),
      0,
      &tensor,
      0,
      nullptr,
      0);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(tensor, nullptr);

  // Delete tensor - memory should NOT be freed
  delete tensor;
  tensor = nullptr;

  // Memory should still be accessible
  float readback = 0.0f;
  cudaError_t cuda_err =
      cudaMemcpy(&readback, data, sizeof(float), cudaMemcpyDeviceToHost);
  EXPECT_EQ(cuda_err, cudaSuccess);
  EXPECT_FLOAT_EQ(readback, 42.0f);
}

// ============================================================================
// Error Cases
// ============================================================================

TEST_F(AOTITorchCreateTensorFromBlobV2SlimTest, NullDataPointer) {
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = calculateContiguousStrides(sizes);

  Tensor* tensor = nullptr;
  AOTITorchError error = aoti_torch_create_tensor_from_blob_v2(
      nullptr, // null data
      sizes.size(),
      sizes.data(),
      strides.data(),
      0,
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0,
      &tensor,
      0,
      nullptr,
      0);

  EXPECT_EQ(error, Error::InvalidArgument);
}

TEST_F(AOTITorchCreateTensorFromBlobV2SlimTest, NullReturnPointer) {
  size_t bytes = 6 * sizeof(float);
  void* data = allocateCpuMemory(bytes);
  ASSERT_NE(data, nullptr);

  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = calculateContiguousStrides(sizes);

  AOTITorchError error = aoti_torch_create_tensor_from_blob_v2(
      data,
      sizes.size(),
      sizes.data(),
      strides.data(),
      0,
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0,
      nullptr, // null return pointer
      0,
      nullptr,
      0);

  EXPECT_EQ(error, Error::InvalidArgument);
}

// ============================================================================
// Verify Device Properties
// ============================================================================

TEST_F(AOTITorchCreateTensorFromBlobV2SlimTest, VerifyCPUDevice) {
  size_t bytes = 6 * sizeof(float);
  void* data = allocateCpuMemory(bytes);
  ASSERT_NE(data, nullptr);

  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = calculateContiguousStrides(sizes);

  Tensor* tensor = nullptr;
  AOTITorchError error = aoti_torch_create_tensor_from_blob_v2(
      data,
      sizes.size(),
      sizes.data(),
      strides.data(),
      0,
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0,
      &tensor,
      0,
      nullptr,
      0);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(tensor, nullptr);

  EXPECT_TRUE(tensor->is_cpu());
  EXPECT_FALSE(tensor->is_cuda());
  EXPECT_EQ(tensor->device_type(), slim_c10::DeviceType::CPU);

  delete tensor;
}

TEST_F(AOTITorchCreateTensorFromBlobV2SlimTest, VerifyCUDADevice) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }

  size_t bytes = 6 * sizeof(float);
  void* data = allocateCudaMemory(bytes);
  ASSERT_NE(data, nullptr);

  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = calculateContiguousStrides(sizes);

  Tensor* tensor = nullptr;
  AOTITorchError error = aoti_torch_create_tensor_from_blob_v2(
      data,
      sizes.size(),
      sizes.data(),
      strides.data(),
      0,
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CUDA),
      0,
      &tensor,
      0,
      nullptr,
      0);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(tensor, nullptr);

  EXPECT_FALSE(tensor->is_cpu());
  EXPECT_TRUE(tensor->is_cuda());
  EXPECT_EQ(tensor->device_type(), slim_c10::DeviceType::CUDA);

  delete tensor;
}
