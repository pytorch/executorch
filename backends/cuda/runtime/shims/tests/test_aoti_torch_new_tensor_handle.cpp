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

#include <executorch/backends/aoti/common_shims_slim.h>
#include <executorch/backends/aoti/slim/c10/core/Device.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/platform.h>

using namespace executorch::backends::cuda;
using namespace executorch::backends::aoti;
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

class AOTITorchNewTensorHandleSlimTest : public ::testing::Test {
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
// Basic Functionality Tests
// ============================================================================

TEST_F(AOTITorchNewTensorHandleSlimTest, BasicFunctionality_CPU) {
  std::vector<int64_t> sizes = {2, 3};
  Tensor* orig_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(orig_tensor, nullptr);

  Tensor* new_tensor;
  AOTITorchError error = aoti_torch_new_tensor_handle(orig_tensor, &new_tensor);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(new_tensor, nullptr);

  EXPECT_EQ(new_tensor->dim(), orig_tensor->dim());
  EXPECT_EQ(new_tensor->size(0), orig_tensor->size(0));
  EXPECT_EQ(new_tensor->size(1), orig_tensor->size(1));
  EXPECT_EQ(new_tensor->numel(), orig_tensor->numel());

  EXPECT_EQ(new_tensor->data_ptr(), orig_tensor->data_ptr());

  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(new_tensor), Error::Ok);
}

TEST_F(AOTITorchNewTensorHandleSlimTest, NullOriginalTensor) {
  Tensor* new_tensor;
  AOTITorchError error = aoti_torch_new_tensor_handle(nullptr, &new_tensor);

  EXPECT_EQ(error, Error::InvalidArgument);
}

TEST_F(AOTITorchNewTensorHandleSlimTest, NullNewHandle) {
  std::vector<int64_t> sizes = {2, 3};
  Tensor* orig_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(orig_tensor, nullptr);

  AOTITorchError error = aoti_torch_new_tensor_handle(orig_tensor, nullptr);

  EXPECT_EQ(error, Error::InvalidArgument);

  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
}

// ============================================================================
// Memory Sharing Tests
// ============================================================================

TEST_F(AOTITorchNewTensorHandleSlimTest, MemorySharing_CPU) {
  std::vector<int64_t> sizes = {3, 4};
  Tensor* orig_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(orig_tensor, nullptr);

  void* orig_ptr = orig_tensor->data_ptr();
  ASSERT_NE(orig_ptr, nullptr);

  Tensor* new_tensor;
  AOTITorchError error = aoti_torch_new_tensor_handle(orig_tensor, &new_tensor);
  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(new_tensor, nullptr);

  void* new_ptr = new_tensor->data_ptr();
  EXPECT_EQ(orig_ptr, new_ptr);

  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);

  void* still_valid_ptr = new_tensor->data_ptr();
  EXPECT_EQ(still_valid_ptr, new_ptr);

  EXPECT_EQ(aoti_torch_delete_tensor_object(new_tensor), Error::Ok);
}

TEST_F(AOTITorchNewTensorHandleSlimTest, MultipleHandles_CPU) {
  std::vector<int64_t> sizes = {2, 3};
  Tensor* orig_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(orig_tensor, nullptr);

  void* orig_ptr = orig_tensor->data_ptr();

  std::vector<Tensor*> handles;
  const int num_handles = 5;

  for (int i = 0; i < num_handles; i++) {
    Tensor* new_tensor;
    AOTITorchError error =
        aoti_torch_new_tensor_handle(orig_tensor, &new_tensor);
    EXPECT_EQ(error, Error::Ok);
    ASSERT_NE(new_tensor, nullptr);
    EXPECT_EQ(new_tensor->data_ptr(), orig_ptr);
    handles.push_back(new_tensor);
  }

  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);

  for (Tensor* handle : handles) {
    EXPECT_EQ(handle->data_ptr(), orig_ptr);
    EXPECT_EQ(handle->dim(), 2);
    EXPECT_EQ(handle->size(0), 2);
    EXPECT_EQ(handle->size(1), 3);
  }

  for (Tensor* handle : handles) {
    EXPECT_EQ(aoti_torch_delete_tensor_object(handle), Error::Ok);
  }
}

// ============================================================================
// Tensor Property Tests
// ============================================================================

TEST_F(AOTITorchNewTensorHandleSlimTest, CustomStrides_CPU) {
  std::vector<int64_t> sizes = {3, 4};
  std::vector<int64_t> strides = {4, 1}; // Row-major strides
  Tensor* orig_tensor = createTestTensor(
      sizes,
      strides,
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(orig_tensor, nullptr);

  Tensor* new_tensor;
  AOTITorchError error = aoti_torch_new_tensor_handle(orig_tensor, &new_tensor);
  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(new_tensor, nullptr);

  EXPECT_EQ(orig_tensor->stride(0), new_tensor->stride(0));
  EXPECT_EQ(orig_tensor->stride(1), new_tensor->stride(1));

  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(new_tensor), Error::Ok);
}

TEST_F(AOTITorchNewTensorHandleSlimTest, BFloat16Tensor_CPU) {
  std::vector<int64_t> sizes = {2, 3, 4};
  Tensor* orig_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::BFloat16),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(orig_tensor, nullptr);

  Tensor* new_tensor;
  AOTITorchError error = aoti_torch_new_tensor_handle(orig_tensor, &new_tensor);
  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(new_tensor, nullptr);

  EXPECT_EQ(new_tensor->itemsize(), 2);

  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(new_tensor), Error::Ok);
}

TEST_F(AOTITorchNewTensorHandleSlimTest, ScalarTensor_CPU) {
  std::vector<int64_t> sizes = {};
  Tensor* orig_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(orig_tensor, nullptr);
  EXPECT_EQ(orig_tensor->dim(), 0);

  Tensor* new_tensor;
  AOTITorchError error = aoti_torch_new_tensor_handle(orig_tensor, &new_tensor);
  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(new_tensor, nullptr);

  EXPECT_EQ(new_tensor->dim(), 0);
  EXPECT_EQ(new_tensor->numel(), 1);

  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(new_tensor), Error::Ok);
}

TEST_F(AOTITorchNewTensorHandleSlimTest, LargeMultiDimensionalTensor_CPU) {
  std::vector<int64_t> sizes = {10, 20, 30};
  Tensor* orig_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(orig_tensor, nullptr);

  Tensor* new_tensor;
  AOTITorchError error = aoti_torch_new_tensor_handle(orig_tensor, &new_tensor);
  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(new_tensor, nullptr);

  EXPECT_EQ(new_tensor->dim(), 3);
  EXPECT_EQ(new_tensor->size(0), 10);
  EXPECT_EQ(new_tensor->size(1), 20);
  EXPECT_EQ(new_tensor->size(2), 30);
  EXPECT_EQ(new_tensor->numel(), 6000);

  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(new_tensor), Error::Ok);
}

// ============================================================================
// Handle Chain Tests
// ============================================================================

TEST_F(AOTITorchNewTensorHandleSlimTest, HandleChain_CPU) {
  std::vector<int64_t> sizes = {2, 3};
  Tensor* orig_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(orig_tensor, nullptr);

  void* orig_ptr = orig_tensor->data_ptr();

  Tensor* handle1;
  AOTITorchError error = aoti_torch_new_tensor_handle(orig_tensor, &handle1);
  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(handle1, nullptr);
  EXPECT_EQ(handle1->data_ptr(), orig_ptr);

  Tensor* handle2;
  error = aoti_torch_new_tensor_handle(handle1, &handle2);
  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(handle2, nullptr);
  EXPECT_EQ(handle2->data_ptr(), orig_ptr);

  EXPECT_EQ(aoti_torch_delete_tensor_object(handle2), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(handle1), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
}

TEST_F(AOTITorchNewTensorHandleSlimTest, ReferenceCountingTest_CPU) {
  std::vector<int64_t> sizes = {2, 3};
  Tensor* orig_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(orig_tensor, nullptr);

  void* orig_ptr = orig_tensor->data_ptr();

  Tensor* handle1;
  Tensor* handle2;
  Tensor* handle3;

  EXPECT_EQ(aoti_torch_new_tensor_handle(orig_tensor, &handle1), Error::Ok);
  EXPECT_EQ(aoti_torch_new_tensor_handle(orig_tensor, &handle2), Error::Ok);
  EXPECT_EQ(aoti_torch_new_tensor_handle(orig_tensor, &handle3), Error::Ok);

  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);

  EXPECT_EQ(handle1->data_ptr(), orig_ptr);
  EXPECT_EQ(handle2->data_ptr(), orig_ptr);
  EXPECT_EQ(handle3->data_ptr(), orig_ptr);

  EXPECT_EQ(aoti_torch_delete_tensor_object(handle1), Error::Ok);

  EXPECT_EQ(handle2->data_ptr(), orig_ptr);
  EXPECT_EQ(handle3->data_ptr(), orig_ptr);

  EXPECT_EQ(aoti_torch_delete_tensor_object(handle2), Error::Ok);

  EXPECT_EQ(handle3->data_ptr(), orig_ptr);

  EXPECT_EQ(aoti_torch_delete_tensor_object(handle3), Error::Ok);
}

// ============================================================================
// Different Dtype Tests
// ============================================================================

TEST_F(AOTITorchNewTensorHandleSlimTest, Int64Tensor_CPU) {
  std::vector<int64_t> sizes = {2, 3};
  Tensor* orig_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Long),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(orig_tensor, nullptr);

  Tensor* new_tensor;
  AOTITorchError error = aoti_torch_new_tensor_handle(orig_tensor, &new_tensor);
  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(new_tensor, nullptr);

  EXPECT_EQ(new_tensor->itemsize(), 8);

  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(new_tensor), Error::Ok);
}

TEST_F(AOTITorchNewTensorHandleSlimTest, IncontiguousLayout_CPU) {
  std::vector<int64_t> sizes = {3, 4};
  std::vector<int64_t> strides = {1, 3}; // Column-major (incontiguous)
  Tensor* orig_tensor = createTestTensor(
      sizes,
      strides,
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(orig_tensor, nullptr);

  Tensor* new_tensor;
  AOTITorchError error = aoti_torch_new_tensor_handle(orig_tensor, &new_tensor);
  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(new_tensor, nullptr);

  EXPECT_EQ(new_tensor->stride(0), 1);
  EXPECT_EQ(new_tensor->stride(1), 3);

  EXPECT_EQ(new_tensor->data_ptr(), orig_tensor->data_ptr());

  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(new_tensor), Error::Ok);
}

// ============================================================================
// CUDA Tests
// ============================================================================

TEST_F(AOTITorchNewTensorHandleSlimTest, BasicFunctionality_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }

  std::vector<int64_t> sizes = {2, 3};
  Tensor* orig_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CUDA),
      0);
  ASSERT_NE(orig_tensor, nullptr);
  EXPECT_TRUE(orig_tensor->is_cuda());

  Tensor* new_tensor;
  AOTITorchError error = aoti_torch_new_tensor_handle(orig_tensor, &new_tensor);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_NE(new_tensor, nullptr);
  EXPECT_TRUE(new_tensor->is_cuda());

  EXPECT_EQ(new_tensor->data_ptr(), orig_tensor->data_ptr());

  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(new_tensor), Error::Ok);
}

TEST_F(AOTITorchNewTensorHandleSlimTest, MemorySharing_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }

  std::vector<int64_t> sizes = {3, 4};
  Tensor* orig_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CUDA),
      0);
  ASSERT_NE(orig_tensor, nullptr);

  void* orig_ptr = orig_tensor->data_ptr();
  ASSERT_NE(orig_ptr, nullptr);

  Tensor* new_tensor;
  AOTITorchError error = aoti_torch_new_tensor_handle(orig_tensor, &new_tensor);
  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(new_tensor, nullptr);

  void* new_ptr = new_tensor->data_ptr();
  EXPECT_EQ(orig_ptr, new_ptr);

  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);

  void* still_valid_ptr = new_tensor->data_ptr();
  EXPECT_EQ(still_valid_ptr, new_ptr);

  EXPECT_EQ(aoti_torch_delete_tensor_object(new_tensor), Error::Ok);
}

TEST_F(AOTITorchNewTensorHandleSlimTest, MultipleHandles_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }

  std::vector<int64_t> sizes = {2, 3};
  Tensor* orig_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CUDA),
      0);
  ASSERT_NE(orig_tensor, nullptr);

  void* orig_ptr = orig_tensor->data_ptr();

  std::vector<Tensor*> handles;
  const int num_handles = 5;

  for (int i = 0; i < num_handles; i++) {
    Tensor* new_tensor;
    AOTITorchError error =
        aoti_torch_new_tensor_handle(orig_tensor, &new_tensor);
    EXPECT_EQ(error, Error::Ok);
    ASSERT_NE(new_tensor, nullptr);
    EXPECT_EQ(new_tensor->data_ptr(), orig_ptr);
    handles.push_back(new_tensor);
  }

  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);

  for (Tensor* handle : handles) {
    EXPECT_EQ(handle->data_ptr(), orig_ptr);
    EXPECT_TRUE(handle->is_cuda());
  }

  for (Tensor* handle : handles) {
    EXPECT_EQ(aoti_torch_delete_tensor_object(handle), Error::Ok);
  }
}

TEST_F(AOTITorchNewTensorHandleSlimTest, ReferenceCountingTest_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }

  std::vector<int64_t> sizes = {2, 3};
  Tensor* orig_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CUDA),
      0);
  ASSERT_NE(orig_tensor, nullptr);

  void* orig_ptr = orig_tensor->data_ptr();

  Tensor* handle1;
  Tensor* handle2;
  Tensor* handle3;

  EXPECT_EQ(aoti_torch_new_tensor_handle(orig_tensor, &handle1), Error::Ok);
  EXPECT_EQ(aoti_torch_new_tensor_handle(orig_tensor, &handle2), Error::Ok);
  EXPECT_EQ(aoti_torch_new_tensor_handle(orig_tensor, &handle3), Error::Ok);

  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);

  EXPECT_EQ(handle1->data_ptr(), orig_ptr);
  EXPECT_EQ(handle2->data_ptr(), orig_ptr);
  EXPECT_EQ(handle3->data_ptr(), orig_ptr);

  EXPECT_EQ(aoti_torch_delete_tensor_object(handle1), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(handle2), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(handle3), Error::Ok);
}

// ============================================================================
// Mixed Device Tests
// ============================================================================

TEST_F(AOTITorchNewTensorHandleSlimTest, MixedDeviceHandles) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }

  std::vector<int64_t> sizes = {2, 3};

  Tensor* cpu_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(cpu_tensor, nullptr);
  EXPECT_TRUE(cpu_tensor->is_cpu());

  Tensor* cuda_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CUDA),
      0);
  ASSERT_NE(cuda_tensor, nullptr);
  EXPECT_TRUE(cuda_tensor->is_cuda());

  Tensor* cpu_handle;
  Tensor* cuda_handle;

  EXPECT_EQ(aoti_torch_new_tensor_handle(cpu_tensor, &cpu_handle), Error::Ok);
  EXPECT_EQ(aoti_torch_new_tensor_handle(cuda_tensor, &cuda_handle), Error::Ok);

  EXPECT_TRUE(cpu_handle->is_cpu());
  EXPECT_TRUE(cuda_handle->is_cuda());
  EXPECT_NE(cpu_handle->data_ptr(), cuda_handle->data_ptr());

  EXPECT_EQ(aoti_torch_delete_tensor_object(cpu_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(cuda_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(cpu_handle), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(cuda_handle), Error::Ok);
}

// ============================================================================
// Stress Tests
// ============================================================================

TEST_F(AOTITorchNewTensorHandleSlimTest, StressTestManyHandles_CPU) {
  std::vector<int64_t> sizes = {2, 3};
  Tensor* orig_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(orig_tensor, nullptr);

  void* orig_ptr = orig_tensor->data_ptr();

  const int num_handles = 100;
  std::vector<Tensor*> handles;

  for (int i = 0; i < num_handles; i++) {
    Tensor* new_tensor;
    AOTITorchError error =
        aoti_torch_new_tensor_handle(orig_tensor, &new_tensor);
    EXPECT_EQ(error, Error::Ok);
    ASSERT_NE(new_tensor, nullptr);
    EXPECT_EQ(new_tensor->data_ptr(), orig_ptr);
    handles.push_back(new_tensor);
  }

  EXPECT_EQ(aoti_torch_delete_tensor_object(orig_tensor), Error::Ok);

  for (Tensor* handle : handles) {
    EXPECT_EQ(handle->data_ptr(), orig_ptr);
  }

  for (Tensor* handle : handles) {
    EXPECT_EQ(aoti_torch_delete_tensor_object(handle), Error::Ok);
  }
}
