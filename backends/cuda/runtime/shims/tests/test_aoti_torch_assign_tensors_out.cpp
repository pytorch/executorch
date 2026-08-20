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

class AOTITorchAssignTensorsOutSlimTest : public ::testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();
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

TEST_F(AOTITorchAssignTensorsOutSlimTest, BasicFunctionality_CPU) {
  std::vector<int64_t> sizes = {2, 3};
  Tensor* src_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(src_tensor, nullptr);

  // Store expected properties before move
  int64_t expected_dim = src_tensor->dim();
  int64_t expected_size0 = src_tensor->size(0);
  int64_t expected_size1 = src_tensor->size(1);
  size_t expected_numel = src_tensor->numel();
  void* expected_data_ptr = src_tensor->data_ptr();

  Tensor* dst_tensor = nullptr;
  AOTITorchError error = aoti_torch_assign_tensors_out(src_tensor, &dst_tensor);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(dst_tensor, nullptr);

  // Verify destination tensor has the moved properties
  EXPECT_EQ(dst_tensor->dim(), expected_dim);
  EXPECT_EQ(dst_tensor->size(0), expected_size0);
  EXPECT_EQ(dst_tensor->size(1), expected_size1);
  EXPECT_EQ(dst_tensor->numel(), expected_numel);
  EXPECT_EQ(dst_tensor->data_ptr(), expected_data_ptr);

  // Source tensor is now in undefined state after move - just delete it
  // (accessing src_tensor properties is undefined behavior after move)
  delete src_tensor; // Direct delete since it's in undefined state
  EXPECT_EQ(aoti_torch_delete_tensor_object(dst_tensor), Error::Ok);
}

TEST_F(AOTITorchAssignTensorsOutSlimTest, NullSrc) {
  Tensor* dst_tensor = nullptr;
  AOTITorchError error = aoti_torch_assign_tensors_out(nullptr, &dst_tensor);

  EXPECT_EQ(error, Error::InvalidArgument);
}

TEST_F(AOTITorchAssignTensorsOutSlimTest, NullDst) {
  std::vector<int64_t> sizes = {2, 3};
  Tensor* src_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(src_tensor, nullptr);

  AOTITorchError error = aoti_torch_assign_tensors_out(src_tensor, nullptr);

  EXPECT_EQ(error, Error::InvalidArgument);

  EXPECT_EQ(aoti_torch_delete_tensor_object(src_tensor), Error::Ok);
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST_F(AOTITorchAssignTensorsOutSlimTest, SourceBecamesUndefinedAfterMove_CPU) {
  std::vector<int64_t> sizes = {3, 4};
  Tensor* src_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(src_tensor, nullptr);

  void* original_ptr = src_tensor->data_ptr();
  ASSERT_NE(original_ptr, nullptr);

  Tensor* dst_tensor = nullptr;
  AOTITorchError error = aoti_torch_assign_tensors_out(src_tensor, &dst_tensor);
  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(dst_tensor, nullptr);

  // Destination has the original pointer
  EXPECT_EQ(dst_tensor->data_ptr(), original_ptr);

  // Source tensor should still alive.
  EXPECT_TRUE(src_tensor->defined());

  // Clean up - delete in this order since src is undefined
  delete src_tensor;
  EXPECT_EQ(aoti_torch_delete_tensor_object(dst_tensor), Error::Ok);
}

// ============================================================================
// Tensor Property Tests
// ============================================================================

TEST_F(AOTITorchAssignTensorsOutSlimTest, CustomStrides_CPU) {
  std::vector<int64_t> sizes = {3, 4};
  std::vector<int64_t> strides = {4, 1};
  Tensor* src_tensor = createTestTensor(
      sizes,
      strides,
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(src_tensor, nullptr);

  // Store expected strides before move
  int64_t expected_stride0 = src_tensor->stride(0);
  int64_t expected_stride1 = src_tensor->stride(1);

  Tensor* dst_tensor = nullptr;
  AOTITorchError error = aoti_torch_assign_tensors_out(src_tensor, &dst_tensor);
  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(dst_tensor, nullptr);

  // Verify destination has the expected strides
  EXPECT_EQ(dst_tensor->stride(0), expected_stride0);
  EXPECT_EQ(dst_tensor->stride(1), expected_stride1);

  delete src_tensor; // Source is undefined after move
  EXPECT_EQ(aoti_torch_delete_tensor_object(dst_tensor), Error::Ok);
}

TEST_F(AOTITorchAssignTensorsOutSlimTest, ScalarTensor_CPU) {
  std::vector<int64_t> sizes = {};
  Tensor* src_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(src_tensor, nullptr);
  EXPECT_EQ(src_tensor->dim(), 0);

  Tensor* dst_tensor = nullptr;
  AOTITorchError error = aoti_torch_assign_tensors_out(src_tensor, &dst_tensor);
  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(dst_tensor, nullptr);

  EXPECT_EQ(dst_tensor->dim(), 0);
  EXPECT_EQ(dst_tensor->numel(), 1);

  EXPECT_EQ(aoti_torch_delete_tensor_object(src_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(dst_tensor), Error::Ok);
}

TEST_F(AOTITorchAssignTensorsOutSlimTest, LargeMultiDimensionalTensor_CPU) {
  std::vector<int64_t> sizes = {10, 20, 30};
  Tensor* src_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(src_tensor, nullptr);

  Tensor* dst_tensor = nullptr;
  AOTITorchError error = aoti_torch_assign_tensors_out(src_tensor, &dst_tensor);
  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(dst_tensor, nullptr);

  EXPECT_EQ(dst_tensor->dim(), 3);
  EXPECT_EQ(dst_tensor->size(0), 10);
  EXPECT_EQ(dst_tensor->size(1), 20);
  EXPECT_EQ(dst_tensor->size(2), 30);
  EXPECT_EQ(dst_tensor->numel(), 6000);

  EXPECT_EQ(aoti_torch_delete_tensor_object(src_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(dst_tensor), Error::Ok);
}

// ============================================================================
// Different Dtype Tests
// ============================================================================

TEST_F(AOTITorchAssignTensorsOutSlimTest, Int64Tensor_CPU) {
  std::vector<int64_t> sizes = {2, 3};
  Tensor* src_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Long),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(src_tensor, nullptr);

  Tensor* dst_tensor = nullptr;
  AOTITorchError error = aoti_torch_assign_tensors_out(src_tensor, &dst_tensor);
  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(dst_tensor, nullptr);

  EXPECT_EQ(dst_tensor->itemsize(), 8);

  EXPECT_EQ(aoti_torch_delete_tensor_object(src_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(dst_tensor), Error::Ok);
}

TEST_F(AOTITorchAssignTensorsOutSlimTest, BFloat16Tensor_CPU) {
  std::vector<int64_t> sizes = {2, 3, 4};
  Tensor* src_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::BFloat16),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(src_tensor, nullptr);

  Tensor* dst_tensor = nullptr;
  AOTITorchError error = aoti_torch_assign_tensors_out(src_tensor, &dst_tensor);
  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(dst_tensor, nullptr);

  EXPECT_EQ(dst_tensor->itemsize(), 2);

  EXPECT_EQ(aoti_torch_delete_tensor_object(src_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(dst_tensor), Error::Ok);
}

TEST_F(AOTITorchAssignTensorsOutSlimTest, BoolTensor_CPU) {
  std::vector<int64_t> sizes = {4};
  Tensor* src_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Bool),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(src_tensor, nullptr);

  Tensor* dst_tensor = nullptr;
  AOTITorchError error = aoti_torch_assign_tensors_out(src_tensor, &dst_tensor);
  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(dst_tensor, nullptr);

  EXPECT_EQ(dst_tensor->itemsize(), 1);

  EXPECT_EQ(aoti_torch_delete_tensor_object(src_tensor), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(dst_tensor), Error::Ok);
}

// ============================================================================
// CUDA Tests
// ============================================================================

TEST_F(AOTITorchAssignTensorsOutSlimTest, BasicFunctionality_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }

  std::vector<int64_t> sizes = {2, 3};
  Tensor* src_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CUDA),
      0);
  ASSERT_NE(src_tensor, nullptr);
  EXPECT_TRUE(src_tensor->is_cuda());

  // Store expected properties before move
  void* expected_data_ptr = src_tensor->data_ptr();

  Tensor* dst_tensor = nullptr;
  AOTITorchError error = aoti_torch_assign_tensors_out(src_tensor, &dst_tensor);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(dst_tensor, nullptr);
  EXPECT_TRUE(dst_tensor->is_cuda());
  EXPECT_EQ(dst_tensor->data_ptr(), expected_data_ptr);

  // Source tensor should still alive.
  EXPECT_TRUE(src_tensor->defined());

  delete src_tensor;
  EXPECT_EQ(aoti_torch_delete_tensor_object(dst_tensor), Error::Ok);
}

TEST_F(
    AOTITorchAssignTensorsOutSlimTest,
    SourceBecamesUndefinedAfterMove_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }

  std::vector<int64_t> sizes = {3, 4};
  Tensor* src_tensor = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CUDA),
      0);
  ASSERT_NE(src_tensor, nullptr);

  void* original_ptr = src_tensor->data_ptr();
  ASSERT_NE(original_ptr, nullptr);

  Tensor* dst_tensor = nullptr;
  AOTITorchError error = aoti_torch_assign_tensors_out(src_tensor, &dst_tensor);
  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(dst_tensor, nullptr);

  // Destination has the original pointer
  EXPECT_EQ(dst_tensor->data_ptr(), original_ptr);

  // Source tensor should still alive.
  EXPECT_TRUE(src_tensor->defined());

  delete src_tensor;
  EXPECT_EQ(aoti_torch_delete_tensor_object(dst_tensor), Error::Ok);
}

// ============================================================================
// Mixed Device Tests
// ============================================================================

TEST_F(AOTITorchAssignTensorsOutSlimTest, MixedDeviceAssignments) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }

  std::vector<int64_t> sizes = {2, 3};

  Tensor* cpu_src = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(cpu_src, nullptr);
  EXPECT_TRUE(cpu_src->is_cpu());

  Tensor* cuda_src = createTestTensor(
      sizes,
      {},
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CUDA),
      0);
  ASSERT_NE(cuda_src, nullptr);
  EXPECT_TRUE(cuda_src->is_cuda());

  Tensor* cpu_dst = nullptr;
  Tensor* cuda_dst = nullptr;

  EXPECT_EQ(aoti_torch_assign_tensors_out(cpu_src, &cpu_dst), Error::Ok);
  EXPECT_EQ(aoti_torch_assign_tensors_out(cuda_src, &cuda_dst), Error::Ok);

  EXPECT_TRUE(cpu_dst->is_cpu());
  EXPECT_TRUE(cuda_dst->is_cuda());
  EXPECT_NE(cpu_dst->data_ptr(), cuda_dst->data_ptr());

  EXPECT_EQ(aoti_torch_delete_tensor_object(cpu_src), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(cuda_src), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(cpu_dst), Error::Ok);
  EXPECT_EQ(aoti_torch_delete_tensor_object(cuda_dst), Error::Ok);
}
