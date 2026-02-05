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

} // namespace

// Test fixture for SlimTensor-based aoti_torch_empty_strided tests
class AOTITorchEmptyStridedSlimTest : public ::testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();
  }

  void TearDown() override {
    // Tensors are cleaned up via their destructors
    for (Tensor* t : tensors_) {
      delete t;
    }
    tensors_.clear();
  }

  // Track tensors for cleanup
  void trackTensor(Tensor* t) {
    if (t != nullptr) {
      tensors_.push_back(t);
    }
  }

 private:
  std::vector<Tensor*> tensors_;
};

// ============================================================================
// Common test body - parameterized by device type
// ============================================================================

void runBasicEmptyStridedTest(int32_t device_type, int32_t device_index) {
  // Test 1D tensor
  std::vector<int64_t> sizes_1d = {5};
  std::vector<int64_t> strides_1d = calculateContiguousStrides(sizes_1d);

  Tensor* tensor_1d = nullptr;
  AOTITorchError error = aoti_torch_empty_strided(
      sizes_1d.size(),
      sizes_1d.data(),
      strides_1d.data(),
      static_cast<int32_t>(slim_c10::ScalarType::Float), // dtype = 6
      device_type,
      device_index,
      &tensor_1d);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(tensor_1d, nullptr);

  // Check tensor properties
  EXPECT_EQ(tensor_1d->dim(), 1);
  EXPECT_EQ(tensor_1d->size(0), 5);
  EXPECT_EQ(tensor_1d->numel(), 5);
  EXPECT_EQ(
      static_cast<int32_t>(tensor_1d->dtype()),
      static_cast<int32_t>(slim_c10::ScalarType::Float));
  EXPECT_NE(tensor_1d->data_ptr(), nullptr);

  // Cleanup
  delete tensor_1d;
}

void runMultiDimensionalEmptyStridedTest(
    int32_t device_type,
    int32_t device_index) {
  // Test 3D tensor
  std::vector<int64_t> sizes = {2, 3, 4};
  std::vector<int64_t> strides = calculateContiguousStrides(sizes);

  Tensor* tensor = nullptr;
  AOTITorchError error = aoti_torch_empty_strided(
      sizes.size(),
      sizes.data(),
      strides.data(),
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      device_type,
      device_index,
      &tensor);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(tensor, nullptr);

  // Check tensor properties
  EXPECT_EQ(tensor->dim(), 3);
  EXPECT_EQ(tensor->size(0), 2);
  EXPECT_EQ(tensor->size(1), 3);
  EXPECT_EQ(tensor->size(2), 4);
  EXPECT_EQ(tensor->numel(), 24);

  // Check strides
  EXPECT_EQ(tensor->stride(0), 12);
  EXPECT_EQ(tensor->stride(1), 4);
  EXPECT_EQ(tensor->stride(2), 1);

  delete tensor;
}

void runScalarTensorEmptyStridedTest(
    int32_t device_type,
    int32_t device_index) {
  std::vector<int64_t> sizes = {}; // 0D tensor
  std::vector<int64_t> strides = {};

  Tensor* tensor = nullptr;
  AOTITorchError error = aoti_torch_empty_strided(
      sizes.size(),
      sizes.data(),
      strides.data(),
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      device_type,
      device_index,
      &tensor);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(tensor, nullptr);

  EXPECT_EQ(tensor->dim(), 0);
  EXPECT_EQ(tensor->numel(), 1);
  EXPECT_NE(tensor->data_ptr(), nullptr);

  delete tensor;
}

void runZeroSizedTensorEmptyStridedTest(
    int32_t device_type,
    int32_t device_index) {
  std::vector<int64_t> sizes = {0, 5};
  std::vector<int64_t> strides = calculateContiguousStrides(sizes);

  Tensor* tensor = nullptr;
  AOTITorchError error = aoti_torch_empty_strided(
      sizes.size(),
      sizes.data(),
      strides.data(),
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      device_type,
      device_index,
      &tensor);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(tensor, nullptr);

  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 0);
  EXPECT_EQ(tensor->size(1), 5);
  EXPECT_EQ(tensor->numel(), 0);

  delete tensor;
}

void runCustomStridesEmptyStridedTest(
    int32_t device_type,
    int32_t device_index) {
  // Create a transposed (column-major) tensor
  std::vector<int64_t> sizes = {3, 4};
  std::vector<int64_t> strides = {1, 3}; // Column-major

  Tensor* tensor = nullptr;
  AOTITorchError error = aoti_torch_empty_strided(
      sizes.size(),
      sizes.data(),
      strides.data(),
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      device_type,
      device_index,
      &tensor);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(tensor, nullptr);

  EXPECT_EQ(tensor->dim(), 2);
  EXPECT_EQ(tensor->size(0), 3);
  EXPECT_EQ(tensor->size(1), 4);
  EXPECT_EQ(tensor->stride(0), 1);
  EXPECT_EQ(tensor->stride(1), 3);

  // Non-contiguous due to custom strides
  EXPECT_FALSE(tensor->is_contiguous());

  delete tensor;
}

void runDifferentDtypesEmptyStridedTest(
    int32_t device_type,
    int32_t device_index) {
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = calculateContiguousStrides(sizes);

  // Test Float32
  {
    Tensor* tensor = nullptr;
    AOTITorchError error = aoti_torch_empty_strided(
        sizes.size(),
        sizes.data(),
        strides.data(),
        static_cast<int32_t>(slim_c10::ScalarType::Float),
        device_type,
        device_index,
        &tensor);
    EXPECT_EQ(error, Error::Ok);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->dtype(), slim_c10::ScalarType::Float);
    EXPECT_EQ(tensor->itemsize(), 4);
    delete tensor;
  }

  // Test BFloat16
  {
    Tensor* tensor = nullptr;
    AOTITorchError error = aoti_torch_empty_strided(
        sizes.size(),
        sizes.data(),
        strides.data(),
        static_cast<int32_t>(slim_c10::ScalarType::BFloat16),
        device_type,
        device_index,
        &tensor);
    EXPECT_EQ(error, Error::Ok);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->dtype(), slim_c10::ScalarType::BFloat16);
    EXPECT_EQ(tensor->itemsize(), 2);
    delete tensor;
  }

  // Test Int64
  {
    Tensor* tensor = nullptr;
    AOTITorchError error = aoti_torch_empty_strided(
        sizes.size(),
        sizes.data(),
        strides.data(),
        static_cast<int32_t>(slim_c10::ScalarType::Long),
        device_type,
        device_index,
        &tensor);
    EXPECT_EQ(error, Error::Ok);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->dtype(), slim_c10::ScalarType::Long);
    EXPECT_EQ(tensor->itemsize(), 8);
    delete tensor;
  }

  // Test Bool
  {
    Tensor* tensor = nullptr;
    AOTITorchError error = aoti_torch_empty_strided(
        sizes.size(),
        sizes.data(),
        strides.data(),
        static_cast<int32_t>(slim_c10::ScalarType::Bool),
        device_type,
        device_index,
        &tensor);
    EXPECT_EQ(error, Error::Ok);
    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->dtype(), slim_c10::ScalarType::Bool);
    EXPECT_EQ(tensor->itemsize(), 1);
    delete tensor;
  }
}

// ============================================================================
// CPU Tests
// ============================================================================

TEST_F(AOTITorchEmptyStridedSlimTest, BasicFunctionality_CPU) {
  runBasicEmptyStridedTest(static_cast<int32_t>(slim_c10::DeviceType::CPU), 0);
}

TEST_F(AOTITorchEmptyStridedSlimTest, MultiDimensional_CPU) {
  runMultiDimensionalEmptyStridedTest(
      static_cast<int32_t>(slim_c10::DeviceType::CPU), 0);
}

TEST_F(AOTITorchEmptyStridedSlimTest, ScalarTensor_CPU) {
  runScalarTensorEmptyStridedTest(
      static_cast<int32_t>(slim_c10::DeviceType::CPU), 0);
}

TEST_F(AOTITorchEmptyStridedSlimTest, ZeroSizedTensor_CPU) {
  runZeroSizedTensorEmptyStridedTest(
      static_cast<int32_t>(slim_c10::DeviceType::CPU), 0);
}

TEST_F(AOTITorchEmptyStridedSlimTest, CustomStrides_CPU) {
  runCustomStridesEmptyStridedTest(
      static_cast<int32_t>(slim_c10::DeviceType::CPU), 0);
}

TEST_F(AOTITorchEmptyStridedSlimTest, DifferentDtypes_CPU) {
  runDifferentDtypesEmptyStridedTest(
      static_cast<int32_t>(slim_c10::DeviceType::CPU), 0);
}

// ============================================================================
// CUDA Tests
// ============================================================================

TEST_F(AOTITorchEmptyStridedSlimTest, BasicFunctionality_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }
  runBasicEmptyStridedTest(static_cast<int32_t>(slim_c10::DeviceType::CUDA), 0);
}

TEST_F(AOTITorchEmptyStridedSlimTest, MultiDimensional_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }
  runMultiDimensionalEmptyStridedTest(
      static_cast<int32_t>(slim_c10::DeviceType::CUDA), 0);
}

TEST_F(AOTITorchEmptyStridedSlimTest, ScalarTensor_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }
  runScalarTensorEmptyStridedTest(
      static_cast<int32_t>(slim_c10::DeviceType::CUDA), 0);
}

TEST_F(AOTITorchEmptyStridedSlimTest, ZeroSizedTensor_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }
  runZeroSizedTensorEmptyStridedTest(
      static_cast<int32_t>(slim_c10::DeviceType::CUDA), 0);
}

TEST_F(AOTITorchEmptyStridedSlimTest, CustomStrides_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }
  runCustomStridesEmptyStridedTest(
      static_cast<int32_t>(slim_c10::DeviceType::CUDA), 0);
}

TEST_F(AOTITorchEmptyStridedSlimTest, DifferentDtypes_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }
  runDifferentDtypesEmptyStridedTest(
      static_cast<int32_t>(slim_c10::DeviceType::CUDA), 0);
}

// ============================================================================
// Verify Device Properties
// ============================================================================

TEST_F(AOTITorchEmptyStridedSlimTest, VerifyCPUDevice) {
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = calculateContiguousStrides(sizes);

  Tensor* tensor = nullptr;
  AOTITorchError error = aoti_torch_empty_strided(
      sizes.size(),
      sizes.data(),
      strides.data(),
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0,
      &tensor);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(tensor, nullptr);

  EXPECT_TRUE(tensor->is_cpu());
  EXPECT_FALSE(tensor->is_cuda());
  EXPECT_EQ(tensor->device_type(), slim_c10::DeviceType::CPU);

  delete tensor;
}

TEST_F(AOTITorchEmptyStridedSlimTest, VerifyCUDADevice) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }

  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = calculateContiguousStrides(sizes);

  Tensor* tensor = nullptr;
  AOTITorchError error = aoti_torch_empty_strided(
      sizes.size(),
      sizes.data(),
      strides.data(),
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CUDA),
      0,
      &tensor);

  EXPECT_EQ(error, Error::Ok);
  ASSERT_NE(tensor, nullptr);

  EXPECT_FALSE(tensor->is_cpu());
  EXPECT_TRUE(tensor->is_cuda());
  EXPECT_EQ(tensor->device_type(), slim_c10::DeviceType::CUDA);

  delete tensor;
}

// ============================================================================
// Error Cases
// ============================================================================

TEST_F(AOTITorchEmptyStridedSlimTest, NullReturnPointer) {
  std::vector<int64_t> sizes = {2, 3};
  std::vector<int64_t> strides = calculateContiguousStrides(sizes);

  AOTITorchError error = aoti_torch_empty_strided(
      sizes.size(),
      sizes.data(),
      strides.data(),
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0,
      nullptr); // null return pointer

  EXPECT_EQ(error, Error::InvalidArgument);
}
