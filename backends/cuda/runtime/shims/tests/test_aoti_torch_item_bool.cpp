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

} // namespace

class AOTITorchItemBoolSlimTest : public ::testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();
  }

  Tensor* createScalarBoolTensor(
      bool value,
      int32_t device_type = static_cast<int32_t>(slim_c10::DeviceType::CPU),
      int32_t device_index = 0) {
    Tensor* tensor = nullptr;

    std::vector<int64_t> sizes = {1};
    std::vector<int64_t> strides = {1};

    AOTITorchError error = aoti_torch_empty_strided(
        sizes.size(),
        sizes.data(),
        strides.data(),
        static_cast<int32_t>(slim_c10::ScalarType::Bool),
        device_type,
        device_index,
        &tensor);

    if (error != Error::Ok || tensor == nullptr) {
      return nullptr;
    }

    if (device_type == static_cast<int32_t>(slim_c10::DeviceType::CPU)) {
      bool* data = static_cast<bool*>(tensor->data_ptr());
      *data = value;
    } else {
      cudaMemcpy(
          tensor->data_ptr(), &value, sizeof(bool), cudaMemcpyHostToDevice);
    }

    return tensor;
  }

  Tensor* createTestTensor(
      const std::vector<int64_t>& sizes,
      int32_t dtype = static_cast<int32_t>(slim_c10::ScalarType::Float),
      int32_t device_type = static_cast<int32_t>(slim_c10::DeviceType::CPU),
      int32_t device_index = 0) {
    Tensor* tensor = nullptr;

    std::vector<int64_t> strides(sizes.size());
    if (!sizes.empty()) {
      strides[sizes.size() - 1] = 1;
      for (int64_t i = static_cast<int64_t>(sizes.size()) - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * sizes[i + 1];
      }
    }

    AOTITorchError error = aoti_torch_empty_strided(
        sizes.size(),
        sizes.data(),
        strides.data(),
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

TEST_F(AOTITorchItemBoolSlimTest, TrueValue_CPU) {
  Tensor* tensor = createScalarBoolTensor(
      true, static_cast<int32_t>(slim_c10::DeviceType::CPU), 0);
  ASSERT_NE(tensor, nullptr);

  bool result = false;
  AOTITorchError error = aoti_torch_item_bool(tensor, &result);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_EQ(result, true);

  EXPECT_EQ(aoti_torch_delete_tensor_object(tensor), Error::Ok);
}

TEST_F(AOTITorchItemBoolSlimTest, FalseValue_CPU) {
  Tensor* tensor = createScalarBoolTensor(
      false, static_cast<int32_t>(slim_c10::DeviceType::CPU), 0);
  ASSERT_NE(tensor, nullptr);

  bool result = true;
  AOTITorchError error = aoti_torch_item_bool(tensor, &result);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_EQ(result, false);

  EXPECT_EQ(aoti_torch_delete_tensor_object(tensor), Error::Ok);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(AOTITorchItemBoolSlimTest, NullTensor) {
  bool result = false;
  AOTITorchError error = aoti_torch_item_bool(nullptr, &result);

  EXPECT_EQ(error, Error::InvalidArgument);
}

TEST_F(AOTITorchItemBoolSlimTest, NullReturnValue) {
  Tensor* tensor = createScalarBoolTensor(
      true, static_cast<int32_t>(slim_c10::DeviceType::CPU), 0);
  ASSERT_NE(tensor, nullptr);

  AOTITorchError error = aoti_torch_item_bool(tensor, nullptr);

  EXPECT_EQ(error, Error::InvalidArgument);

  EXPECT_EQ(aoti_torch_delete_tensor_object(tensor), Error::Ok);
}

TEST_F(AOTITorchItemBoolSlimTest, MultiElementTensor) {
  std::vector<int64_t> sizes = {2, 3};
  Tensor* tensor = createTestTensor(
      sizes,
      static_cast<int32_t>(slim_c10::ScalarType::Bool),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(tensor, nullptr);
  EXPECT_GT(tensor->numel(), 1);

  bool result = false;
  AOTITorchError error = aoti_torch_item_bool(tensor, &result);

  EXPECT_EQ(error, Error::InvalidArgument);

  EXPECT_EQ(aoti_torch_delete_tensor_object(tensor), Error::Ok);
}

TEST_F(AOTITorchItemBoolSlimTest, WrongDtype_Float) {
  std::vector<int64_t> sizes = {1};
  Tensor* tensor = createTestTensor(
      sizes,
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(tensor, nullptr);

  bool result = false;
  AOTITorchError error = aoti_torch_item_bool(tensor, &result);

  EXPECT_EQ(error, Error::InvalidArgument);

  EXPECT_EQ(aoti_torch_delete_tensor_object(tensor), Error::Ok);
}

TEST_F(AOTITorchItemBoolSlimTest, WrongDtype_Long) {
  std::vector<int64_t> sizes = {1};
  Tensor* tensor = createTestTensor(
      sizes,
      static_cast<int32_t>(slim_c10::ScalarType::Long),
      static_cast<int32_t>(slim_c10::DeviceType::CPU),
      0);
  ASSERT_NE(tensor, nullptr);

  bool result = false;
  AOTITorchError error = aoti_torch_item_bool(tensor, &result);

  EXPECT_EQ(error, Error::InvalidArgument);

  EXPECT_EQ(aoti_torch_delete_tensor_object(tensor), Error::Ok);
}

// ============================================================================
// CUDA Tests
// ============================================================================

TEST_F(AOTITorchItemBoolSlimTest, TrueValue_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }

  Tensor* tensor = createScalarBoolTensor(
      true, static_cast<int32_t>(slim_c10::DeviceType::CUDA), 0);
  ASSERT_NE(tensor, nullptr);
  EXPECT_TRUE(tensor->is_cuda());

  bool result = false;
  AOTITorchError error = aoti_torch_item_bool(tensor, &result);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_EQ(result, true);

  EXPECT_EQ(aoti_torch_delete_tensor_object(tensor), Error::Ok);
}

TEST_F(AOTITorchItemBoolSlimTest, FalseValue_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }

  Tensor* tensor = createScalarBoolTensor(
      false, static_cast<int32_t>(slim_c10::DeviceType::CUDA), 0);
  ASSERT_NE(tensor, nullptr);
  EXPECT_TRUE(tensor->is_cuda());

  bool result = true;
  AOTITorchError error = aoti_torch_item_bool(tensor, &result);

  EXPECT_EQ(error, Error::Ok);
  EXPECT_EQ(result, false);

  EXPECT_EQ(aoti_torch_delete_tensor_object(tensor), Error::Ok);
}

TEST_F(AOTITorchItemBoolSlimTest, MultiElementTensor_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }

  std::vector<int64_t> sizes = {2, 3};
  Tensor* tensor = createTestTensor(
      sizes,
      static_cast<int32_t>(slim_c10::ScalarType::Bool),
      static_cast<int32_t>(slim_c10::DeviceType::CUDA),
      0);
  ASSERT_NE(tensor, nullptr);
  EXPECT_TRUE(tensor->is_cuda());

  bool result = false;
  AOTITorchError error = aoti_torch_item_bool(tensor, &result);

  EXPECT_EQ(error, Error::InvalidArgument);

  EXPECT_EQ(aoti_torch_delete_tensor_object(tensor), Error::Ok);
}

TEST_F(AOTITorchItemBoolSlimTest, WrongDtype_Float_CUDA) {
  if (!isCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
  }

  std::vector<int64_t> sizes = {1};
  Tensor* tensor = createTestTensor(
      sizes,
      static_cast<int32_t>(slim_c10::ScalarType::Float),
      static_cast<int32_t>(slim_c10::DeviceType::CUDA),
      0);
  ASSERT_NE(tensor, nullptr);

  bool result = false;
  AOTITorchError error = aoti_torch_item_bool(tensor, &result);

  EXPECT_EQ(error, Error::InvalidArgument);

  EXPECT_EQ(aoti_torch_delete_tensor_object(tensor), Error::Ok);
}
