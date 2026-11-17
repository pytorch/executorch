/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_runtime.h>
#include <executorch/backends/aoti/slim/c10/core/DeviceType.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/cuda/runtime/shims/index.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/platform.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using executorch::backends::cuda::aoti_torch_cuda_masked_select;
using executorch::backends::cuda::aoti_torch_delete_tensor_object;
using executorch::backends::cuda::aoti_torch_empty_strided;
using executorch::runtime::Error;
using Tensor = executorch::backends::aoti::slim::SlimTensor;
namespace slim_c10 = executorch::backends::aoti::slim::c10;

class AOTITorchCudaMaskedSelectTest : public ::testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "CUDA not available, skipping CUDA tests";
    }
  }

  void TearDown() override {
    for (Tensor* tensor : tensors_) {
      aoti_torch_delete_tensor_object(tensor);
    }
  }

  template <typename T>
  Tensor* to_device(
      const std::vector<T>& values,
      const std::vector<int64_t>& sizes,
      slim_c10::ScalarType dtype,
      const std::vector<int64_t>& strides = {}) {
    Tensor* tensor = nullptr;
    EXPECT_EQ(
        aoti_torch_empty_strided(
            static_cast<int64_t>(sizes.size()),
            sizes.data(),
            strides.empty() ? nullptr : strides.data(),
            static_cast<int32_t>(dtype),
            static_cast<int32_t>(slim_c10::DeviceType::CUDA),
            0,
            &tensor),
        Error::Ok);
    EXPECT_EQ(
        cudaMemcpy(
            tensor->data_ptr(),
            values.data(),
            values.size() * sizeof(T),
            cudaMemcpyHostToDevice),
        cudaSuccess);
    tensors_.push_back(tensor);
    return tensor;
  }

  template <typename T>
  std::vector<T> masked_select(Tensor* self, Tensor* mask) {
    Tensor* result = nullptr;
    EXPECT_EQ(aoti_torch_cuda_masked_select(self, mask, &result), Error::Ok);
    if (result == nullptr) {
      return {};
    }
    tensors_.push_back(result);
    EXPECT_EQ(result->dim(), 1);
    std::vector<T> host(result->numel());
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(
        cudaMemcpy(
            host.data(),
            result->data_ptr(),
            host.size() * sizeof(T),
            cudaMemcpyDeviceToHost),
        cudaSuccess);
    return host;
  }

  std::vector<Tensor*> tensors_;
};

TEST_F(AOTITorchCudaMaskedSelectTest, SameShape) {
  Tensor* self = to_device<float>(
      {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f}, {2, 3}, slim_c10::kFloat);
  Tensor* mask =
      to_device<uint8_t>({1, 0, 1, 0, 1, 0}, {2, 3}, slim_c10::kBool);

  const std::vector<float> expected = {1.0f, 3.0f, 5.0f};
  EXPECT_EQ(masked_select<float>(self, mask), expected);
}

TEST_F(AOTITorchCudaMaskedSelectTest, MaskBroadcastsOverRows) {
  Tensor* self =
      to_device<int64_t>({10, 11, 12, 20, 21, 22}, {2, 3}, slim_c10::kLong);
  Tensor* mask = to_device<uint8_t>({0, 1, 1}, {3}, slim_c10::kBool);

  const std::vector<int64_t> expected = {11, 12, 21, 22};
  EXPECT_EQ(masked_select<int64_t>(self, mask), expected);
}

TEST_F(AOTITorchCudaMaskedSelectTest, SelfBroadcastsToMask) {
  // self is a column, so each of its values repeats along the mask's rows.
  Tensor* self = to_device<int32_t>({7, 8}, {2, 1}, slim_c10::kInt);
  Tensor* mask =
      to_device<uint8_t>({1, 0, 1, 1, 1, 0}, {2, 3}, slim_c10::kByte);

  const std::vector<int32_t> expected = {7, 7, 8, 8};
  EXPECT_EQ(masked_select<int32_t>(self, mask), expected);
}

TEST_F(AOTITorchCudaMaskedSelectTest, NonContiguousSelf) {
  // Stored column-major, so the logical tensor is {{1, 2}, {3, 4}}.
  Tensor* self = to_device<float>(
      {1.0f, 3.0f, 2.0f, 4.0f}, {2, 2}, slim_c10::kFloat, {1, 2});
  Tensor* mask = to_device<uint8_t>({0, 1, 1, 0}, {2, 2}, slim_c10::kBool);

  const std::vector<float> expected = {2.0f, 3.0f};
  EXPECT_EQ(masked_select<float>(self, mask), expected);
}

TEST_F(AOTITorchCudaMaskedSelectTest, ZeroDimSelfAndMask) {
  Tensor* self = to_device<float>({4.0f}, {}, slim_c10::kFloat);
  Tensor* set = to_device<uint8_t>({1}, {}, slim_c10::kBool);
  Tensor* unset = to_device<uint8_t>({0}, {}, slim_c10::kBool);

  EXPECT_EQ(masked_select<float>(self, set), std::vector<float>{4.0f});
  EXPECT_EQ(masked_select<float>(self, unset), std::vector<float>{});
}

TEST_F(AOTITorchCudaMaskedSelectTest, IncompatibleShapesAreRejected) {
  Tensor* self = to_device<float>({1.0f, 2.0f, 3.0f}, {3}, slim_c10::kFloat);
  Tensor* mask = to_device<uint8_t>({1, 0}, {2}, slim_c10::kBool);
  Tensor* result = nullptr;

  EXPECT_EQ(
      aoti_torch_cuda_masked_select(self, mask, &result),
      Error::InvalidArgument);
}
