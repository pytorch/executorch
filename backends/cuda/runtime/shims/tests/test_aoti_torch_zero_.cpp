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

class AOTITorchZeroTest : public ::testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "CUDA not available";
    }
  }

  Tensor* create(
      const std::vector<int64_t>& sizes,
      const std::vector<int64_t>& strides,
      slim_c10::DeviceType device_type) {
    Tensor* tensor = nullptr;
    EXPECT_EQ(
        aoti_torch_empty_strided(
            sizes.size(),
            sizes.data(),
            strides.data(),
            static_cast<int32_t>(slim_c10::ScalarType::Float),
            static_cast<int32_t>(device_type),
            0,
            &tensor),
        Error::Ok);
    return tensor;
  }
};

TEST_F(AOTITorchZeroTest, ZerosContiguousCudaTensor) {
  Tensor* tensor = create({2, 3}, {3, 1}, slim_c10::DeviceType::CUDA);
  const std::vector<float> ones(6, 1.0f);
  ASSERT_EQ(
      cudaMemcpy(
          tensor->data_ptr(),
          ones.data(),
          sizeof(float) * ones.size(),
          cudaMemcpyHostToDevice),
      cudaSuccess);

  EXPECT_EQ(aoti_torch_zero_(tensor), Error::Ok);

  std::vector<float> actual(6, 1.0f);
  ASSERT_EQ(
      cudaMemcpy(
          actual.data(),
          tensor->data_ptr(),
          sizeof(float) * actual.size(),
          cudaMemcpyDeviceToHost),
      cudaSuccess);
  EXPECT_EQ(actual, std::vector<float>(6, 0.0f));
  aoti_torch_delete_tensor_object(tensor);
}

TEST_F(AOTITorchZeroTest, RejectsNullAndNonContiguousTensors) {
  EXPECT_EQ(aoti_torch_zero_(nullptr), Error::InvalidArgument);
  Tensor* tensor = create({2, 3}, {1, 2}, slim_c10::DeviceType::CUDA);
  EXPECT_EQ(aoti_torch_zero_(tensor), Error::InvalidArgument);
  aoti_torch_delete_tensor_object(tensor);
}
