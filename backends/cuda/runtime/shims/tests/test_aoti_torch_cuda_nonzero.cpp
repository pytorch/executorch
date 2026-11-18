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
#include <limits>
#include <utility>
#include <vector>

using executorch::backends::cuda::aoti_torch_cuda_nonzero;
using executorch::backends::cuda::aoti_torch_delete_tensor_object;
using executorch::backends::cuda::aoti_torch_empty_strided;
using executorch::runtime::Error;
using Tensor = executorch::backends::aoti::slim::SlimTensor;
namespace slim_c10 = executorch::backends::aoti::slim::c10;

class AOTITorchCudaNonzeroTest : public ::testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "CUDA not available, skipping CUDA tests";
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
    return tensor;
  }

  // The (count, ndim) result read back row by row, so the expected
  // coordinates can be written one element per row.
  std::vector<std::vector<int64_t>> coordinates(const Tensor* result) {
    EXPECT_EQ(result->dim(), 2);
    const int64_t count = result->size(0);
    const int64_t ndim = result->size(1);
    std::vector<int64_t> host(count * ndim);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(
        cudaMemcpy(
            host.data(),
            result->data_ptr(),
            host.size() * sizeof(int64_t),
            cudaMemcpyDeviceToHost),
        cudaSuccess);
    std::vector<std::vector<int64_t>> rows(count, std::vector<int64_t>(ndim));
    for (int64_t i = 0; i < count; ++i) {
      for (int64_t d = 0; d < ndim; ++d) {
        rows[i][d] = host[i * result->stride(0) + d * result->stride(1)];
      }
    }
    return rows;
  }

  Tensor* nonzero(Tensor* self) {
    Tensor* result = nullptr;
    EXPECT_EQ(aoti_torch_cuda_nonzero(self, &result), Error::Ok);
    return result;
  }
};

TEST_F(AOTITorchCudaNonzeroTest, FloatMatchesATenLayout) {
  // -0.0 is zero and NaN is not, as in ATen.
  const float nan = std::numeric_limits<float>::quiet_NaN();
  Tensor* self = to_device<float>(
      {1.0f, 0.0f, -2.0f, -0.0f, 0.0f, nan}, {2, 3}, slim_c10::kFloat);
  Tensor* result = nonzero(self);
  ASSERT_NE(result, nullptr);

  EXPECT_EQ(result->dtype(), slim_c10::kLong);
  EXPECT_EQ(result->stride(0), 1);
  EXPECT_EQ(result->stride(1), 3);
  const std::vector<std::vector<int64_t>> expected = {{0, 0}, {0, 2}, {1, 2}};
  EXPECT_EQ(coordinates(result), expected);

  aoti_torch_delete_tensor_object(result);
  aoti_torch_delete_tensor_object(self);
}

TEST_F(AOTITorchCudaNonzeroTest, NonContiguousInput) {
  // Stored column-major, so the logical tensor is {{0, 5}, {7, 0}, {0, 9}}.
  Tensor* self =
      to_device<int64_t>({0, 7, 0, 5, 0, 9}, {3, 2}, slim_c10::kLong, {1, 3});
  Tensor* result = nonzero(self);
  ASSERT_NE(result, nullptr);

  const std::vector<std::vector<int64_t>> expected = {{0, 1}, {1, 0}, {2, 1}};
  EXPECT_EQ(coordinates(result), expected);

  aoti_torch_delete_tensor_object(result);
  aoti_torch_delete_tensor_object(self);
}

TEST_F(AOTITorchCudaNonzeroTest, HalfIgnoresSignOfZero) {
  // 0x8000 is -0.0 and 0x3c00 is 1.0 in half; bfloat16 shares the sign bit.
  for (auto dtype : {slim_c10::kHalf, slim_c10::kBFloat16}) {
    Tensor* self =
        to_device<uint16_t>({0x8000, 0x3c00, 0x0000, 0x0001}, {4}, dtype);
    Tensor* result = nonzero(self);
    ASSERT_NE(result, nullptr);

    const std::vector<std::vector<int64_t>> expected = {{1}, {3}};
    EXPECT_EQ(coordinates(result), expected);

    aoti_torch_delete_tensor_object(result);
    aoti_torch_delete_tensor_object(self);
  }
}

TEST_F(AOTITorchCudaNonzeroTest, AllZeroGivesEmptyResult) {
  Tensor* self = to_device<uint8_t>({0, 0, 0}, {3}, slim_c10::kBool);
  Tensor* result = nonzero(self);
  ASSERT_NE(result, nullptr);

  EXPECT_EQ(result->size(0), 0);
  EXPECT_EQ(result->size(1), 1);

  aoti_torch_delete_tensor_object(result);
  aoti_torch_delete_tensor_object(self);
}

TEST_F(AOTITorchCudaNonzeroTest, ZeroDimInput) {
  // ATen gives (1, 0) for a nonzero scalar and (0, 0) for a zero one.
  for (const auto& [value, count] :
       std::vector<std::pair<int32_t, int64_t>>{{3, 1}, {0, 0}}) {
    Tensor* self = to_device<int32_t>({value}, {}, slim_c10::kInt);
    Tensor* result = nonzero(self);
    ASSERT_NE(result, nullptr);

    EXPECT_EQ(result->size(0), count);
    EXPECT_EQ(result->size(1), 0);

    aoti_torch_delete_tensor_object(result);
    aoti_torch_delete_tensor_object(self);
  }
}

TEST_F(AOTITorchCudaNonzeroTest, NullArgumentIsRejected) {
  Tensor* result = nullptr;
  EXPECT_EQ(aoti_torch_cuda_nonzero(nullptr, &result), Error::InvalidArgument);
}
