/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Runs the sort and rand shims on whatever device is present. They are the
// callers of the allocator's stream-ordered scratch path (thrust's temporary
// storage, sort's transpose buffers, rand's generator state), which the other
// runtime tests never reach; on a device without memory pools that is the
// cudaMalloc fallback.

#include <gtest/gtest.h>

#include <executorch/extension/cuda/runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <vector>

#include <executorch/backends/aoti/slim/c10/core/DeviceType.h>
#include <executorch/backends/aoti/slim/c10/core/ScalarType.h>
#include <executorch/backends/cuda/runtime/shims/memory.h>
#include <executorch/backends/cuda/runtime/shims/rand.h>
#include <executorch/backends/cuda/runtime/shims/sort.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/platform/platform.h>

using executorch::backends::cuda::aoti_torch_cuda_rand;
using executorch::backends::cuda::aoti_torch_cuda_sort_stable;
using executorch::backends::cuda::aoti_torch_delete_tensor_object;
using executorch::backends::cuda::aoti_torch_empty_strided;
using executorch::runtime::Error;
using Tensor = executorch::backends::aoti::slim::SlimTensor;
namespace slim_c10 = executorch::backends::aoti::slim::c10;

class CudaSortRandTest : public testing::Test {
 protected:
  void SetUp() override {
    et_pal_init();
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      // The Windows job sets this to run these on its GPU without memory
      // pools; skipping there would leave the fallback untested.
      if (std::getenv("EXECUTORCH_CUDA_TEST_REQUIRE_NO_MEMORY_POOLS") !=
          nullptr) {
        FAIL() << "this job expects a CUDA device, but none is available";
      }
      GTEST_SKIP() << "CUDA not available";
    }
  }

  Tensor* float_tensor(const std::vector<int64_t>& sizes) {
    Tensor* tensor = nullptr;
    EXPECT_EQ(
        aoti_torch_empty_strided(
            static_cast<int64_t>(sizes.size()),
            sizes.data(),
            nullptr,
            static_cast<int32_t>(slim_c10::ScalarType::Float),
            static_cast<int32_t>(slim_c10::DeviceType::CUDA),
            0,
            &tensor),
        Error::Ok);
    return tensor;
  }

  template <typename T>
  std::vector<T> to_host(const Tensor* tensor) {
    std::vector<T> host(tensor->numel());
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    EXPECT_EQ(
        cudaMemcpy(
            host.data(),
            tensor->data_ptr(),
            host.size() * sizeof(T),
            cudaMemcpyDeviceToHost),
        cudaSuccess);
    return host;
  }
};

// Sorting along dim 0 of a 2-D tensor transposes through two scratch buffers;
// sorting along the last dim uses thrust's scratch. Both are checked against a
// host sort.
TEST_F(CudaSortRandTest, SortMatchesHost) {
  constexpr int64_t kRows = 64;
  constexpr int64_t kCols = 32;
  std::vector<float> input(kRows * kCols);
  for (size_t i = 0; i < input.size(); ++i) {
    // Modulo 7, so every slice repeats values and a sort that is not stable
    // gives different indices.
    input[i] = static_cast<float>((i * 7919) % 7) - 3.0f;
  }
  Tensor* self = float_tensor({kRows, kCols});
  ASSERT_NE(self, nullptr);
  ASSERT_EQ(
      cudaMemcpy(
          self->data_ptr(),
          input.data(),
          input.size() * sizeof(float),
          cudaMemcpyHostToDevice),
      cudaSuccess);

  for (int64_t dim : {0, 1}) {
    int32_t stable = 1;
    Tensor* values = nullptr;
    Tensor* indices = nullptr;
    ASSERT_EQ(
        aoti_torch_cuda_sort_stable(
            self, &stable, dim, /*descending=*/0, &values, &indices),
        Error::Ok)
        << "sort along dim " << dim << " failed";
    const std::vector<float> got = to_host<float>(values);
    const std::vector<int64_t> got_indices = to_host<int64_t>(indices);
    const int64_t outer = dim == 0 ? kCols : kRows;
    const int64_t length = dim == 0 ? kRows : kCols;
    for (int64_t o = 0; o < outer; ++o) {
      std::vector<float> slice(length), actual(length);
      std::vector<int64_t> actual_indices(length);
      for (int64_t i = 0; i < length; ++i) {
        const int64_t at = dim == 0 ? i * kCols + o : o * kCols + i;
        slice[i] = input[at];
        actual[i] = got[at];
        actual_indices[i] = got_indices[at];
      }
      // The input repeats values, so a stable sort is what makes the indices
      // unique: equal values keep their original order.
      std::vector<int64_t> expected_indices(length);
      std::iota(expected_indices.begin(), expected_indices.end(), 0);
      std::stable_sort(
          expected_indices.begin(),
          expected_indices.end(),
          [&](int64_t a, int64_t b) { return slice[a] < slice[b]; });
      std::vector<float> expected(length);
      for (int64_t i = 0; i < length; ++i) {
        expected[i] = slice[expected_indices[i]];
      }
      ASSERT_EQ(actual, expected) << "slice " << o << " along dim " << dim;
      ASSERT_EQ(actual_indices, expected_indices)
          << "indices of slice " << o << " along dim " << dim;
    }
    EXPECT_EQ(aoti_torch_delete_tensor_object(values), Error::Ok);
    EXPECT_EQ(aoti_torch_delete_tensor_object(indices), Error::Ok);
  }
  EXPECT_EQ(aoti_torch_delete_tensor_object(self), Error::Ok);
}

// rand allocates its generator state on first use.
TEST_F(CudaSortRandTest, RandProducesUnitInterval) {
  std::vector<int64_t> sizes = {16, 16};
  int32_t dtype = static_cast<int32_t>(slim_c10::ScalarType::Float);
  Tensor* out = nullptr;
  ASSERT_EQ(
      aoti_torch_cuda_rand(
          sizes.data(),
          static_cast<int64_t>(sizes.size()),
          &dtype,
          /*layout=*/nullptr,
          /*device=*/nullptr,
          /*device_index_=*/0,
          /*pin_memory=*/nullptr,
          &out),
      Error::Ok);
  ASSERT_NE(out, nullptr);
  for (float value : to_host<float>(out)) {
    ASSERT_GE(value, 0.0f);
    ASSERT_LT(value, 1.0f);
  }
  EXPECT_EQ(aoti_torch_delete_tensor_object(out), Error::Ok);
}
