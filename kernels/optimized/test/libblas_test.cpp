/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/kernels/optimized/blas/CPUBlas.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

#include <vector>

#define TEST_FORALL_SUPPORTED_CTYPES(_, N) \
  _<double, N>();                          \
  _<float, N>();                           \
  _<int64_t, N>();                         \
  _<uint8_t, N>();                         \
  _<int32_t, N>();                         \
  _<exec_aten::BFloat16, N>();

namespace {

// Fill a vector with a monotonic sequence of integer values
template <typename T>
void fill_ones(std::vector<T>& arr) {
  for (size_t i = 0; i < arr.size(); ++i) {
    arr[i] = static_cast<T>(1);
  }
}

template <typename T>
bool check_all_equal_to(std::vector<T>& arr, const float value) {
  for (size_t i = 0; i < arr.size(); ++i) {
    if (arr[i] != static_cast<T>(value)) {
      return false;
    }
  }
  return true;
}

} // namespace

template <class CTYPE, int64_t N>
void test_matmul_ones() {
  using executorch::cpublas::TransposeType;

  std::vector<CTYPE> in_1(N * N);
  fill_ones(in_1);
  std::vector<CTYPE> in_2(N * N);
  fill_ones(in_2);

  std::vector<CTYPE> out(N * N);

  const CTYPE* in_1_data = in_1.data();
  const CTYPE* in_2_data = in_2.data();

  CTYPE* out_data = out.data();

  // clang-format off
  executorch::cpublas::gemm(
      TransposeType::NoTranspose, TransposeType::NoTranspose,
      N, N, N,
      static_cast<CTYPE>(1),
      in_1_data, N,
      in_2_data, N,
      static_cast<CTYPE>(0),
      out_data, N);
  // clang-format on

  EXPECT_TRUE(check_all_equal_to(out, static_cast<float>(N)));
}

TEST(BlasTest, MatmulOnes) {
  TEST_FORALL_SUPPORTED_CTYPES(test_matmul_ones, 25);
}
