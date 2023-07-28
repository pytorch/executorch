/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/kernels/optimized/vec/functional.h>
#include <executorch/kernels/optimized/vec/vec.h>

#include <vector>

#define TEST_FORALL_SUPPORTED_CTYPES(_) \
  _<int32_t>();                         \
  _<int64_t>();                         \
  _<float>();                           \
  _<double>();

namespace {

// Fill a vector with a monotonic sequence of integer values
template <typename T>
void fill_monotonic(
    std::vector<T>& arr,
    const int start = 0,
    const int step = 1) {
  int value = start;
  for (size_t i = 0; i < arr.size(); ++i) {
    arr[i] = static_cast<T>(value);
    value += step;
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

template <typename T>
void test_load_and_add() {
  using Vec = executorch::vec::Vectorized<T>;

  constexpr size_t kVecSize = static_cast<size_t>(Vec::size());

  std::vector<T> in_1(kVecSize);
  fill_monotonic(in_1);

  std::vector<T> in_2(kVecSize);
  fill_monotonic(in_2, kVecSize, -1);

  const Vec in_1_vec = Vec::loadu(in_1.data());
  const Vec in_2_vec = Vec::loadu(in_2.data());

  const Vec out_vec = in_1_vec + in_2_vec;

  std::vector<T> out(kVecSize);
  out_vec.store(out.data());

  EXPECT_TRUE(check_all_equal_to(out, static_cast<T>(kVecSize)));
}

TEST(VecFloatTest, LoadAndAdd) {
  TEST_FORALL_SUPPORTED_CTYPES(test_load_and_add);
}
