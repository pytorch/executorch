/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/kernels/optimized/cpu/moments_utils.h>

#include <vector>

#define TEST_FORALL_FLOAT_CTYPES(_) \
  _<double>();                      \
  _<float>();                       \
  _<short>();

namespace {

// Check if a float value is close to a reference value
template <class T>
bool is_close(T val, float ref, float tol = 1e-5) {
  T diff = std::abs(val - static_cast<T>(ref));
  return diff <= static_cast<T>(tol);
}

} // namespace

template <class CTYPE>
void test_calc_moments() {
  using torch::executor::native::RowwiseMoments;

  std::vector<CTYPE> in({2, 3, 4, 5, 9, 10, 12, 13});

  float mean;
  float variance;
  const CTYPE* in_data = in.data();
  std::tie(mean, variance) = RowwiseMoments(in_data, 8);

  EXPECT_TRUE(is_close<CTYPE>(mean, 7.25f));
  EXPECT_TRUE(is_close<CTYPE>(variance, 15.9375f));
}

TEST(MomentsUtilTest, CalculateMoments) {
  TEST_FORALL_FLOAT_CTYPES(test_calc_moments)
}
