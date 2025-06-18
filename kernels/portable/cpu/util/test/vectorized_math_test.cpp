/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/vectorized_math.h>

#include <c10/util/irange.h>

#include <gtest/gtest.h>

#include <cstdint>

#ifndef ET_USE_PYTORCH_HEADERS
#error "This test requires ET_USE_PYTORCH_HEADERS!"
#endif // ET_USE_PYTORCH_HEADERS

TEST(VectorizedMathTest, BasicUnary) {
  __at_align__ float result_floats[at::vec::Vectorized<float>::size()] = {0};
  const auto x_vec = at::vec::Vectorized<float>::arange(0, 1);
  const auto result_vec = executorch::math::exp(x_vec);
  result_vec.store(result_floats);
  for (const auto ii : c10::irange(at::vec::Vectorized<float>::size())) {
    EXPECT_FLOAT_EQ(result_floats[ii], std::exp(ii));
  }
}

namespace {
template <typename T>
void test_unary_t_to_float() {
  __at_align__ float result_floats[at::vec::Vectorized<T>::size()] = {0};
  const auto x_vec = at::vec::Vectorized<T>::arange(0, 1);
  const auto result_vec = executorch::math::exp(x_vec);
  static_assert(decltype(result_vec)::size() >= at::vec::Vectorized<T>::size());
  result_vec.store(result_floats, at::vec::Vectorized<T>::size());
  for (const auto ii : c10::irange(at::vec::Vectorized<T>::size())) {
    EXPECT_FLOAT_EQ(result_floats[ii], std::exp((float)ii)) << ii;
  }
}

} // namespace

TEST(VectorizedMathTest, UnaryInt16ToFloat) {
  test_unary_t_to_float<std::uint16_t>();
}

TEST(VectorizedMathTest, UnaryInt32ToFloat) {
  test_unary_t_to_float<std::uint32_t>();
}

TEST(VectorizedMathTest, UnaryInt64ToFloat) {
  test_unary_t_to_float<std::uint64_t>();
}

TEST(VectorizedMathTest, BasicBinary) {
  __at_align__ float result_floats[at::vec::Vectorized<float>::size()] = {0};
  const auto x_vec = at::vec::Vectorized<float>::arange(0, 1);
  const auto y_vec = at::vec::Vectorized<float>(2);
  const auto result_vec = executorch::math::pow(x_vec, y_vec);
  result_vec.store(result_floats);
  for (const auto ii : c10::irange(at::vec::Vectorized<float>::size())) {
    EXPECT_FLOAT_EQ(result_floats[ii], std::pow((float)ii, 2.0f));
  }
}

namespace {
template <typename T>
void test_binary_t_to_float() {
  __at_align__ float result_floats[at::vec::Vectorized<T>::size()] = {0};
  const auto x_vec = at::vec::Vectorized<T>::arange(0, 1);
  const auto y_vec = at::vec::Vectorized<T>(2);
  const auto result_vec = executorch::math::pow(x_vec, y_vec);
  static_assert(decltype(result_vec)::size() >= at::vec::Vectorized<T>::size());
  result_vec.store(result_floats, at::vec::Vectorized<T>::size());
  for (const auto ii : c10::irange(at::vec::Vectorized<T>::size())) {
    EXPECT_EQ(result_floats[ii], std::pow((float)ii, 2.0f)) << ii;
  }
}

TEST(VectorizedMathTest, BinaryInt16ToFloat) {
  test_binary_t_to_float<std::int16_t>();
}

TEST(VectorizedMathTest, BinaryInt32ToFloat) {
  test_binary_t_to_float<std::int32_t>();
}

TEST(VectorizedMathTest, BinaryInt64ToFloat) {
  test_binary_t_to_float<std::uint64_t>();
}

} // namespace
