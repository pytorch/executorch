/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <gtest/gtest.h>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_shape_to_c_string.h>
#include <executorch/runtime/platform/runtime.h>
#include <array>

using executorch::runtime::kTensorDimensionLimit;
using executorch::runtime::Span;
using executorch::runtime::tensor_shape_to_c_string;
using executorch::runtime::internal::kMaximumPrintableTensorShapeElement;

TEST(TensorShapeToCStringTest, Basic) {
  std::array<executorch::aten::SizesType, 3> sizes = {123, 456, 789};
  auto str = tensor_shape_to_c_string(
      Span<const executorch::aten::SizesType>(sizes.data(), sizes.size()));
  EXPECT_STREQ(str.data(), "(123, 456, 789)");

  std::array<executorch::aten::SizesType, 1> one_size = {1234567890};
  str = tensor_shape_to_c_string(Span<const executorch::aten::SizesType>(
      one_size.data(), one_size.size()));
  EXPECT_STREQ(str.data(), "(1234567890)");
}

TEST(TensorShapeToCStringTest, NegativeItems) {
  std::array<executorch::aten::SizesType, 4> sizes = {-1, -3, -2, 4};
  auto str = tensor_shape_to_c_string(
      Span<const executorch::aten::SizesType>(sizes.data(), sizes.size()));
  EXPECT_STREQ(str.data(), "(ERR, ERR, ERR, 4)");

  std::array<executorch::aten::SizesType, 1> one_size = {-1234567890};
  str = tensor_shape_to_c_string(Span<const executorch::aten::SizesType>(
      one_size.data(), one_size.size()));
  if constexpr (std::numeric_limits<executorch::aten::SizesType>::is_signed) {
    EXPECT_STREQ(str.data(), "(ERR)");
  } else {
    EXPECT_EQ(str.data(), "(" + std::to_string(one_size[0]) + ")");
  }
}
TEST(TensorShapeToCStringTest, MaximumElement) {
  std::array<executorch::aten::SizesType, 3> sizes = {
      123, std::numeric_limits<executorch::aten::SizesType>::max(), 789};
  auto str = tensor_shape_to_c_string(
      Span<const executorch::aten::SizesType>(sizes.data(), sizes.size()));
  std::ostringstream expected;
  expected << '(';
  for (const auto elem : sizes) {
    expected << elem << ", ";
  }
  auto expected_str = expected.str();
  expected_str.pop_back();
  expected_str.back() = ')';
  EXPECT_EQ(str.data(), expected_str);
}

TEST(TensorShapeToCStringTest, MaximumLength) {
  std::array<executorch::aten::SizesType, kTensorDimensionLimit> sizes;
  std::fill(sizes.begin(), sizes.end(), kMaximumPrintableTensorShapeElement);

  auto str = tensor_shape_to_c_string(
      Span<const executorch::aten::SizesType>(sizes.data(), sizes.size()));

  std::ostringstream expected;
  expected << '(' << kMaximumPrintableTensorShapeElement;
  for (int ii = 0; ii < kTensorDimensionLimit - 1; ++ii) {
    expected << ", " << kMaximumPrintableTensorShapeElement;
  }
  expected << ')';
  auto expected_str = expected.str();

  EXPECT_EQ(expected_str, str.data());
}

TEST(TensorShapeToCStringTest, ExceedsDimensionLimit) {
  std::array<executorch::aten::SizesType, kTensorDimensionLimit + 1> sizes;
  std::fill(sizes.begin(), sizes.end(), kMaximumPrintableTensorShapeElement);

  auto str = tensor_shape_to_c_string(
      Span<const executorch::aten::SizesType>(sizes.data(), sizes.size()));

  EXPECT_STREQ(str.data(), "(ERR: tensor ndim exceeds limit)");
}
