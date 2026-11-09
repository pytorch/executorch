/*
 * Copyright 2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

#include <executorch/backends/arm/runtime/VGFShapeUtils.h>

namespace executorch {
namespace backends {
namespace vgf {
namespace {

// cppcheck-suppress syntaxError
TEST(VgfShapeUtilsTest, CalculatesValidElementCount) {
  size_t count = 0;

  ASSERT_TRUE(detail::checked_element_count_from_shape({2, 3, 4}, &count));
  EXPECT_EQ(count, 24u);
}

// cppcheck-suppress syntaxError
TEST(VgfShapeUtilsTest, ScalarHasOneElement) {
  size_t count = 0;

  ASSERT_TRUE(detail::checked_element_count_from_shape({}, &count));
  EXPECT_EQ(count, 1u);
}

// cppcheck-suppress syntaxError
TEST(VgfShapeUtilsTest, RejectsZeroDimension) {
  size_t count = 0;

  EXPECT_FALSE(detail::checked_element_count_from_shape({2, 0, 4}, &count));
}

// cppcheck-suppress syntaxError
TEST(VgfShapeUtilsTest, RejectsNegativeDimension) {
  size_t count = 0;

  EXPECT_FALSE(detail::checked_element_count_from_shape({2, -1, 4}, &count));
}

// cppcheck-suppress syntaxError
TEST(VgfShapeUtilsTest, RejectsElementCountOverflow) {
  size_t count = 0;

  EXPECT_FALSE(detail::checked_element_count_from_shape(
      {std::numeric_limits<int64_t>::max(), 2}, &count));
}

// cppcheck-suppress syntaxError
TEST(VgfShapeUtilsTest, RejectsLogicalByteSizeOverflow) {
  size_t byte_size = 0;

  EXPECT_FALSE(detail::checked_logical_byte_size_from_shape(
      {std::numeric_limits<int64_t>::max()}, 4, &byte_size));
}

// cppcheck-suppress syntaxError
TEST(VgfShapeUtilsTest, GeneratesContiguousStride) {
  std::vector<int64_t> stride;

  ASSERT_TRUE(detail::normalize_stride_checked({2, 3, 4}, {}, &stride));

  EXPECT_EQ(stride, (std::vector<int64_t>{12, 4, 1}));
}

// cppcheck-suppress syntaxError
TEST(VgfShapeUtilsTest, PreservesExplicitStride) {
  const std::vector<int64_t> explicit_stride{16, 4, 1};
  std::vector<int64_t> stride;

  ASSERT_TRUE(
      detail::normalize_stride_checked({2, 3, 4}, explicit_stride, &stride));

  EXPECT_EQ(stride, explicit_stride);
}

// cppcheck-suppress syntaxError
TEST(VgfShapeUtilsTest, RejectsShortExplicitStride) {
  std::vector<int64_t> normalized;

  EXPECT_FALSE(
      detail::normalize_stride_checked({2, 3, 4}, {12, 4}, &normalized));
}

// cppcheck-suppress syntaxError
TEST(VgfShapeUtilsTest, RejectsLongExplicitStride) {
  std::vector<int64_t> normalized;

  EXPECT_FALSE(
      detail::normalize_stride_checked({2, 3, 4}, {48, 12, 4, 1}, &normalized));
}

// cppcheck-suppress syntaxError
TEST(VgfShapeUtilsTest, RejectsExplicitStrideOnScalar) {
  std::vector<int64_t> normalized;

  EXPECT_FALSE(detail::normalize_stride_checked({}, {1}, &normalized));
}

// cppcheck-suppress syntaxError
TEST(VgfShapeUtilsTest, RejectsStrideGenerationOverflow) {
  std::vector<int64_t> stride;

  EXPECT_FALSE(detail::normalize_stride_checked(
      {2, std::numeric_limits<int64_t>::max()}, {}, &stride));
}

// cppcheck-suppress syntaxError
TEST(VgfShapeUtilsTest, ExplicitStrideCannotBypassInvalidShape) {
  std::vector<int64_t> stride;

  EXPECT_FALSE(detail::normalize_stride_checked({2, 0, 4}, {4, 4, 1}, &stride));
}

} // namespace
} // namespace vgf
} // namespace backends
} // namespace executorch
