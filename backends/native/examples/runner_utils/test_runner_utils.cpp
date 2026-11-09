// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/examples/runner_utils/runner_utils.h>

#include <limits>
#include <vector>

#include <gtest/gtest.h>

namespace ptn::runner_utils {
namespace {

// cppcheck-suppress-begin syntaxError
TEST(RunnerUtilsTest, CompareFloatsUsesAbsoluteAndRelativeTolerance) {
  EXPECT_TRUE(compare_floats({1.0f, 100.0f}, {1.0005f, 100.05f}, 1e-3, 1e-3));
  EXPECT_FALSE(compare_floats({1.0f, 100.0f}, {1.1f, 100.0f}, 1e-3, 1e-3));
  EXPECT_FALSE(compare_floats({1.0f}, {1.0f, 2.0f}, 1e-3, 1e-3));
}

TEST(RunnerUtilsTest, CompareFloatsRejectsNonFiniteValues) {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float infinity = std::numeric_limits<float>::infinity();

  EXPECT_FALSE(compare_floats({nan}, {1.0f}, 1e-3, 1e-3));
  EXPECT_FALSE(compare_floats({1.0f}, {nan}, 1e-3, 1e-3));
  EXPECT_FALSE(compare_floats({infinity}, {infinity}, 1e-3, 1e-3));
}

TEST(RunnerUtilsTest, CompareFloatsRejectsInvalidTolerances) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();

  EXPECT_FALSE(compare_floats({1.0f}, {1.0f}, nan, 1e-3));
  EXPECT_FALSE(compare_floats({1.0f}, {1.0f}, 1e-3, infinity));
  EXPECT_FALSE(compare_floats({1.0f}, {1.0f}, -1e-3, 1e-3));
}

TEST(RunnerUtilsTest, TopKReturnsEmptyForEmptyInput) {
  EXPECT_TRUE(top_k({}, 5).empty());
  EXPECT_TRUE(top_k({1.0f, 2.0f}, -1).empty());
}

TEST(RunnerUtilsTest, TopKReturnsIndicesByDescendingValue) {
  EXPECT_EQ(top_k({0.1f, 0.9f, 0.5f, 0.7f}, 3), (std::vector<int>{1, 3, 2}));
  EXPECT_EQ(top_k({2.0f, 3.0f}, 5), (std::vector<int>{1, 0}));
}
// cppcheck-suppress-end syntaxError

} // namespace
} // namespace ptn::runner_utils
