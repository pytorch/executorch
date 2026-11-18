// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/native/runtime/graph/TensorMeta.h>

#include <gtest/gtest.h>

namespace ptn {
namespace {

// cppcheck-suppress-begin syntaxError
TEST(TensorMetaTest, AcceptsSizesWithinDynamicBounds) {
  const TensorMeta meta{
      .dtype = ScalarType::Int,
      .sizes = {1, 8},
      .dim_order_hint = {},
      .lower_bounds = {1, 1},
  };

  EXPECT_TRUE(meta.accepts_sizes({1, 1}));
  EXPECT_TRUE(meta.accepts_sizes({1, 5}));
  EXPECT_TRUE(meta.accepts_sizes({1, 8}));
  EXPECT_FALSE(meta.accepts_sizes({1, 0}));
  EXPECT_FALSE(meta.accepts_sizes({1, 9}));
  EXPECT_FALSE(meta.accepts_sizes({8}));
}

TEST(TensorMetaTest, StaticMetadataRequiresExactSizes) {
  const TensorMeta meta{
      .dtype = ScalarType::Float,
      .sizes = {2, 4},
      .dim_order_hint = {},
      .lower_bounds = {},
  };

  EXPECT_TRUE(meta.accepts_sizes({2, 4}));
  EXPECT_FALSE(meta.accepts_sizes({1, 4}));
}

TEST(TensorMetaTest, RejectsSizesBelowLowerBound) {
  const TensorMeta meta{
      .dtype = ScalarType::Int,
      .sizes = {8, 4},
      .dim_order_hint = {},
      .lower_bounds = {2, 4},
  };

  EXPECT_TRUE(meta.accepts_sizes({2, 4}));
  EXPECT_FALSE(meta.accepts_sizes({1, 4}));
}
// cppcheck-suppress-end syntaxError

} // namespace
} // namespace ptn
