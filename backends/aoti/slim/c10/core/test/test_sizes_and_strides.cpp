/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <executorch/backends/aoti/slim/c10/core/SizesAndStrides.h>

namespace executorch::backends::aoti::slim::c10 {

// =============================================================================
// Default Construction Tests
// =============================================================================

TEST(SizesAndStridesTest, DefaultConstruct) {
  SizesAndStrides ss;

  EXPECT_EQ(ss.size(), 1u);
  EXPECT_EQ(ss.size_at(0), 0);
  EXPECT_EQ(ss.stride_at(0), 1);
}

// =============================================================================
// Set Sizes and Strides Tests
// =============================================================================

TEST(SizesAndStridesTest, SetSizes) {
  SizesAndStrides ss;
  ss.set_sizes({2, 3, 4});

  EXPECT_EQ(ss.size(), 3u);
  EXPECT_EQ(ss.sizes_arrayref()[0], 2);
  EXPECT_EQ(ss.sizes_arrayref()[1], 3);
  EXPECT_EQ(ss.sizes_arrayref()[2], 4);
}

TEST(SizesAndStridesTest, SetSizesAndStrides) {
  SizesAndStrides ss;
  ss.set_sizes({2, 3, 4});
  ss.set_strides({12, 4, 1});

  EXPECT_EQ(ss.size(), 3u);
  EXPECT_EQ(ss.sizes_arrayref()[0], 2);
  EXPECT_EQ(ss.sizes_arrayref()[1], 3);
  EXPECT_EQ(ss.sizes_arrayref()[2], 4);
  EXPECT_EQ(ss.strides_arrayref()[0], 12);
  EXPECT_EQ(ss.strides_arrayref()[1], 4);
  EXPECT_EQ(ss.strides_arrayref()[2], 1);
}

TEST(SizesAndStridesTest, SizeAt) {
  SizesAndStrides ss;
  ss.set_sizes({10, 20, 30});

  EXPECT_EQ(ss.size_at(0), 10);
  EXPECT_EQ(ss.size_at(1), 20);
  EXPECT_EQ(ss.size_at(2), 30);
}

TEST(SizesAndStridesTest, StrideAt) {
  SizesAndStrides ss;
  ss.set_sizes({2, 3});
  ss.set_strides({3, 1});

  EXPECT_EQ(ss.stride_at(0), 3);
  EXPECT_EQ(ss.stride_at(1), 1);
}

TEST(SizesAndStridesTest, SizeAtUnchecked) {
  SizesAndStrides ss;
  ss.set_sizes({5, 6, 7});

  EXPECT_EQ(ss.size_at_unchecked(0), 5);
  EXPECT_EQ(ss.size_at_unchecked(1), 6);
  EXPECT_EQ(ss.size_at_unchecked(2), 7);
}

TEST(SizesAndStridesTest, ModifySizeAt) {
  SizesAndStrides ss;
  ss.set_sizes({2, 3, 4});

  ss.size_at(1) = 10;

  EXPECT_EQ(ss.size_at(1), 10);
}

TEST(SizesAndStridesTest, ModifyStrideAt) {
  SizesAndStrides ss;
  ss.set_sizes({2, 3});
  ss.set_strides({3, 1});

  ss.stride_at(0) = 100;

  EXPECT_EQ(ss.stride_at(0), 100);
}

// =============================================================================
// Inline vs Out-of-Line Storage Tests
// =============================================================================

TEST(SizesAndStridesTest, InlineStorage) {
  SizesAndStrides ss;
  ss.set_sizes({1, 2, 3, 4, 5});
  ss.set_strides({120, 60, 20, 5, 1});

  EXPECT_EQ(ss.size(), 5u);
  EXPECT_EQ(ss.sizes_arrayref()[4], 5);
  EXPECT_EQ(ss.strides_arrayref()[4], 1);
}

TEST(SizesAndStridesTest, OutOfLineStorage) {
  SizesAndStrides ss;
  ss.set_sizes({1, 2, 3, 4, 5, 6, 7});
  ss.set_strides({5040, 2520, 840, 210, 42, 7, 1});

  EXPECT_EQ(ss.size(), 7u);
  EXPECT_EQ(ss.sizes_arrayref()[6], 7);
  EXPECT_EQ(ss.strides_arrayref()[6], 1);
}

TEST(SizesAndStridesTest, ResizeFromInlineToOutOfLine) {
  SizesAndStrides ss;
  ss.set_sizes({1, 2, 3});

  EXPECT_EQ(ss.size(), 3u);

  ss.set_sizes({1, 2, 3, 4, 5, 6, 7, 8});
  ss.set_strides({40320, 20160, 6720, 1680, 336, 56, 8, 1});

  EXPECT_EQ(ss.size(), 8u);
  EXPECT_EQ(ss.sizes_arrayref()[7], 8);
}

TEST(SizesAndStridesTest, ResizeFromOutOfLineToInline) {
  SizesAndStrides ss;
  ss.set_sizes({1, 2, 3, 4, 5, 6, 7});
  ss.set_strides({5040, 2520, 840, 210, 42, 7, 1});

  EXPECT_EQ(ss.size(), 7u);

  ss.set_sizes({2, 3});
  ss.set_strides({3, 1});

  EXPECT_EQ(ss.size(), 2u);
  EXPECT_EQ(ss.sizes_arrayref()[0], 2);
  EXPECT_EQ(ss.strides_arrayref()[0], 3);
}

// =============================================================================
// Copy and Move Tests
// =============================================================================

TEST(SizesAndStridesTest, CopyConstructInline) {
  SizesAndStrides original;
  original.set_sizes({2, 3, 4});
  original.set_strides({12, 4, 1});

  SizesAndStrides copy = original;

  EXPECT_EQ(copy.size(), 3u);
  EXPECT_EQ(copy.sizes_arrayref()[0], 2);
  EXPECT_EQ(copy.strides_arrayref()[0], 12);
}

TEST(SizesAndStridesTest, CopyConstructOutOfLine) {
  SizesAndStrides original;
  original.set_sizes({1, 2, 3, 4, 5, 6, 7});
  original.set_strides({5040, 2520, 840, 210, 42, 7, 1});

  SizesAndStrides copy = original;

  EXPECT_EQ(copy.size(), 7u);
  EXPECT_EQ(copy.sizes_arrayref()[6], 7);
  EXPECT_EQ(copy.strides_arrayref()[6], 1);
}

TEST(SizesAndStridesTest, CopyAssignInline) {
  SizesAndStrides original;
  original.set_sizes({2, 3});
  original.set_strides({3, 1});

  SizesAndStrides copy;
  copy = original;

  EXPECT_EQ(copy.size(), 2u);
  EXPECT_EQ(copy.sizes_arrayref()[1], 3);
}

TEST(SizesAndStridesTest, MoveConstructInline) {
  SizesAndStrides original;
  original.set_sizes({2, 3, 4});
  original.set_strides({12, 4, 1});

  SizesAndStrides moved = std::move(original);

  EXPECT_EQ(moved.size(), 3u);
  EXPECT_EQ(moved.sizes_arrayref()[0], 2);
  EXPECT_EQ(original.size(), 0u);
}

TEST(SizesAndStridesTest, MoveConstructOutOfLine) {
  SizesAndStrides original;
  original.set_sizes({1, 2, 3, 4, 5, 6, 7});
  original.set_strides({5040, 2520, 840, 210, 42, 7, 1});

  SizesAndStrides moved = std::move(original);

  EXPECT_EQ(moved.size(), 7u);
  EXPECT_EQ(moved.sizes_arrayref()[6], 7);
  EXPECT_EQ(original.size(), 0u);
}

TEST(SizesAndStridesTest, MoveAssignOutOfLine) {
  SizesAndStrides original;
  original.set_sizes({1, 2, 3, 4, 5, 6, 7});
  original.set_strides({5040, 2520, 840, 210, 42, 7, 1});

  SizesAndStrides target;
  target = std::move(original);

  EXPECT_EQ(target.size(), 7u);
  EXPECT_EQ(target.sizes_arrayref()[6], 7);
  EXPECT_EQ(original.size(), 0u);
}

// =============================================================================
// Equality Tests
// =============================================================================

TEST(SizesAndStridesTest, EqualityTrue) {
  SizesAndStrides ss1;
  ss1.set_sizes({2, 3, 4});
  ss1.set_strides({12, 4, 1});

  SizesAndStrides ss2;
  ss2.set_sizes({2, 3, 4});
  ss2.set_strides({12, 4, 1});

  EXPECT_TRUE(ss1 == ss2);
}

TEST(SizesAndStridesTest, EqualityFalseDifferentSizes) {
  SizesAndStrides ss1;
  ss1.set_sizes({2, 3, 4});
  ss1.set_strides({12, 4, 1});

  SizesAndStrides ss2;
  ss2.set_sizes({2, 3, 5});
  ss2.set_strides({15, 5, 1});

  EXPECT_FALSE(ss1 == ss2);
}

TEST(SizesAndStridesTest, EqualityFalseDifferentDims) {
  SizesAndStrides ss1;
  ss1.set_sizes({2, 3});

  SizesAndStrides ss2;
  ss2.set_sizes({2, 3, 4});

  EXPECT_FALSE(ss1 == ss2);
}

// =============================================================================
// Iterator Tests
// =============================================================================

TEST(SizesAndStridesTest, SizesIterator) {
  SizesAndStrides ss;
  ss.set_sizes({2, 3, 4});

  int64_t sum = 0;
  for (auto it = ss.sizes_begin(); it != ss.sizes_end(); ++it) {
    sum += *it;
  }
  EXPECT_EQ(sum, 9);
}

TEST(SizesAndStridesTest, StridesIterator) {
  SizesAndStrides ss;
  ss.set_sizes({2, 3, 4});
  ss.set_strides({12, 4, 1});

  int64_t sum = 0;
  for (auto it = ss.strides_begin(); it != ss.strides_end(); ++it) {
    sum += *it;
  }
  EXPECT_EQ(sum, 17);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(SizesAndStridesTest, ScalarTensor) {
  SizesAndStrides ss;
  ss.resize(0);

  EXPECT_EQ(ss.size(), 0u);
}

TEST(SizesAndStridesTest, OneDimensional) {
  SizesAndStrides ss;
  ss.set_sizes({10});
  ss.set_strides({1});

  EXPECT_EQ(ss.size(), 1u);
  EXPECT_EQ(ss.size_at(0), 10);
  EXPECT_EQ(ss.stride_at(0), 1);
}

} // namespace executorch::backends::aoti::slim::c10
