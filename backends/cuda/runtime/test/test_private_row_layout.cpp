/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/runtime/private_row_layout.h>
#include <gtest/gtest.h>

#include <limits>
#include <vector>

namespace {

using executorch::backends::cuda::make_private_row_layout;
using executorch::backends::cuda::PrivateRowLayout;
using executorch::backends::cuda::PrivateRowSpan;
using executorch::runtime::Error;

executorch::runtime::Result<PrivateRowLayout> plan(
    const std::vector<int64_t>& required,
    const std::vector<PrivateRowSpan>& previous,
    int64_t initial,
    int64_t limit) {
  return make_private_row_layout(
      {required.data(), required.size()},
      {previous.data(), previous.size()},
      initial,
      limit);
}

TEST(PrivateRowLayoutTest, EmptyOwnersAllocateOnlyScratch) {
  const std::vector<int64_t> rows(64, 0);
  const auto result = plan(rows, {}, 512, 131072);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.get().rows, 1);
  ASSERT_EQ(result.get().spans.size(), 65);
  for (size_t i = 0; i < 64; ++i) {
    EXPECT_EQ(result.get().spans[i].capacity, 0);
  }
  EXPECT_EQ(result.get().spans.back().capacity, 1);
}

TEST(PrivateRowLayoutTest, LongOwnerDoesNotGrowShortOwners) {
  std::vector<int64_t> rows(64, 1);
  const auto first = plan(rows, {}, 512, 131072);
  ASSERT_TRUE(first.ok());
  EXPECT_EQ(first.get().rows, 64 * 512 + 1);
  rows[17] = 8193;
  const auto grown = plan(rows, first.get().spans, 512, 131072);
  ASSERT_TRUE(grown.ok());
  EXPECT_EQ(grown.get().spans[17].capacity, 8193);
  EXPECT_EQ(grown.get().rows, 63 * 512 + 8193 + 1);
  for (size_t i = 0; i < 64; ++i) {
    if (i != 17) {
      EXPECT_EQ(grown.get().spans[i].capacity, 512);
    }
  }
  EXPECT_EQ(first.get().spans[17].capacity, 512);
}

TEST(PrivateRowLayoutTest, GrowthIsGeometricAndRetainsUntouchedSpans) {
  const std::vector<int64_t> rows{513, 3};
  const std::vector<PrivateRowSpan> previous{{0, 512}, {512, 512}, {1024, 1}};
  const auto result = plan(rows, previous, 512, 131072);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.get().spans[0].capacity, 1024);
  EXPECT_EQ(result.get().spans[1].capacity, 512);
  EXPECT_EQ(result.get().spans[1].base, 1024);
  EXPECT_EQ(result.get().rows, 1537);
  EXPECT_EQ(previous[1].base, 512);
}

TEST(PrivateRowLayoutTest, ClosingAnOwnerReclaimsOnlyItsSpan) {
  const std::vector<int64_t> rows{8, 0, 4};
  const std::vector<PrivateRowSpan> previous{
      {0, 16}, {16, 8192}, {8208, 8}, {8216, 1}};
  const auto result = plan(rows, previous, 4, 131072);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.get().spans[0].capacity, 16);
  EXPECT_EQ(result.get().spans[1].capacity, 0);
  EXPECT_EQ(result.get().spans[2].capacity, 8);
  EXPECT_EQ(result.get().spans[2].base, 16);
  EXPECT_EQ(result.get().rows, 25);
}

TEST(PrivateRowLayoutTest, ReopenedOwnerStartsAtInitialCapacity) {
  const std::vector<int64_t> rows{1, 1};
  const std::vector<PrivateRowSpan> previous{{0, 2048}, {2048, 0}, {2048, 1}};
  const auto result = plan(rows, previous, 512, 131072);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.get().spans[0].capacity, 2048);
  EXPECT_EQ(result.get().spans[1].capacity, 512);
  EXPECT_EQ(result.get().rows, 2561);
}

TEST(PrivateRowLayoutTest, RingRowsGrowBeforeReachingWindowLimit) {
  const std::vector<int64_t> rows{513, 1};
  const std::vector<PrivateRowSpan> previous{{0, 512}, {512, 512}, {1024, 1}};
  const auto result = plan(rows, previous, 512, 768);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result.get().spans[0].capacity, 768);
  EXPECT_EQ(result.get().spans[1].capacity, 512);
  EXPECT_EQ(result.get().rows, 1281);
  const std::vector<int64_t> small_window{80};
  const auto small = plan(small_window, {}, 512, 80);
  ASSERT_TRUE(small.ok());
  EXPECT_EQ(small.get().rows, 81);
}

TEST(PrivateRowLayoutTest, InvalidLaterOwnerDoesNotMutateLiveLayout) {
  const std::vector<int64_t> rows{513, 131073};
  const std::vector<PrivateRowSpan> previous{{0, 512}, {512, 512}, {1024, 1}};
  const auto result = plan(rows, previous, 512, 131072);
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.error(), Error::InvalidArgument);
  EXPECT_EQ(previous[0].capacity, 512);
  EXPECT_EQ(previous[1].base, 512);
}

TEST(PrivateRowLayoutTest, RejectsDescriptorOverflowAndInvalidGeometry) {
  const int64_t max = std::numeric_limits<int32_t>::max();
  const std::vector<int64_t> rows{max};
  EXPECT_FALSE(plan(rows, {}, 512, max).ok());
  const std::vector<int64_t> two{max / 2 + 1, max / 2 + 1};
  EXPECT_FALSE(plan(two, {}, 512, max).ok());
  const std::vector<int64_t> one{1};
  EXPECT_FALSE(plan(one, {}, 0, 128).ok());
  EXPECT_FALSE(plan(one, {}, 512, 0).ok());
  const std::vector<PrivateRowSpan> wrong{{0, 2}};
  EXPECT_FALSE(plan(one, wrong, 2, 128).ok());
  const std::vector<int64_t> negative{-1};
  EXPECT_FALSE(plan(negative, {}, 512, 128).ok());
}

TEST(PrivateRowLayoutTest, RejectsOverlappingSpansAndInvalidScratch) {
  const std::vector<int64_t> rows{1, 1};
  const std::vector<PrivateRowSpan> overlap{{0, 8}, {7, 8}, {15, 1}};
  EXPECT_FALSE(plan(rows, overlap, 8, 128).ok());
  const std::vector<PrivateRowSpan> gap{{0, 8}, {9, 8}, {17, 1}};
  EXPECT_FALSE(plan(rows, gap, 8, 128).ok());
  const std::vector<PrivateRowSpan> scratch{{0, 8}, {8, 8}, {16, 0}};
  EXPECT_FALSE(plan(rows, scratch, 8, 128).ok());
}

} // namespace
