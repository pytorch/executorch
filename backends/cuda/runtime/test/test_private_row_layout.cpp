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
#include <random>
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
    int64_t limit,
    bool compact = false) {
  return make_private_row_layout(
      {required.data(), required.size()},
      {previous.data(), previous.size()},
      initial,
      limit,
      compact);
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
  EXPECT_EQ(grown.get().rows, 64 * 512 * 2 + 1);
  for (size_t i = 0; i < 64; ++i) {
    if (i != 17) {
      EXPECT_EQ(grown.get().spans[i].capacity, 512);
      EXPECT_EQ(grown.get().spans[i].base, first.get().spans[i].base);
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
  EXPECT_EQ(result.get().spans[1].base, 512);
  EXPECT_EQ(result.get().spans[0].base, 1024);
  EXPECT_EQ(result.get().rows, 2049);
  EXPECT_EQ(previous[1].base, 512);
}

TEST(PrivateRowLayoutTest, ClosingAnOwnerReclaimsOnlyItsSpan) {
  const std::vector<int64_t> rows{8, 0, 4};
  const std::vector<PrivateRowSpan> previous{
      {0, 16}, {16, 8192}, {8208, 8}, {8216, 1}};
  const auto result = plan(rows, previous, 4, 131072, true);
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
  EXPECT_EQ(result.get().rows, 4097);
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
  EXPECT_TRUE(plan(rows, gap, 8, 128).ok());
  const std::vector<PrivateRowSpan> scratch{{0, 8}, {8, 8}, {16, 0}};
  EXPECT_FALSE(plan(rows, scratch, 8, 128).ok());
}

TEST(PrivateRowLayoutTest, StaggeredAdmissionUsesGeometricArenaGrowth) {
  std::vector<int64_t> required(64, 0);
  PrivateRowLayout previous;
  int moves = 0;
  for (size_t owner = 0; owner < required.size(); ++owner) {
    required[owner] = 1;
    auto result = plan(required, previous.spans, 2, 131072);
    ASSERT_TRUE(result.ok());
    const auto& next = result.get();
    moves += next.rows != previous.rows;
    EXPECT_FALSE(next.repacked);
    for (size_t old = 0; old < owner; ++old) {
      EXPECT_EQ(next.spans[old].base, previous.spans[old].base);
      EXPECT_EQ(next.spans[old].capacity, previous.spans[old].capacity);
    }
    previous = next;
  }
  EXPECT_LE(moves, 7);
  EXPECT_EQ(previous.rows, 129);
}

TEST(PrivateRowLayoutTest, SlackSupportsInPlaceGrowthAndDisjointAdmission) {
  const std::vector<PrivateRowSpan> previous{{0, 4}, {12, 4}, {0, 0}, {31, 1}};
  auto result = plan({5, 1, 1}, previous, 4, 128);
  ASSERT_TRUE(result.ok());
  const auto& next = result.get();
  EXPECT_EQ(next.rows, 32);
  EXPECT_FALSE(next.repacked);
  EXPECT_EQ(next.spans[0].base, 0);
  EXPECT_EQ(next.spans[0].capacity, 8);
  EXPECT_EQ(next.spans[1].base, 12);
  EXPECT_EQ(next.spans[2].base, 8);
}

TEST(PrivateRowLayoutTest, FragmentationAtTheCeilingPacksIntoFreshStorage) {
  const std::vector<PrivateRowSpan> previous{{8, 8}, {4, 4}, {16, 1}};
  const auto result = plan({8, 8}, previous, 4, 8);
  ASSERT_TRUE(result.ok());
  EXPECT_TRUE(result.get().repacked);
  EXPECT_EQ(result.get().rows, 17);
  EXPECT_EQ(result.get().spans[0].base, 0);
  EXPECT_EQ(result.get().spans[1].base, 8);
}

TEST(PrivateRowLayoutTest, MixedGrowthAndReusePreserveDisjointPayloadRanges) {
  std::mt19937 rng(917);
  std::vector<int64_t> required(64, 0);
  PrivateRowLayout previous;
  for (int iteration = 0; iteration < 1500; ++iteration) {
    const size_t owner = rng() % required.size();
    required[owner] = rng() % 257;
    const auto result =
        plan(required, previous.spans, 2, 256, iteration % 23 == 0);
    ASSERT_TRUE(result.ok()) << iteration;
    const auto& next = result.get();
    EXPECT_LE(next.rows, 64 * 256 + 1);
    for (size_t i = 0; i < required.size(); ++i) {
      const auto dst = next.spans[i];
      EXPECT_GE(dst.capacity, required[i]);
      EXPECT_LE(dst.capacity, 256);
      EXPECT_GE(dst.base, 0);
      EXPECT_LE(static_cast<int64_t>(dst.base) + dst.capacity, next.rows - 1);
      if (dst.capacity == 0) {
        continue;
      }
      for (size_t j = 0; j < i; ++j) {
        const auto other = next.spans[j];
        if (other.capacity > 0) {
          EXPECT_TRUE(
              dst.base + dst.capacity <= other.base ||
              other.base + other.capacity <= dst.base);
        }
      }
      // In-place copies must never overwrite another old source, including
      // an owner's source that is relocated later in the same transaction.
      if (!previous.spans.empty() && !next.repacked &&
          next.rows == previous.rows && dst.base != previous.spans[i].base) {
        for (size_t j = 0; j < required.size(); ++j) {
          const auto src = previous.spans[j];
          if (src.capacity > 0) {
            EXPECT_TRUE(
                dst.base + dst.capacity <= src.base ||
                src.base + src.capacity <= dst.base);
          }
        }
      }
    }
    previous = next;
  }
}

} // namespace
