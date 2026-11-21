/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/runtime/ring_history.h>
#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace {

using executorch::backends::cuda::RingHistory;
using executorch::runtime::Error;
using executorch::runtime::Span;

constexpr size_t kRowBytes = sizeof(uint32_t);
constexpr size_t kPlanes = 4;

uint32_t payload(int32_t owner, int position, size_t plane) {
  return static_cast<uint32_t>(owner) * 65536 + position * 8 + plane;
}

std::vector<uint8_t> pack(
    const std::vector<RingHistory::Run>& runs,
    int width) {
  std::vector<uint8_t> packed(kPlanes * width * kRowBytes, 0xff);
  for (const auto& run : runs) {
    for (size_t plane = 0; plane < kPlanes; ++plane) {
      for (int row = 0; row < run.length; ++row) {
        const uint32_t value =
            payload(run.owner, run.absolute_start + row, plane);
        std::memcpy(
            packed.data() +
                (plane * width + run.packed_start + row) * kRowBytes,
            &value,
            kRowBytes);
      }
    }
  }
  return packed;
}

void append(
    RingHistory& history,
    const std::vector<RingHistory::Run>& runs,
    int width) {
  const auto packed = pack(runs, width);
  ASSERT_EQ(
      history.append(
          {runs.data(), runs.size()}, {packed.data(), packed.size()}, width),
      Error::Ok);
}

void expect_rows(
    const RingHistory& history,
    int32_t owner,
    int start,
    int count) {
  for (size_t plane = 0; plane < kPlanes; ++plane) {
    int row = start;
    while (row < start + count) {
      const auto chunk = history.read(owner, row, start + count - row, plane);
      ASSERT_TRUE(chunk);
      ASSERT_GT(chunk->count, 0);
      for (int offset = 0; offset < chunk->count; ++offset) {
        uint32_t actual = 0;
        std::memcpy(&actual, chunk->data + offset * kRowBytes, kRowBytes);
        EXPECT_EQ(actual, payload(owner, row + offset, plane));
      }
      row += chunk->count;
    }
  }
}

class RingHistoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto result = RingHistory::create(80, 96, 512, kRowBytes, kPlanes);
    ASSERT_TRUE(result.ok());
    history = std::move(result.get());
  }
  std::unique_ptr<RingHistory> history;
};

TEST(RingRestorePlanTest, RestoresOnlyTheOverwrittenPrefix) {
  const auto plan = RingHistory::restore_plan(2048, 9999, 10000);
  ASSERT_TRUE(plan);
  EXPECT_EQ(plan->start, 7952);
  EXPECT_EQ(plan->resident_start, 7953);
  EXPECT_EQ(plan->end, 9999);
  EXPECT_EQ(plan->resident_start - plan->start, 1);
}

TEST(RingRestorePlanTest, NoopAndUnwrappedRewindsNeedNoHostCopies) {
  for (const auto& [keep, end] :
       {std::pair{10000, 10000}, std::pair{7, 11}, std::pair{0, 0}}) {
    const auto plan = RingHistory::restore_plan(2048, keep, end);
    ASSERT_TRUE(plan);
    EXPECT_EQ(plan->resident_start, plan->start);
    EXPECT_EQ(plan->end, keep);
  }
}

TEST(RingRestorePlanTest, OldCloneAndWindowOneBoundaries) {
  const auto old = RingHistory::restore_plan(2048, 5000, 10000);
  ASSERT_TRUE(old);
  EXPECT_EQ(old->start, 2953);
  EXPECT_EQ(old->resident_start, 5000);
  EXPECT_EQ(old->end, 5000);
  const auto single = RingHistory::restore_plan(1, 5000, 10000);
  ASSERT_TRUE(single);
  EXPECT_EQ(single->start, 5000);
  EXPECT_EQ(single->resident_start, 5000);
  EXPECT_EQ(single->end, 5000);
  EXPECT_FALSE(RingHistory::restore_plan(0, 1, 2));
  EXPECT_FALSE(RingHistory::restore_plan(8, -1, 2));
  EXPECT_FALSE(RingHistory::restore_plan(8, 3, 2));
}

TEST_F(RingHistoryTest, InterleavedOwnersAndPlanesRemainIndependent) {
  append(*history, {{0, 0, 3, 0}, {1, 3, 2, 0}, {0, 5, 4, 3}}, 16);
  expect_rows(*history, 0, 0, 7);
  expect_rows(*history, 1, 0, 2);
  EXPECT_FALSE(history->read(0, 7, 1, 0));
  EXPECT_FALSE(history->read(1, 2, 1, 0));
  EXPECT_FALSE(history->read(2, 0, 1, 0));
}

TEST_F(RingHistoryTest, PageAndArchiveBoundariesPreserveExactRows) {
  append(*history, {{0, 0, 96, 0}}, 96);
  append(*history, {{0, 0, 96, 96}}, 96);
  append(*history, {{0, 0, 96, 192}}, 96);
  // The 175-row archive wraps at a different boundary from its 64-row pages.
  expect_rows(*history, 0, 113, 175);
  EXPECT_FALSE(history->read(0, 112, 1, 0));
  const auto boundary = history->read(0, 174, 20, 0);
  ASSERT_TRUE(boundary);
  EXPECT_EQ(boundary->count, 1);
  EXPECT_TRUE(history->rewind(0, 193));
  expect_rows(*history, 0, 113, 80);
  EXPECT_TRUE(history->rewind(0, 192));
  EXPECT_FALSE(history->rewind(0, 191));
}

TEST_F(RingHistoryTest, LargeStepAndManyWrapsRetainTheRewindWindow) {
  for (int start = 0; start < 480; start += 96) {
    append(*history, {{5, 0, 96, start}}, 96);
  }
  expect_rows(*history, 5, 305, 175);
  EXPECT_TRUE(history->rewind(5, 385));
  expect_rows(*history, 5, 305, 80);
  append(*history, {{5, 0, 96, 385}}, 96);
  expect_rows(*history, 5, 401, 80);
  EXPECT_FALSE(history->read(5, 304, 1, 0));
}

TEST_F(RingHistoryTest, CloneSharesPagesAndDetachDoesNotChangeItsSource) {
  append(*history, {{0, 0, 40, 0}}, 40);
  const size_t source_bytes = history->resident_bytes();
  ASSERT_GT(source_bytes, 0);
  ASSERT_TRUE(history->clone(0, 7, 32));
  EXPECT_EQ(history->resident_bytes(), source_bytes);
  // A clone retains the source's values, not values derived from its new id.
  const auto before = history->read(7, 31, 1, 2);
  ASSERT_TRUE(before);
  uint32_t value = 0;
  std::memcpy(&value, before->data, kRowBytes);
  EXPECT_EQ(value, payload(0, 31, 2));
  append(*history, {{7, 0, 8, 32}}, 8);
  EXPECT_GT(history->resident_bytes(), source_bytes);
  expect_rows(*history, 0, 0, 40);
  expect_rows(*history, 7, 32, 8);
  history->erase(7);
  EXPECT_EQ(history->resident_bytes(), source_bytes);
}

TEST_F(RingHistoryTest, RewindingOneOwnerDoesNotRewindItsClone) {
  append(*history, {{3, 0, 90, 0}}, 90);
  ASSERT_TRUE(history->clone(3, 9, 90));
  ASSERT_TRUE(history->rewind(3, 80));
  EXPECT_FALSE(history->read(3, 80, 1, 0));
  const auto preserved = history->read(9, 80, 10, 0);
  ASSERT_TRUE(preserved);
  EXPECT_EQ(preserved->count, 10);
  append(*history, {{3, 0, 10, 80}}, 10);
  expect_rows(*history, 3, 80, 10);
  uint32_t value = 0;
  const auto clone = history->read(9, 89, 1, 3);
  ASSERT_TRUE(clone);
  std::memcpy(&value, clone->data, kRowBytes);
  EXPECT_EQ(value, payload(3, 89, 3));
}

TEST_F(RingHistoryTest, StaleRowsCannotBeRestoredOrCloned) {
  append(*history, {{0, 0, 96, 0}}, 96);
  append(*history, {{0, 0, 96, 96}}, 96);
  const size_t bytes = history->resident_bytes();
  EXPECT_FALSE(history->clone(0, 1, 95));
  EXPECT_FALSE(history->rewind(0, 95));
  EXPECT_EQ(history->resident_bytes(), bytes);
  expect_rows(*history, 0, 112, 80);
  EXPECT_TRUE(history->clone(0, 1, 96));
  EXPECT_FALSE(history->clone(0, 1, 192));
  EXPECT_FALSE(history->clone(0, -1, 192));
  EXPECT_FALSE(history->clone(123, 2, 1));
}

TEST_F(RingHistoryTest, OneTokenRollbackPreservesTheOriginalWrittenFloor) {
  for (int window : {1, 5}) {
    auto result = RingHistory::create(window, 1, 32, kRowBytes, kPlanes);
    ASSERT_TRUE(result.ok());
    auto archive = std::move(result.get());
    for (int position = 0; position < 12; ++position) {
      append(*archive, {{0, 0, 1, position}}, 1);
    }
    ASSERT_TRUE(archive->clone(0, 1, 11));
    ASSERT_TRUE(archive->rewind(0, 11));
    EXPECT_FALSE(archive->rewind(0, 10));
    EXPECT_FALSE(archive->clone(1, 2, 10));
    if (window > 1) {
      expect_rows(*archive, 0, 7, 4);
    }
    append(*archive, {{0, 0, 1, 11}}, 1);
    EXPECT_TRUE(archive->has_window(0, 11));
    EXPECT_FALSE(archive->has_window(0, 10));
  }
}

TEST_F(RingHistoryTest, InvalidLaterSegmentDoesNotModifyEarlierOwners) {
  append(*history, {{0, 0, 4, 0}, {1, 4, 4, 0}}, 8);
  const auto bytes = history->resident_bytes();
  const std::vector<RingHistory::Run> bad{{0, 0, 4, 4}, {1, 4, 4, 99}};
  const auto packed = pack(bad, 8);
  EXPECT_EQ(
      history->append(
          {bad.data(), bad.size()}, {packed.data(), packed.size()}, 8),
      Error::InvalidState);
  EXPECT_EQ(history->resident_bytes(), bytes);
  expect_rows(*history, 0, 0, 4);
  expect_rows(*history, 1, 0, 4);
  EXPECT_FALSE(history->read(0, 4, 1, 0));
  append(*history, {{0, 0, 4, 4}, {1, 4, 4, 4}}, 8);
  expect_rows(*history, 0, 0, 8);
}

TEST_F(RingHistoryTest, RejectsOverlappingPackedRowsWithoutMutation) {
  const std::vector<RingHistory::Run> bad{{0, 0, 4, 0}, {1, 3, 4, 0}};
  const auto packed = pack(bad, 8);
  EXPECT_EQ(
      history->append(
          {bad.data(), bad.size()}, {packed.data(), packed.size()}, 8),
      Error::InvalidArgument);
  EXPECT_EQ(history->resident_bytes(), 0);
  append(*history, {{0, 2, 4, 0}, {1, 6, 2, 0}}, 12);
  expect_rows(*history, 0, 0, 4);
  expect_rows(*history, 1, 0, 2);
}

TEST_F(RingHistoryTest, RejectsShortTransferAndContextOverflow) {
  const std::vector<RingHistory::Run> runs{{0, 0, 8, 0}};
  auto packed = pack(runs, 8);
  packed.pop_back();
  EXPECT_EQ(
      history->append(
          {runs.data(), runs.size()}, {packed.data(), packed.size()}, 8),
      Error::InvalidArgument);
  EXPECT_EQ(history->resident_bytes(), 0);
  for (int start = 0; start < 480; start += 96) {
    append(*history, {{0, 0, 96, start}}, 96);
  }
  const std::vector<RingHistory::Run> overflow{{0, 0, 33, 480}};
  const auto overpacked = pack(overflow, 33);
  EXPECT_EQ(
      history->append(
          {overflow.data(), overflow.size()},
          {overpacked.data(), overpacked.size()},
          33),
      Error::InvalidArgument);
  expect_rows(*history, 0, 400, 80);
}

TEST_F(RingHistoryTest, ErasingAndReusingAnOwnerCannotExposeOldRows) {
  append(*history, {{0, 0, 20, 0}, {1, 20, 20, 0}}, 40);
  history->erase(0);
  EXPECT_FALSE(history->read(0, 0, 1, 0));
  expect_rows(*history, 1, 0, 20);
  append(*history, {{0, 0, 3, 0}}, 8);
  expect_rows(*history, 0, 0, 3);
  EXPECT_FALSE(history->read(0, 3, 1, 0));
  history->clear();
  EXPECT_EQ(history->resident_bytes(), 0);
  EXPECT_FALSE(history->read(1, 0, 1, 0));
}

TEST(RingHistoryGeometryTest, ContextShorterThanWindowRetainsEveryRow) {
  auto result = RingHistory::create(80, 8, 12, kRowBytes, kPlanes);
  ASSERT_TRUE(result.ok());
  auto history = std::move(result.get());
  append(*history, {{0, 0, 8, 0}}, 8);
  append(*history, {{0, 0, 4, 8}}, 8);
  expect_rows(*history, 0, 0, 12);
  EXPECT_TRUE(history->rewind(0, 8));
  expect_rows(*history, 0, 0, 8);
}

TEST(RingHistoryGeometryTest, RejectsImpossibleGeometryBeforeAllocatingPages) {
  EXPECT_FALSE(RingHistory::create(0, 8, 12, kRowBytes, kPlanes).ok());
  EXPECT_FALSE(RingHistory::create(80, 0, 12, kRowBytes, kPlanes).ok());
  EXPECT_FALSE(RingHistory::create(80, 16, 12, kRowBytes, kPlanes).ok());
  EXPECT_FALSE(RingHistory::create(80, 8, 12, 0, kPlanes).ok());
  EXPECT_FALSE(RingHistory::create(80, 8, 12, kRowBytes, 0).ok());
  EXPECT_FALSE(
      RingHistory::create(80, 8, 12, std::numeric_limits<size_t>::max(), 2)
          .ok());
}

} // namespace
