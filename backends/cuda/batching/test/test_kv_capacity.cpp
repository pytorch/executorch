/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cuda/batching/kv_capacity.h>

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace cb = ::executorch::backends::cuda::batching;

namespace {

bool contains(const std::string& text, const std::string& part) {
  return text.find(part) != std::string::npos;
}

// Muse Glimmer: 52 layers x K/V x 2 KV heads x 128 dims x BF16.
constexpr std::int64_t kMuseGlimmerBytesPerCell = 52 * 2 * 2 * 128 * 2;

// The serving configuration of issue 25648: four live sessions, eight retained
// prefixes and four capture slots, each of 131072 tokens, over 131072 cells.
const cb::KvReservations kIssue{4, 8, 4};

} // namespace

TEST(KvCapacityTest, BudgetKeepsOneCellForPadding) {
  // n sessions of t tokens fit exactly when n * t + 1 <= cells.
  EXPECT_TRUE(cb::cell_budget(4, 8191, 32765).fits());
  EXPECT_FALSE(cb::cell_budget(4, 8191, 32764).fits());
  EXPECT_TRUE(cb::cell_budget(1, 131072, 131073).fits());
  EXPECT_FALSE(cb::cell_budget(1, 131072, 131072).fits());
}

TEST(KvCapacityTest, BudgetOfTheIssueConfiguration) {
  const cb::CellBudget budget = cb::cell_budget(16, 131072, 131072);
  EXPECT_FALSE(budget.fits());
  EXPECT_EQ(budget.required_cells, 2097153);
  EXPECT_EQ(budget.available_cells, 131072);
  EXPECT_EQ(budget.max_session_tokens, 8191);
  // The suggested context fits; one more token does not.
  EXPECT_TRUE(cb::cell_budget(16, 8191, 131072).fits());
  EXPECT_FALSE(cb::cell_budget(16, 8192, 131072).fits());
}

TEST(KvCapacityTest, BudgetWithoutRoomForAnySession) {
  EXPECT_EQ(cb::cell_budget(8, 1, 8).max_session_tokens, 0);
  EXPECT_EQ(cb::cell_budget(7, 1, 8).max_session_tokens, 1);
  EXPECT_EQ(cb::cell_budget(0, 1, 8).max_session_tokens, 0);
  EXPECT_EQ(cb::cell_budget(4, 1, 1).max_session_tokens, 0);
}

TEST(KvCapacityTest, ReservationsTotalAndKnown) {
  EXPECT_FALSE(cb::KvReservations{}.known());
  EXPECT_EQ(cb::KvReservations{}.total(), 0);
  EXPECT_TRUE(kIssue.known());
  EXPECT_EQ(kIssue.total(), 16);
  EXPECT_TRUE((cb::KvReservations{0, 0, 1}.known()));
}

TEST(KvCapacityTest, DescribesTheBreakdown) {
  EXPECT_EQ(
      cb::describe_sessions(16, kIssue),
      "16 resident sessions (4 live + 8 prefix entries + 4 capture slots)");
  EXPECT_EQ(cb::describe_sessions(16, {}), "16 resident sessions");
  // A caller that holds extra rows (e.g. temporaries) leaves them "other".
  EXPECT_EQ(
      cb::describe_sessions(18, kIssue),
      "18 resident sessions (4 live + 8 prefix entries + 4 capture slots"
      " + 2 other)");
  // A breakdown larger than the total is inconsistent and is dropped.
  EXPECT_EQ(cb::describe_sessions(8, kIssue), "8 resident sessions");
}

TEST(KvCapacityTest, FormatsMemoryInMibBelowOneGib) {
  EXPECT_EQ(cb::format_memory(0), "0.0 MiB");
  EXPECT_EQ(cb::format_memory(1536 * 1024), "1.5 MiB");
  EXPECT_EQ(cb::format_memory(std::int64_t{512} << 20), "512.0 MiB");
  EXPECT_EQ(cb::format_memory(std::int64_t{1} << 30), "1.0 GiB");
  // Units follow the rounded value: just under 1 GiB is not "1024.0 MiB".
  EXPECT_EQ(cb::format_memory((std::int64_t{1} << 30) - 1), "1.0 GiB");
  EXPECT_EQ(cb::format_memory(1073636966), "1023.9 MiB");
  // The issue's estimate: 2097153 cells x 53248 bytes.
  EXPECT_EQ(cb::format_memory(2097153 * kMuseGlimmerBytesPerCell), "104.0 GiB");
}

TEST(KvCapacityTest, BudgetHoldsReservedCells) {
  // 4 sessions x 100 tokens + 64 reserved + 1 padding = 465 cells.
  EXPECT_TRUE(cb::cell_budget(4, 100, 465, 64).fits());
  EXPECT_FALSE(cb::cell_budget(4, 100, 464, 64).fits());
  EXPECT_EQ(cb::cell_budget(4, 100, 464, 64).required_cells, 465);
  EXPECT_EQ(cb::cell_budget(4, 100, 464, 64).max_session_tokens, 99);
  // Reservations that consume every cell leave no budget.
  EXPECT_EQ(cb::cell_budget(4, 1, 65, 64).max_session_tokens, 0);
}

TEST(KvCapacityTest, CellShortfallOfTheIssueConfiguration) {
  ASSERT_EQ(kMuseGlimmerBytesPerCell, 53248);
  const std::vector<std::string> lines = cb::cell_shortfall_lines(
      16,
      131072,
      131072,
      kIssue,
      {kMuseGlimmerBytesPerCell, /*has_windowed_layers=*/true});
  ASSERT_EQ(lines.size(), 5u);
  // The first line keeps the message existing logs and scripts match.
  EXPECT_EQ(
      lines[0],
      "CudaExecutor: 16 sessions of 131072 tokens exceed the program's 131072 "
      "cells, one of which is reserved");
  EXPECT_EQ(
      lines[1],
      "CudaExecutor: 16 resident sessions (4 live + 8 prefix entries + 4 "
      "capture slots) share one cell pool; 16 x 131072 + 1 padding = 2097153 "
      "cells needed, 131072 available");
  EXPECT_TRUE(contains(lines[2], "at most 8191 tokens per session fit"))
      << lines[2];
  EXPECT_EQ(
      lines[3],
      "CudaExecutor: or re-export with --max-cells >= 2097153 (104.0 GiB of "
      "K/V at 53248 bytes per cell)");
  EXPECT_EQ(
      lines[4],
      "CudaExecutor: this model has sliding-window layers; --kv-layout "
      "sequence keeps them in ring buffers and may need far less memory");
  for (const std::string& line : lines) {
    // ET_LOG truncates a message at 256 bytes.
    EXPECT_LT(line.size(), 256u) << line;
  }
}

TEST(KvCapacityTest, CellShortfallWithoutBreakdownOrGeometry) {
  const std::vector<std::string> lines =
      cb::cell_shortfall_lines(4, 40000, 131072, {});
  // Without windowed layers there is no sequence-layout hint.
  ASSERT_EQ(lines.size(), 4u);
  EXPECT_TRUE(contains(
      lines[1],
      "4 resident sessions share one cell pool; 4 x 40000 + 1 padding = "
      "160001 cells needed, 131072 available"))
      << lines[1];
  EXPECT_TRUE(contains(lines[2], "at most 32767 tokens per session fit"))
      << lines[2];
  EXPECT_EQ(lines[3], "CudaExecutor: or re-export with --max-cells >= 160001");
}

TEST(KvCapacityTest, CellShortfallWithNoTokenPerSession) {
  const std::vector<std::string> lines = cb::cell_shortfall_lines(
      8, 4, 8, cb::KvReservations{8, 0, 0}, {kMuseGlimmerBytesPerCell, false});
  ASSERT_EQ(lines.size(), 4u);
  EXPECT_EQ(
      lines[2],
      "CudaExecutor: not even one token per session fits; reserve at most 7 "
      "sessions");
  // A small pool's estimate reads in MiB, not "0.0 GiB".
  EXPECT_EQ(
      lines[3],
      "CudaExecutor: or re-export with --max-cells >= 33 (1.7 MiB of K/V at "
      "53248 bytes per cell)");
}

TEST(KvCapacityTest, SequenceLayoutShortfall) {
  const std::vector<std::string> lines =
      cb::sequence_shortfall_lines(16, 4, kIssue, /*sequence_layout=*/true);
  ASSERT_EQ(lines.size(), 3u);
  EXPECT_EQ(
      lines[0],
      "CudaExecutor: 16 resident sessions requested, but the layout holds 4");
  EXPECT_EQ(
      lines[1],
      "CudaExecutor: 16 resident sessions (4 live + 8 prefix entries + 4 "
      "capture slots) each need a sequence column");
  EXPECT_TRUE(contains(lines[2], "--max-kv-seqs >= 16")) << lines[2];
}

TEST(KvCapacityTest, CellLayoutSequenceLimit) {
  // The cell layout tracks at most 64 sequences, whatever its cells.
  const std::vector<std::string> lines = cb::sequence_shortfall_lines(
      80, 64, cb::KvReservations{16, 48, 16}, /*sequence_layout=*/false);
  ASSERT_EQ(lines.size(), 3u);
  EXPECT_EQ(
      lines[0],
      "CudaExecutor: 80 resident sessions requested, but the cell layout "
      "tracks at most 64");
  EXPECT_EQ(
      lines[1],
      "CudaExecutor: 80 resident sessions (16 live + 48 prefix entries + 16 "
      "capture slots) each take one of its sequence slots");
  EXPECT_EQ(
      lines[2],
      "CudaExecutor: reserve fewer live sessions, prefix entries or capture "
      "slots, or re-export with --kv-layout sequence --max-kv-seqs >= 80");
  for (const std::string& line : lines) {
    EXPECT_FALSE(contains(line, "sequence column")) << line;
  }
}

TEST(KvCapacityTest, DraftShortfallCountsTemporaryBlocks) {
  // 4 sessions x 1000 tokens + 4 owners x 16 block cells + 1 padding.
  const std::vector<std::string> lines =
      cb::draft_shortfall_lines(4, 1000, 4000, 64, cb::KvReservations{2, 1, 1});
  ASSERT_EQ(lines.size(), 2u);
  EXPECT_EQ(
      lines[0],
      "DFlash: 4 resident sessions (2 live + 1 prefix entries + 1 capture "
      "slots) x 1000 + 64 temporary draft-block cells + 1 padding = 4065 "
      "draft cells needed, 4000 available");
  // floor((4000 - 1 - 64) / 4) = 983.
  EXPECT_EQ(
      lines[1],
      "DFlash: at most 983 tokens per session fit the draft pool; lower the "
      "session token limit, reserve fewer sessions, or re-export with "
      "--draft-max-cells >= 4065");
  EXPECT_TRUE(cb::cell_budget(4, 983, 4000, 64).fits());
  EXPECT_FALSE(cb::cell_budget(4, 984, 4000, 64).fits());
}

TEST(KvCapacityTest, DraftShortfallWithoutRoomForBlocks) {
  const std::vector<std::string> lines =
      cb::draft_shortfall_lines(2, 10, 50, 64, {});
  ASSERT_EQ(lines.size(), 2u);
  EXPECT_EQ(
      lines[1],
      "DFlash: the draft pool cannot hold its temporary blocks; re-export with "
      "--draft-max-cells >= 85");
}
