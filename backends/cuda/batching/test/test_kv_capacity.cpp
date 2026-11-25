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
namespace llm_cache = ::executorch::extension::llm::cache;
using ::executorch::aten::ScalarType;

namespace {

bool contains(const std::string& text, const std::string& part) {
  return text.find(part) != std::string::npos;
}

bool any_contains(
    const std::vector<std::string>& lines,
    const std::string& part) {
  for (const std::string& line : lines) {
    if (contains(line, part)) {
      return true;
    }
  }
  return false;
}

// 52 layers x K/V x 2 KV heads x 128 dims x BF16.
constexpr std::int64_t kBytesPerCell = 52 * 2 * 2 * 128 * 2;

// The configuration of issue 25648: four live sessions, eight retained
// prefixes and four capture slots, each of 131072 tokens, over 131072 cells.
const cb::KvReservations kIssue{4, 8, 4};

cb::KvCapacityReport cell_report(
    cb::KvAdmission outcome,
    int sessions,
    int session_tokens,
    std::int64_t cells,
    cb::KvReservations reservations = {}) {
  cb::KvCapacityReport report;
  report.outcome = outcome;
  report.sessions = sessions;
  report.reservations = reservations;
  report.session_tokens = session_tokens;
  report.context_length = 131072;
  report.cells = cells;
  report.session_limit = 64;
  return report;
}

llm_cache::LayerGeometry layer(int heads, int dim, int window = 0) {
  llm_cache::LayerGeometry geometry{};
  geometry.n_kv_heads = heads;
  geometry.head_dim = dim;
  if (window > 0) {
    geometry.policy.kind = llm_cache::LayerPolicy::Kind::Ring;
    geometry.policy.window = window;
  }
  return geometry;
}

void expect_short_lines(const std::vector<std::string>& lines) {
  for (const std::string& line : lines) {
    // ET_LOG truncates a message at 256 bytes.
    EXPECT_LT(line.size(), 256u) << line;
    // The backend's wording names no export or serving option.
    EXPECT_FALSE(contains(line, "--")) << line;
  }
}

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
  EXPECT_TRUE(cb::cell_budget(16, 8191, 131072).fits());
  EXPECT_FALSE(cb::cell_budget(16, 8192, 131072).fits());
}

TEST(KvCapacityTest, BudgetHoldsReservedCells) {
  // 4 sessions x 100 tokens + 64 reserved + 1 padding = 465 cells.
  EXPECT_TRUE(cb::cell_budget(4, 100, 465, 64).fits());
  EXPECT_FALSE(cb::cell_budget(4, 100, 464, 64).fits());
  EXPECT_EQ(cb::cell_budget(4, 100, 464, 64).required_cells, 465);
  EXPECT_EQ(cb::cell_budget(4, 100, 464, 64).max_session_tokens, 99);
  EXPECT_EQ(cb::cell_budget(4, 1, 65, 64).max_session_tokens, 0);
  EXPECT_EQ(cb::cell_budget(0, 1, 8).max_session_tokens, 0);
}

TEST(KvCapacityTest, DescribesTheBreakdownOnlyWhenItTotals) {
  EXPECT_EQ(
      cb::describe_sessions(16, kIssue),
      "16 resident sessions (4 live + 8 prefix entries + 4 capture slots)");
  EXPECT_EQ(cb::describe_sessions(16, {}), "16 resident sessions");
  EXPECT_EQ(cb::describe_sessions(18, kIssue), "18 resident sessions");
}

TEST(KvCapacityTest, FormatsMemoryByTheRoundedValue) {
  EXPECT_EQ(cb::format_memory(0), "0.0 MiB");
  EXPECT_EQ(cb::format_memory(1536 * 1024), "1.5 MiB");
  EXPECT_EQ(cb::format_memory(std::int64_t{512} << 20), "512.0 MiB");
  EXPECT_EQ(cb::format_memory(1073636966), "1023.9 MiB");
  // Just under 1 GiB rounds to 1024.0 MiB, so it reads in GiB.
  EXPECT_EQ(cb::format_memory((std::int64_t{1} << 30) - 1), "1.0 GiB");
  EXPECT_EQ(cb::format_memory(std::int64_t{1} << 30), "1.0 GiB");
  EXPECT_EQ(cb::format_memory(2097153 * kBytesPerCell), "104.0 GiB");
}

TEST(KvCapacityTest, PoolFactsSizeEveryLayerAndFindWindows) {
  llm_cache::CacheGeometry geometry;
  geometry.layers = {layer(2, 128, 2048), layer(2, 128), layer(4, 64, 512)};
  // 2 x (2 x 128 + 2 x 128 + 4 x 64) elements per token.
  const cb::CellPoolFacts bf16 =
      cb::cell_pool_facts(geometry, static_cast<int>(ScalarType::BFloat16));
  EXPECT_EQ(bf16.bytes_per_cell, 2 * 768 * 2);
  EXPECT_TRUE(bf16.has_windowed_layers);
  // The cache's other dense dtypes are sized too.
  EXPECT_EQ(
      cb::cell_pool_facts(geometry, static_cast<int>(ScalarType::Float))
          .bytes_per_cell,
      2 * 768 * 4);
  EXPECT_EQ(
      cb::cell_pool_facts(geometry, static_cast<int>(ScalarType::Char))
          .bytes_per_cell,
      2 * 768);
  // A dtype that is not a ScalarType leaves the estimate out.
  EXPECT_EQ(cb::cell_pool_facts(geometry, -1).bytes_per_cell, 0);
  EXPECT_EQ(
      cb::cell_pool_facts(geometry, static_cast<int>(ScalarType::Undefined))
          .bytes_per_cell,
      0);
}

TEST(KvCapacityTest, PoolFactsWithoutWindows) {
  llm_cache::CacheGeometry geometry;
  geometry.layers = {layer(8, 128), layer(8, 128)};
  const cb::CellPoolFacts facts =
      cb::cell_pool_facts(geometry, static_cast<int>(ScalarType::Half));
  EXPECT_EQ(facts.bytes_per_cell, 2 * 2 * 8 * 128 * 2);
  EXPECT_FALSE(facts.has_windowed_layers);
}

TEST(KvCapacityTest, CellShortfallOfTheIssueConfiguration) {
  cb::KvCapacityReport report =
      cell_report(cb::KvAdmission::kCells, 16, 131072, 131072, kIssue);
  report.facts = {kBytesPerCell, /*has_windowed_layers=*/true};
  EXPECT_EQ(cb::required_cells(report), 2097153);
  EXPECT_EQ(cb::fitting_session_tokens(report), 8191);
  EXPECT_EQ(cb::fitting_sessions(report), 0);
  const std::vector<std::string> lines = cb::describe_capacity(report);
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
  EXPECT_EQ(
      lines[2],
      "CudaExecutor: at most 8191 tokens per session fit 16 sessions");
  EXPECT_EQ(
      lines[3],
      "CudaExecutor: a program with at least 2097153 cells would hold them "
      "(104.0 GiB of K/V at 53248 bytes per cell)");
  EXPECT_TRUE(contains(lines[4], "sliding-window layers")) << lines[4];
  expect_short_lines(lines);
}

TEST(KvCapacityTest, CellShortfallWithoutWindowsHasNoLayoutHint) {
  const std::vector<std::string> lines = cb::describe_capacity(
      cell_report(cb::KvAdmission::kCells, 4, 40000, 131072));
  ASSERT_EQ(lines.size(), 4u);
  EXPECT_TRUE(contains(
      lines[1],
      "4 resident sessions share one cell pool; 4 x 40000 + 1 padding = "
      "160001 cells needed, 131072 available"))
      << lines[1];
  EXPECT_TRUE(contains(lines[2], "at most 32767 tokens per session fit"))
      << lines[2];
  EXPECT_TRUE(contains(lines[2], "or 3 sessions of 40000 tokens")) << lines[2];
  EXPECT_EQ(
      lines[3],
      "CudaExecutor: a program with at least 160001 cells would hold them");
  expect_short_lines(lines);
}

TEST(KvCapacityTest, ZeroTokenBudgetSuggestsSessionsThatFitTheLimit) {
  // 8 sessions x 4 tokens on 8 cells: 7 sessions would still need 29 cells,
  // so only one session of 4 tokens fits.
  const cb::KvCapacityReport report =
      cell_report(cb::KvAdmission::kCells, 8, 4, 8, {8, 0, 0});
  EXPECT_EQ(cb::fitting_session_tokens(report), 0);
  EXPECT_EQ(cb::fitting_sessions(report), 1);
  const std::vector<std::string> lines = cb::describe_capacity(report);
  EXPECT_EQ(
      lines[2],
      "CudaExecutor: not even one token per session fits 8 sessions; at most "
      "1 sessions of 4 tokens fit");
  EXPECT_FALSE(any_contains(lines, "at most 7")) << lines[2];
  expect_short_lines(lines);
}

TEST(KvCapacityTest, ZeroTokenBudgetWhenOneSessionDoesNotFit) {
  const std::vector<std::string> lines =
      cb::describe_capacity(cell_report(cb::KvAdmission::kCells, 8, 20, 8));
  EXPECT_EQ(
      lines[2], "CudaExecutor: one session of 20 tokens alone needs 21 cells");
}

TEST(KvCapacityTest, FittingSessionsRespectTheSessionLimit) {
  // 1000 sessions of 10 tokens fit 10001 cells, but the layout tracks 64.
  EXPECT_EQ(
      cb::fitting_sessions(cell_report(cb::KvAdmission::kCells, 70, 10, 10001)),
      64);
}

TEST(KvCapacityTest, CellLayoutSessionLimitLeadsWithTheLimit) {
  // 16 live + 48 prefix + 16 capture: no token limit or cell count helps.
  cb::KvCapacityReport report = cell_report(
      cb::KvAdmission::kSessionLimit, 80, 8192, 131072, {16, 48, 16});
  report.facts = {kBytesPerCell, true};
  const std::vector<std::string> lines = cb::describe_capacity(report);
  ASSERT_EQ(lines.size(), 2u);
  EXPECT_EQ(
      lines[0],
      "CudaExecutor: 80 resident sessions (16 live + 48 prefix entries + 16 "
      "capture slots) exceed the cell layout's limit of 64 resident sequences, "
      "whatever its cells");
  // At 64 sessions the cells would still be short: 15 x 8192 + 1 fit.
  EXPECT_EQ(
      lines[1],
      "CudaExecutor: 15 sessions of 8192 tokens fit its cells and limit; a "
      "sequence-layout program would need at least 80 sequence columns");
  EXPECT_FALSE(any_contains(lines, "tokens per session"));
  EXPECT_FALSE(any_contains(lines, "at most 64"));
  expect_short_lines(lines);
}

TEST(KvCapacityTest, AutomaticSessionLimitNamesTheLimit) {
  cb::KvCapacityReport report = cell_report(
      cb::KvAdmission::kSessionLimit, 68, 1927, 131072, {32, 32, 4});
  report.automatic = true;
  const std::vector<std::string> lines = cb::describe_capacity(report);
  ASSERT_EQ(lines.size(), 2u);
  EXPECT_EQ(
      lines[1],
      "CudaExecutor: at most 64 sessions fit its limit; a sequence-layout "
      "program would need at least 68 sequence columns");
}

TEST(KvCapacityTest, AdmittedReportsLogNothing) {
  EXPECT_TRUE(cb::describe_capacity(
                  cell_report(cb::KvAdmission::kAdmitted, 4, 1024, 131072))
                  .empty());
}

TEST(KvCapacityTest, AutomaticSessionTokens) {
  EXPECT_EQ(cb::automatic_session_tokens(false, 16, 131072, 131072), 8191);
  EXPECT_EQ(cb::automatic_session_tokens(false, 1, 64, 256), 64);
  EXPECT_EQ(cb::automatic_session_tokens(false, 256, 64, 256), 0);
  EXPECT_EQ(cb::automatic_session_tokens(true, 16, 131072, 0), 131072);
}

TEST(KvCapacityTest, CheckAdmissionPutsTheSessionLimitFirst) {
  using cb::KvAdmission;
  const auto outcome = [](bool sequence,
                          int sessions,
                          int tokens,
                          std::int64_t cells,
                          int limit) {
    return cb::check_admission(sequence, sessions, tokens, 131072, cells, limit)
        .outcome;
  };
  EXPECT_EQ(outcome(false, 4, 1024, 4097, 64), KvAdmission::kAdmitted);
  EXPECT_EQ(outcome(false, 4, 1024, 4096, 64), KvAdmission::kCells);
  EXPECT_EQ(outcome(false, 16, 0, 131072, 64), KvAdmission::kAutomatic);
  EXPECT_EQ(outcome(false, 16, 0, 16, 64), KvAdmission::kNoBudget);
  // Past the limit, whatever the cells or the budget.
  EXPECT_EQ(outcome(false, 65, 0, 1 << 20, 64), KvAdmission::kSessionLimit);
  EXPECT_EQ(outcome(false, 65, 1024, 64, 64), KvAdmission::kSessionLimit);
  EXPECT_EQ(outcome(true, 5, 0, 0, 4), KvAdmission::kSessionLimit);
  EXPECT_EQ(outcome(true, 4, 0, 0, 4), KvAdmission::kAutomatic);
  // The sequence layout has no shared cells.
  EXPECT_EQ(outcome(true, 4, 131072, 0, 4), KvAdmission::kAdmitted);
  // The resolved budget comes with the outcome.
  EXPECT_EQ(
      cb::check_admission(false, 65, 0, 131072, 256, 64).session_tokens, 3);
}

TEST(KvCapacityTest, SequenceLayoutSessionLimit) {
  cb::KvCapacityReport report;
  report.outcome = cb::KvAdmission::kSessionLimit;
  report.sequence_layout = true;
  report.sessions = 16;
  report.reservations = kIssue;
  report.session_tokens = 131072;
  report.context_length = 131072;
  report.session_limit = 4;
  const std::vector<std::string> lines = cb::describe_capacity(report);
  ASSERT_EQ(lines.size(), 2u);
  EXPECT_EQ(
      lines[0],
      "CudaExecutor: 16 resident sessions (4 live + 8 prefix entries + 4 "
      "capture slots) exceed the program's 4 sequence columns, one per "
      "resident session");
  EXPECT_EQ(
      lines[1],
      "CudaExecutor: reserve at most 4 sessions, or serve a program with at "
      "least 16 sequence columns");
  expect_short_lines(lines);
}

TEST(KvCapacityTest, AutomaticContextReportsTheResolvedBudget) {
  const std::vector<std::string> lines = cb::describe_capacity(
      cell_report(cb::KvAdmission::kAutomatic, 16, 8191, 131072, kIssue));
  ASSERT_EQ(lines.size(), 1u);
  EXPECT_EQ(
      lines[0],
      "CudaExecutor: automatic context: 16 resident sessions (4 live + 8 "
      "prefix entries + 4 capture slots) share 131072 cells; each may hold "
      "8191 tokens (model context 131072)");
  expect_short_lines(lines);
}

TEST(KvCapacityTest, AutomaticContextOnTheSequenceLayout) {
  cb::KvCapacityReport report;
  report.outcome = cb::KvAdmission::kAutomatic;
  report.sequence_layout = true;
  report.sessions = 4;
  report.session_tokens = 131072;
  report.context_length = 131072;
  report.session_limit = 4;
  EXPECT_EQ(
      cb::describe_capacity(report),
      std::vector<std::string>{
          "CudaExecutor: automatic context: 4 resident sessions each get "
          "131072 tokens of a private sequence column"});
}

TEST(KvCapacityTest, NoBudgetInAutomaticContext) {
  const std::vector<std::string> lines = cb::describe_capacity(
      cell_report(cb::KvAdmission::kNoBudget, 16, 0, 8, kIssue));
  ASSERT_EQ(lines.size(), 2u);
  EXPECT_EQ(
      lines[0],
      "CudaExecutor: no cell budget remains: 16 resident sessions (4 live + 8 "
      "prefix entries + 4 capture slots) need at least 17 cells with the "
      "padding cell, the program has 8");
  EXPECT_EQ(lines[1], "CudaExecutor: at most 7 one-token sessions fit");
  expect_short_lines(lines);
}

TEST(KvCapacityTest, EntryChecksTheExecutorShares) {
  constexpr int kMax = std::numeric_limits<int>::max();
  EXPECT_TRUE(cb::valid_session_limits(4, 0));
  EXPECT_TRUE(cb::valid_session_limits(4, 1024));
  EXPECT_FALSE(cb::valid_session_limits(0, 1024));
  EXPECT_FALSE(cb::valid_session_limits(4, -1));
  // sessions x tokens must fit an int.
  EXPECT_TRUE(cb::valid_session_limits(2, kMax / 2));
  EXPECT_FALSE(cb::valid_session_limits(3, kMax / 2));
  EXPECT_TRUE(cb::within_context(131072, 131072));
  EXPECT_FALSE(cb::within_context(131073, 131072));
  EXPECT_FALSE(cb::valid_cell_count(1));
  EXPECT_TRUE(cb::valid_cell_count(2));
  EXPECT_TRUE(cb::valid_cell_count(kMax));
  EXPECT_FALSE(cb::valid_cell_count(std::int64_t{kMax} + 1));
}
