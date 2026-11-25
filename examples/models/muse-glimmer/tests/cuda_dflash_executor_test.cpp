/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <cuda_runtime.h>
#include <executorch/examples/models/muse-glimmer/runtime/cuda_dflash_executor.h>
#include <executorch/examples/models/muse-glimmer/runtime/kv_capacity_advice.h>
#include <executorch/extension/cuda/caller_stream.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/platform/runtime.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <regex>
#include <string>
#include <vector>

using executorch::extension::from_blob;
namespace aten = executorch::aten;

namespace df = executorch::extension::llm::cuda_dflash_detail;
using executorch::extension::llm::batching::Token;

TEST(CudaDFlash, FirstRejectionReturnsOnlyCorrection) {
  const auto burst = df::make_burst({11, 12, 13}, 0, 21, 64);
  ASSERT_TRUE(burst);
  EXPECT_EQ(burst->tokens, std::vector<Token>({21}));
  EXPECT_EQ(burst->accepted, 0);
  EXPECT_FALSE(burst->bonus);
}

TEST(CudaDFlash, PartialAcceptanceEndsWithCorrection) {
  const auto burst = df::make_burst({11, 12, 13}, 1, 22, 64);
  ASSERT_TRUE(burst);
  EXPECT_EQ(burst->tokens, std::vector<Token>({11, 22}));
  EXPECT_EQ(burst->accepted, 1);
  EXPECT_FALSE(burst->bonus);
}

TEST(CudaDFlash, FullAcceptanceAddsOneBonus) {
  const auto burst = df::make_burst({11, 12, 13}, 3, 14, 64);
  ASSERT_TRUE(burst);
  EXPECT_EQ(burst->tokens, std::vector<Token>({11, 12, 13, 14}));
  EXPECT_EQ(burst->accepted, 3);
  EXPECT_TRUE(burst->bonus);
}

TEST(CudaDFlash, InvalidVerificationRejected) {
  EXPECT_FALSE(df::make_burst({}, 0, 1, 64));
  EXPECT_FALSE(df::make_burst({1, 2}, -1, 1, 64));
  EXPECT_FALSE(df::make_burst({1, 2}, 3, 1, 64));
  EXPECT_FALSE(df::make_burst({1, 2}, 1, 64, 64));
  EXPECT_FALSE(df::make_burst({1, 2}, 1, -1, 64));
}

TEST(CudaDFlash, PositionalUniformsAreSeededAndIndependentOfBatching) {
  const float u = df::positional_uniform(7, 100, 2);
  EXPECT_EQ(u, df::positional_uniform(7, 100, 2));
  EXPECT_GE(u, 0.0f);
  EXPECT_LT(u, 1.0f);
  EXPECT_NE(u, df::positional_uniform(8, 100, 2));
  EXPECT_NE(u, df::positional_uniform(7, 101, 2));
  EXPECT_NE(u, df::positional_uniform(7, 100, 3));
}

TEST(CudaDFlash, StaticPaddingPutsUnusedContextAfterEveryBlock) {
  const std::vector<std::uint8_t> a{1, 1, 2, 2}, b{3, 3};
  auto plan =
      df::pack_draft({{9, 10, 70, &a, 2}, {3, 20, 80, &b, 1}}, 4, 99, 3, 64, 2);
  ASSERT_TRUE(plan);
  const auto ids = plan->ids;
  ASSERT_TRUE(df::pad_to_static(*plan, 2, 4, 2));
  EXPECT_EQ(plan->ids, ids);
  EXPECT_EQ(
      plan->hidden,
      std::vector<std::uint8_t>(
          {1, 1, 2, 2, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}));
  // Context rows 0..2, padding 3..7, queries 8..15.
  EXPECT_EQ(
      plan->positions,
      std::vector<std::int64_t>(
          {10, 11, 20, 0, 0, 0, 0, 0, 12, 13, 14, 15, 21, 22, 23, 24}));
  EXPECT_EQ(
      plan->kv_order,
      std::vector<std::int64_t>(
          {0, 1, 8, 9, 10, 11, 2, 12, 13, 14, 15, 3, 4, 5, 6, 7}));
}

TEST(CudaDFlash, StaticPaddingRejectsLongContext) {
  const std::vector<std::uint8_t> a(2 * 5, 1);
  auto plan = df::pack_draft({{9, 10, 70, &a, 5}}, 4, 99, 5, 64, 2);
  ASSERT_TRUE(plan);
  EXPECT_FALSE(df::pad_to_static(*plan, 1, 4, 2));
}

TEST(CudaDFlash, NewPreambleAnchorIsRetainedOnce) {
  // The input token (7) has already committed. The regular preamble predicts
  // 10, then verification consumes 10 and predicts the accepted/corrected tail.
  for (const auto& predictions : std::vector<std::vector<Token>>{
           {21, 12, 13, 14}, {11, 22, 13, 14}, {11, 12, 13, 14}}) {
    const auto accepted = static_cast<std::int64_t>(
        std::mismatch(
            predictions.begin(),
            predictions.begin() + 3,
            std::vector<Token>({11, 12, 13}).begin())
            .first -
        predictions.begin());
    auto burst =
        df::make_burst({11, 12, 13}, accepted, predictions[accepted], 64);
    ASSERT_TRUE(burst);
    std::vector<Token> output{10};
    output.insert(output.end(), burst->tokens.begin(), burst->tokens.end());
    EXPECT_EQ(output.front(), 10);
    EXPECT_EQ(std::count(output.begin(), output.end(), 10), 1);
    EXPECT_EQ(std::count(output.begin(), output.end(), 7), 0);
    // Verification commits the anchor plus accepted proposals, never the
    // correction/bonus. This matches the runner's end + out.size() - 1 rule.
    EXPECT_EQ(output.size() - 1, burst->accepted + 1);
  }
}

TEST(CudaDFlash, BudgetRewindTrimsOnlyCommittedRows) {
  const auto burst = df::make_burst({11, 12, 13}, 3, 14, 64);
  ASSERT_TRUE(burst);
  // Draft has warmed through input.end=100; target verified four rows.
  for (int kept = 1; kept <= 4; ++kept) {
    std::vector<std::uint8_t> hidden{10, 11, 12, 13, 14, 15, 16, 17};
    const int frontier = 100 + kept - 1;
    ASSERT_TRUE(df::trim_hidden(hidden, 2, 100, frontier));
    EXPECT_EQ(hidden.size(), static_cast<std::size_t>(kept - 1) * 2);
    if (kept > 1) {
      EXPECT_EQ(hidden.front(), 10);
      EXPECT_EQ(hidden.back(), 9 + (kept - 1) * 2);
    }
  }
}

TEST(CudaDFlash, RewindBeforeDraftFrontierDropsFreshRows) {
  std::vector<std::uint8_t> hidden{1, 2, 3, 4};
  EXPECT_TRUE(df::trim_hidden(hidden, 2, 10, 8));
  EXPECT_TRUE(hidden.empty());
}

TEST(CudaDFlash, InvalidHiddenExtentRejected) {
  std::vector<std::uint8_t> hidden{1, 2, 3};
  EXPECT_FALSE(df::trim_hidden(hidden, 2, 0, 1));
  EXPECT_FALSE(df::trim_hidden(hidden, 0, 0, 1));
  EXPECT_FALSE(df::trim_hidden(hidden, 1, -1, 1));
  EXPECT_FALSE(df::trim_hidden(hidden, 1, 0, 4));
  EXPECT_EQ(hidden, std::vector<std::uint8_t>({1, 2, 3}));
}

TEST(CudaDFlash, PackingUsesExactKAndOriginalOwnerQueryOrder) {
  std::vector<std::uint8_t> a{1, 2, 3, 4}, b{5, 6};
  const auto plan =
      df::pack_draft({{9, 10, 70, &a, 2}, {3, 20, 80, &b, 1}}, 4, 99, 3, 64, 2);
  ASSERT_TRUE(plan);
  EXPECT_EQ(
      plan->tokens,
      std::vector<std::int64_t>({70, 99, 99, 99, 80, 99, 99, 99}));
  EXPECT_EQ(
      plan->positions,
      std::vector<std::int64_t>({10, 11, 20, 12, 13, 14, 15, 21, 22, 23, 24}));
  EXPECT_EQ(
      plan->kv_order,
      std::vector<std::int64_t>({0, 1, 3, 4, 5, 6, 2, 7, 8, 9, 10}));
  EXPECT_EQ(
      plan->ids, std::vector<std::int32_t>({9, 9, 9, 9, 9, 9, 3, 3, 3, 3, 3}));
  EXPECT_EQ(plan->hidden, std::vector<std::uint8_t>({1, 2, 3, 4, 5, 6}));
  EXPECT_EQ(plan->ends, std::vector<int>({12, 21}));
  EXPECT_EQ(plan->ids.size(), plan->tokens.size() + plan->hidden.size() / 2);
}

TEST(CudaDFlash, ZeroFreshContextHasNoQueryPadding) {
  std::vector<std::uint8_t> empty;
  const auto plan = df::pack_draft({{4, 10, 70, &empty, 0}}, 4, 99, 0, 64, 2);
  ASSERT_TRUE(plan);
  EXPECT_TRUE(plan->hidden.empty());
  EXPECT_EQ(plan->ids.size(), 4);
  EXPECT_EQ(plan->positions, std::vector<std::int64_t>({10, 11, 12, 13}));
  EXPECT_EQ(plan->kv_order, std::vector<std::int64_t>({0, 1, 2, 3}));
}

TEST(CudaDFlash, NearContextLimitRequiresWholeTrainedBlock) {
  std::vector<std::uint8_t> hidden{1, 2};
  EXPECT_TRUE(df::pack_draft({{4, 59, 70, &hidden, 1}}, 4, 99, 1, 64, 2));
  EXPECT_FALSE(df::pack_draft({{4, 60, 70, &hidden, 1}}, 4, 99, 1, 64, 2));
}

TEST(CudaDFlash, EveryProposalMismatchKeepsOnlyConsumedPrefix) {
  const std::vector<Token> proposals{11, 12, 13, 14};
  for (std::size_t rejected = 0; rejected <= proposals.size(); ++rejected) {
    const auto burst = df::make_burst(
        proposals,
        static_cast<std::int64_t>(rejected),
        rejected == proposals.size() ? 90 : 80,
        128);
    ASSERT_TRUE(burst);
    std::vector<Token> expected(
        proposals.begin(), proposals.begin() + rejected);
    expected.push_back(rejected == proposals.size() ? 90 : 80);
    EXPECT_EQ(burst->tokens, expected);
    EXPECT_EQ(burst->accepted, rejected);
    EXPECT_EQ(burst->bonus, rejected == proposals.size());
  }
}

TEST(CudaDFlash, HostSnapshotPartialRowsAndInvalidExtents) {
  df::HiddenSnapshot snapshot;
  std::vector<std::uint8_t> source{1, 2, 3, 4, 5, 6, 99, 99};
  auto taps = from_blob(source.data(), {1, 4, 2});
  ASSERT_TRUE(snapshot.begin(3, 2));
  EXPECT_FALSE(snapshot.capture(*taps, 1, 3));
  ASSERT_TRUE(snapshot.capture(*taps, 0, 3));
  std::fill(source.begin(), source.end(), 0);
  std::vector<std::uint8_t> kept{7, 8};
  EXPECT_FALSE(snapshot.copy_rows(kept, 2, 2));
  ASSERT_TRUE(snapshot.copy_rows(kept, 1, 2));
  ASSERT_TRUE(snapshot.fence());
  EXPECT_EQ(kept, std::vector<std::uint8_t>({7, 8, 3, 4, 5, 6}));
  EXPECT_FALSE(snapshot.begin(0, 2));
  EXPECT_FALSE(snapshot.begin(1, 0));
  ASSERT_TRUE(snapshot.begin(1, 2));
  EXPECT_FALSE(snapshot.copy_rows(kept, 0, 1));
}

TEST(CudaDFlashCuda, SnapshotSurvivesOverwritesSlicesAndFourUnequalPrefixes) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    GTEST_SKIP() << "CUDA device required";
  }
  void* borrowed = nullptr;
  ASSERT_EQ(cudaMalloc(&borrowed, 20), cudaSuccess);
  auto taps = from_blob(
      borrowed,
      {1, 5, 4},
      aten::ScalarType::Byte,
      aten::Device(aten::DeviceType::CUDA, 0),
      [](void* data) { (void)cudaFree(data); });
  cudaStream_t stream;
  ASSERT_EQ(
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  {
    executorch::extension::cuda::CallerStreamGuard guard(stream);
    df::HiddenSnapshot snapshot;
    // Grow, shrink, then reuse capacity across independent verification calls.
    for (int total : {16, 8, 16}) {
      ASSERT_TRUE(snapshot.begin(total, 4));
      for (int offset = 0; offset < total; offset += 3) {
        const int rows = std::min(3, total - offset);
        std::vector<std::uint8_t> source(20, 255); // static padding sentinel
        for (int byte = 0; byte < rows * 4; ++byte) {
          source[byte] = offset * 4 + byte + 1;
        }
        ASSERT_EQ(
            cudaMemcpyAsync(
                borrowed, source.data(), 20, cudaMemcpyHostToDevice, stream),
            cudaSuccess);
        ASSERT_TRUE(snapshot.capture(*taps, offset, rows));
        // Model the sampler/next Module call overwriting graph-owned output.
        ASSERT_EQ(cudaMemsetAsync(borrowed, 0, 20, stream), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      }
      std::array<std::vector<std::uint8_t>, 4> kept;
      const int owners = total / 4;
      for (int owner = 0; owner < owners; ++owner) {
        kept[owner] = {200, 201, 202, 203};
        ASSERT_TRUE(snapshot.copy_rows(kept[owner], owner * 4, owner + 1));
      }
      ASSERT_TRUE(snapshot.fence());
      for (int owner = 0; owner < owners; ++owner) {
        std::vector<std::uint8_t> expected{200, 201, 202, 203};
        for (int byte = 0; byte < (owner + 1) * 4; ++byte) {
          expected.push_back(owner * 16 + byte + 1);
        }
        EXPECT_EQ(kept[owner], expected);
      }
    }
  }
  // A successfully fenced snapshot must not retain a live-stream requirement.
  ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(CudaDFlash, PackingRejectsDuplicateOwnersAndOversizedContext) {
  std::vector<std::uint8_t> hidden{1, 2};
  EXPECT_FALSE(df::pack_draft(
      {{4, 0, 70, &hidden, 1}, {4, 0, 80, &hidden, 1}}, 4, 99, 2, 64, 2));
  EXPECT_FALSE(df::pack_draft({{4, 0, 70, &hidden, 2}}, 4, 99, 2, 64, 2));
  EXPECT_FALSE(df::pack_draft({{4, 0, 70, &hidden, 1}}, 4, 99, 0, 64, 2));
}

namespace kv_advice = executorch::extension::llm::kv_advice;
namespace cb = executorch::backends::cuda::batching;

namespace {

bool any_contains(
    const std::vector<std::string>& lines,
    const std::string& part) {
  return std::any_of(lines.begin(), lines.end(), [&](const std::string& line) {
    return line.find(part) != std::string::npos;
  });
}

constexpr std::int64_t kContext = 131072;
constexpr std::int64_t kTargetBytesPerCell = 53248;
constexpr std::int64_t kDraftBytesPerCell = 2048;

// `owners` x `block` draft blocks over a shared draft pool of `cells` cells.
kv_advice::DraftPool cell_pool(int cells, int owners = 4, int block = 16) {
  return {false, cells, owners, block, kDraftBytesPerCell};
}

// The report CudaExecutor makes for `config`, from the same admission check.
cb::KvCapacityReport report_for(
    const kv_advice::StartupConfig& config,
    cb::KvReservations reservations = {}) {
  const cb::KvAdmissionCheck check = cb::check_admission(
      config.sequence_layout,
      config.sessions,
      config.requested_tokens,
      config.context_length,
      config.cells,
      config.session_limit);
  cb::KvCapacityReport report;
  report.outcome = check.outcome;
  report.sequence_layout = config.sequence_layout;
  report.sessions = config.sessions;
  report.reservations = reservations;
  report.automatic = config.requested_tokens == 0;
  report.session_tokens = static_cast<int>(check.session_tokens);
  report.context_length = config.context_length;
  report.cells = config.cells;
  report.session_limit = config.session_limit;
  report.facts = {kTargetBytesPerCell, /*has_windowed_layers=*/true};
  return report;
}

kv_advice::StartupConfig cell_config(
    int sessions,
    int tokens,
    std::int64_t cells,
    std::optional<kv_advice::DraftPool> draft = std::nullopt) {
  kv_advice::StartupConfig config;
  config.sessions = sessions;
  config.requested_tokens = tokens;
  config.context_length = kContext;
  config.cells = cells;
  config.session_limit = 64;
  config.draft = draft;
  return config;
}

std::optional<kv_advice::Fix> find_fix(
    const kv_advice::Advice& advice,
    kv_advice::FixKind kind) {
  for (const auto& fix : advice.fixes) {
    if (fix.kind == kind) {
      return fix;
    }
  }
  return std::nullopt;
}

} // namespace

TEST(KvAdvice, DraftBudgetFollowsTheExecutorsRule) {
  // (cells - 1 - owners x block) / sessions for a shared pool.
  EXPECT_EQ(kv_advice::draft_session_budget(16, cell_pool(131072)), 8187);
  EXPECT_EQ(kv_advice::draft_session_budget(16, cell_pool(32768)), 2043);
  // Each sequence's rows less the block for a private history.
  EXPECT_EQ(
      kv_advice::draft_session_budget(16, {true, 131072, 4, 16, 0}), 131056);
}

TEST(KvAdvice, DraftAdmissionMatchesTheExecutorsCondition) {
  auto admitted = kv_advice::draft_admission(16, 0, 8191, cell_pool(131072));
  EXPECT_EQ(admitted.max_context, 8187);
  EXPECT_TRUE(admitted.fits);
  EXPECT_TRUE(
      kv_advice::draft_admission(16, 2043, 2043, cell_pool(32768)).fits);
  EXPECT_FALSE(
      kv_advice::draft_admission(16, 2044, 2044, cell_pool(32768)).fits);
  // (336 - 65) / 16 = 16 tokens, the block; (337 - 65) / 16 = 17.
  EXPECT_FALSE(kv_advice::draft_admission(16, 0, 8191, cell_pool(336)).fits);
  EXPECT_TRUE(kv_advice::draft_admission(16, 0, 8191, cell_pool(337)).fits);
  EXPECT_FALSE(kv_advice::draft_admission(16, 0, 16, cell_pool(131072)).fits);
}

TEST(KvAdvice, StartupChecksFollowTheFactoriesAndExecutors) {
  EXPECT_TRUE(kv_advice::check_startup(cell_config(4, 1024, 4097)).admitted);
  EXPECT_FALSE(kv_advice::check_startup(cell_config(4, 1024, 4096)).admitted);
  // One-token sessions are refused, requested or resolved.
  EXPECT_FALSE(kv_advice::check_startup(cell_config(4, 1, 4097)).admitted);
  EXPECT_FALSE(kv_advice::check_startup(cell_config(8, 0, 16)).admitted);
  EXPECT_FALSE(kv_advice::check_startup(cell_config(65, 0, 1 << 20)).admitted);
  const auto dflash =
      kv_advice::check_startup(cell_config(16, 0, 131072, cell_pool(131072)));
  EXPECT_TRUE(dflash.admitted);
  EXPECT_EQ(dflash.session_tokens, 8187);
}

TEST(KvAdvice, DFlashCellShortfallSizesBothPools) {
  // 16 sessions of 131072 tokens on a 131072-cell DFlash export whose draft
  // pool is the default --max-cells.
  const auto config = cell_config(16, 131072, 131072, cell_pool(131072));
  const auto advice =
      kv_advice::advise(report_for(config, {4, 8, 4}), config.draft);
  EXPECT_FALSE(advice.admitted);
  const auto cells = find_fix(advice, kv_advice::FixKind::kCells);
  ASSERT_TRUE(cells);
  EXPECT_EQ(cells->value, 2097153);
  EXPECT_EQ(cells->draft_cells, 2097153 + 64);
  EXPECT_TRUE(any_contains(
      advice.lines, "--max-cells >= 2097153 and --draft-max-cells >= 2097217"));
  EXPECT_TRUE(any_contains(
      advice.lines, "without --draft-max-cells, --max-cells >= 2097217"));
  // Both pools' K/V.
  EXPECT_TRUE(any_contains(
      advice.lines,
      cb::format_memory(
          2097153 * kTargetBytesPerCell + 2097217 * kDraftBytesPerCell)));
  // The token limit fits the draft too and exceeds the block.
  const auto limit = find_fix(advice, kv_advice::FixKind::kTokenLimit);
  ASSERT_TRUE(limit);
  EXPECT_EQ(limit->low, 17);
  EXPECT_EQ(limit->value, 8187);
  EXPECT_TRUE(any_contains(
      advice.lines,
      "set --max_session_tokens/--max-context between 17 and 8187"));
  // No sessions count fits 131072 tokens in 131072 cells.
  EXPECT_FALSE(find_fix(advice, kv_advice::FixKind::kSessions));
  EXPECT_FALSE(find_fix(advice, kv_advice::FixKind::kSequenceLayout));
}

TEST(KvAdvice, AutomaticDraftLimitAsksOnlyForDraftCells) {
  // The target already holds the full context; only the draft limits it.
  const auto config = cell_config(16, 0, 2097153, cell_pool(32768));
  const auto advice = kv_advice::advise(report_for(config), config.draft);
  EXPECT_TRUE(advice.admitted);
  EXPECT_TRUE(any_contains(advice.lines, "each session may use 2043 of"));
  const auto cells = find_fix(advice, kv_advice::FixKind::kCells);
  ASSERT_TRUE(cells);
  EXPECT_EQ(cells->value, 2097153);
  EXPECT_EQ(cells->draft_cells, 2097217);
  EXPECT_TRUE(any_contains(
      advice.lines,
      "full context, re-export with --draft-max-cells >= 2097217"));
  EXPECT_FALSE(any_contains(advice.lines, "--max-cells >= 2097153"));
}

TEST(KvAdvice, SessionLimitWithShortCellsAdvisesSessionsThatFit) {
  const auto config = cell_config(80, 8192, 131072);
  const auto advice = kv_advice::advise(report_for(config, {16, 48, 16}));
  const auto fewer = find_fix(advice, kv_advice::FixKind::kSessions);
  ASSERT_TRUE(fewer);
  EXPECT_EQ(fewer->value, 15);
  EXPECT_TRUE(any_contains(advice.lines, "to total at most 15 resident"));
  EXPECT_TRUE(
      any_contains(advice.lines, "--kv-layout sequence --max-kv-seqs >= 80"));
  EXPECT_FALSE(find_fix(advice, kv_advice::FixKind::kCells));
  EXPECT_FALSE(find_fix(advice, kv_advice::FixKind::kTokenLimit));
}

TEST(KvAdvice, AutomaticSessionLimitNamesBothFlagSpellings) {
  // 32 live + 32 prefix + 4 capture on the worker's automatic context.
  const auto config = cell_config(68, 0, 131072);
  const auto advice = kv_advice::advise(report_for(config, {32, 32, 4}));
  const auto fewer = find_fix(advice, kv_advice::FixKind::kSessions);
  ASSERT_TRUE(fewer);
  EXPECT_EQ(fewer->value, 64);
  for (const char* flag :
       {"--max_sessions/--max-sessions",
        "--prefix_cache_entries/--prefix-cache-entries",
        "--prefix_capture_slots/--prefix-capture-slots"}) {
    EXPECT_TRUE(any_contains(advice.lines, flag)) << flag;
  }
  // Capture slots size themselves unless set explicitly.
  EXPECT_TRUE(any_contains(advice.lines, "default to automatic sizing"));
  EXPECT_FALSE(find_fix(advice, kv_advice::FixKind::kCells));
}

TEST(KvAdvice, DraftShortfallWithARequestedLimitSizesForTheLimit) {
  // The draft leaves (300 - 65) / 16 = 14 tokens; 4096 were requested.
  const auto config = cell_config(16, 4096, 131072, cell_pool(300));
  const auto advice = kv_advice::advise(report_for(config), config.draft);
  EXPECT_FALSE(advice.admitted);
  EXPECT_TRUE(any_contains(advice.lines, "leave 14 tokens for each of 16"));
  // Re-export the draft for 4096 tokens each: 16 x 4096 + 64 + 1.
  const auto cells = find_fix(advice, kv_advice::FixKind::kCells);
  ASSERT_TRUE(cells);
  EXPECT_EQ(cells->value, 131072);
  EXPECT_EQ(cells->draft_cells, 65601);
  // No session count or token limit above the block fits 300 draft cells
  // with a 4096-token limit.
  EXPECT_FALSE(find_fix(advice, kv_advice::FixKind::kSessions));
  EXPECT_FALSE(find_fix(advice, kv_advice::FixKind::kTokenLimit));
}

TEST(KvAdvice, DFlashZeroBudgetNeverAdvisesOneTokenSessions) {
  // Automatic: 15 / 16 = 0 tokens per session.
  const auto config = cell_config(16, 0, 16, cell_pool(131072));
  const auto advice = kv_advice::advise(report_for(config), config.draft);
  const auto cells = find_fix(advice, kv_advice::FixKind::kCells);
  ASSERT_TRUE(cells);
  // Sized for the block plus one token: 16 x 17 + 1.
  EXPECT_EQ(cells->value, 16 * 17 + 1);
  EXPECT_EQ(
      kv_advice::check_startup(kv_advice::apply_fix(config, *cells))
          .session_tokens,
      17);
}

namespace {

// A configuration the advice text lets a user choose; `serving` when only
// serving options change, so the same program applies.
struct TextChoice {
  kv_advice::StartupConfig config;
  bool serving;
};

// Reads the fixes back from the advice text, independently of the structured
// fixes: every figure a line names becomes the configuration it asks for, at
// each end of a range. `primary` counts the lines that state a fix.
std::vector<TextChoice> choices_from_text(
    const kv_advice::StartupConfig& config,
    const std::vector<std::string>& lines,
    int* primary) {
  std::vector<TextChoice> choices;
  *primary = 0;
  std::smatch m;
  const auto number = [&](int i) { return std::stoll(m[i].str()); };
  for (const std::string& line : lines) {
    auto choice = config;
    if (std::regex_search(
            line,
            m,
            std::regex(R"(to total at most (\d+) resident sessions)"))) {
      ++*primary;
      choice.sessions = static_cast<int>(number(1));
      choices.push_back({choice, true});
      choice.sessions = 1;
      choices.push_back({choice, true});
    } else if (
        std::regex_search(
            line,
            m,
            std::regex(
                R"(--max-context (?:to (\d+)|between (\d+) and (\d+)))"))) {
      ++*primary;
      for (const int i : {1, 2, 3}) {
        if (m[i].matched) {
          choice.requested_tokens = static_cast<int>(number(i));
          choices.push_back({choice, true});
        }
      }
    } else if (
        std::regex_search(
            line,
            m,
            std::regex(
                R"(without --draft-max-cells, --max-cells >= (\d+) sizes both)"))) {
      choice.cells = number(1);
      choice.draft->cells = number(1);
      choices.push_back({choice, false});
    } else if (
        std::regex_search(
            line,
            m,
            std::regex(
                R"(re-export with --max-cells >= (\d+) and --draft-max-cells >= (\d+))"))) {
      ++*primary;
      choice.cells = number(1);
      choice.draft->cells = number(2);
      choices.push_back({choice, false});
    } else if (std::regex_search(
                   line,
                   m,
                   std::regex(
                       R"(re-export with --draft-max-cells >= (\d+))"))) {
      ++*primary;
      choice.draft->cells = number(1);
      choices.push_back({choice, false});
    } else if (std::regex_search(
                   line,
                   m,
                   std::regex(R"(re-export with --max-cells >= (\d+))"))) {
      ++*primary;
      choice.cells = number(1);
      choices.push_back({choice, false});
    } else if (std::regex_search(
                   line,
                   m,
                   std::regex(
                       R"(--kv-layout sequence --max-kv-seqs >= (\d+))"))) {
      ++*primary;
      choice.sequence_layout = true;
      choice.session_limit = static_cast<int>(number(1));
      choices.push_back({choice, false});
    } else if (std::regex_search(
                   line,
                   m,
                   std::regex(R"(re-export with --max-kv-seqs >= (\d+))"))) {
      ++*primary;
      choice.session_limit = static_cast<int>(number(1));
      choices.push_back({choice, false});
    } else if (std::regex_search(
                   line,
                   m,
                   std::regex(R"(sequence columns of at least (\d+) rows)"))) {
      ++*primary;
      choice.draft->cells = number(1);
      choices.push_back({choice, false});
    }
  }
  return choices;
}

} // namespace

TEST(KvAdvice, SharedTokenRules) {
  EXPECT_TRUE(kv_advice::valid_requested_tokens(0));
  EXPECT_FALSE(kv_advice::valid_requested_tokens(1));
  EXPECT_TRUE(kv_advice::valid_requested_tokens(2));
  EXPECT_FALSE(kv_advice::holds_prompt_and_prediction(1));
  EXPECT_TRUE(kv_advice::holds_prompt_and_prediction(2));
  EXPECT_EQ(kv_advice::least_session_tokens(std::nullopt), 2);
  EXPECT_EQ(kv_advice::least_session_tokens(cell_pool(1000)), 17);
}

TEST(KvAdvice, StartupChecksModelTheIntLimits) {
  constexpr int kMax = std::numeric_limits<int>::max();
  // sessions x tokens must fit an int, as CudaExecutor's entry check asks.
  auto config = cell_config(64, 1 << 25, kMax);
  config.context_length = std::int64_t{1} << 26;
  EXPECT_FALSE(kv_advice::check_startup(config).admitted);
  // A cell count past an int is not a program CudaExecutor accepts.
  EXPECT_FALSE(
      kv_advice::check_startup(cell_config(4, 1024, std::int64_t{kMax} + 1))
          .admitted);
  // Nor is a draft pool past an int, and the fix that would need one is not
  // offered.
  auto big = cell_config(64, 131072, 131072, cell_pool(131072));
  big.draft->cells = std::int64_t{kMax} + 1;
  EXPECT_FALSE(kv_advice::check_startup(big).admitted);
  const auto huge = cell_config(64, 131072, 131072, cell_pool(131072));
  const auto advice = kv_advice::advise(report_for(huge), huge.draft);
  // 64 x 131072 + 1 cells still fit an int, so that fix closes.
  EXPECT_TRUE(find_fix(advice, kv_advice::FixKind::kCells));
  auto wide = cell_config(64, 65536, 65536);
  wide.context_length = 1 << 26;
  wide.requested_tokens = 1 << 25;
  const auto none = kv_advice::advise(report_for(wide));
  // 64 x 2^25 + 1 cells would not fit an int: no cell re-export is named.
  EXPECT_FALSE(find_fix(none, kv_advice::FixKind::kCells));
}

TEST(KvAdviceClosure, EveryFixPassesEveryStartupCheck) {
  struct Split {
    int live, prefix;
  };
  int configs = 0;
  int refused = 0;
  int fixed = 0;
  int unfixable = 0;
  for (const bool sequence : {false, true}) {
    for (const bool dflash : {false, true}) {
      for (const int block :
           dflash ? std::vector<int>{2, 16} : std::vector<int>{0}) {
        const std::vector<int> draft_cells = !dflash ? std::vector<int>{0}
            : sequence ? std::vector<int>{2048, 131088}
                       : std::vector<int>{300, 32768, 131072, 2097217};
        const std::vector<std::int64_t> cells_options = sequence
            ? std::vector<std::int64_t>{0}
            : std::vector<std::int64_t>{16, 4096, 131072, 2097153};
        const std::vector<int> limits =
            sequence ? std::vector<int>{4, 16, 64, 128} : std::vector<int>{64};
        for (const int sessions : {1, 7, 16, 63, 64, 65, 80}) {
          for (const int tokens : {0, 17, 4096, 131072}) {
            for (const std::int64_t cells : cells_options) {
              for (const int limit : limits) {
                for (const int draft : draft_cells) {
                  for (const Split split :
                       {Split{sessions, 0},
                        Split{sessions / 2, sessions / 4}}) {
                    kv_advice::StartupConfig config;
                    config.sequence_layout = sequence;
                    config.sessions = sessions;
                    config.requested_tokens = tokens;
                    config.context_length = kContext;
                    config.cells = cells;
                    config.session_limit = limit;
                    if (dflash) {
                      config.draft = kv_advice::DraftPool{
                          sequence, draft, 4, block, kDraftBytesPerCell};
                    }
                    const cb::KvReservations reservations{
                        split.live,
                        split.prefix,
                        sessions - split.live - split.prefix};
                    SCOPED_TRACE(
                        "sequence=" + std::to_string(sequence) +
                        " dflash=" + std::to_string(dflash) +
                        " block=" + std::to_string(block) +
                        " sessions=" + std::to_string(sessions) +
                        " tokens=" + std::to_string(tokens) +
                        " cells=" + std::to_string(cells) +
                        " limit=" + std::to_string(limit) +
                        " draft=" + std::to_string(draft));
                    ++configs;
                    const auto base = kv_advice::check_startup(config);
                    const auto advice = kv_advice::advise(
                        report_for(config, reservations), config.draft);
                    ASSERT_EQ(advice.admitted, base.admitted);
                    const bool full_context = base.admitted && tokens == 0 &&
                        base.session_tokens < kContext;
                    refused += !base.admitted;
                    if (!base.admitted && advice.fixes.empty()) {
                      ++unfixable;
                      EXPECT_TRUE(any_contains(advice.lines, "no single"));
                    }
                    fixed += !base.admitted && !advice.fixes.empty();
                    for (const auto& fix : advice.fixes) {
                      SCOPED_TRACE(static_cast<int>(fix.kind));
                      // Applied alone, at its least favorable value, the fix
                      // passes every startup check.
                      const auto after = kv_advice::check_startup(
                          kv_advice::apply_fix(config, fix));
                      EXPECT_TRUE(after.admitted);
                      if (full_context) {
                        EXPECT_EQ(after.session_tokens, kContext);
                      }
                      // The advice names the fix's figure: the draft cells
                      // when the target's cells already suffice.
                      const std::int64_t shown =
                          fix.kind == kv_advice::FixKind::kCells && dflash &&
                              fix.value == config.cells
                          ? fix.draft_cells
                          : fix.value;
                      EXPECT_TRUE(
                          any_contains(advice.lines, std::to_string(shown)));
                      switch (fix.kind) {
                        case kv_advice::FixKind::kSessions: {
                          auto one = config;
                          one.sessions = 1;
                          EXPECT_TRUE(kv_advice::check_startup(one).admitted);
                          break;
                        }
                        case kv_advice::FixKind::kTokenLimit: {
                          EXPECT_GT(fix.low, dflash ? block : 1);
                          auto low = config;
                          low.requested_tokens = static_cast<int>(fix.low);
                          EXPECT_TRUE(kv_advice::check_startup(low).admitted);
                          break;
                        }
                        case kv_advice::FixKind::kCells:
                          if (dflash) {
                            // The default draft pool: --max-cells alone.
                            auto both = config;
                            both.cells = std::max(fix.value, fix.draft_cells);
                            both.draft->cells = both.cells;
                            EXPECT_TRUE(
                                kv_advice::check_startup(both).admitted);
                          }
                          break;
                        case kv_advice::FixKind::kSequenceLayout:
                          EXPECT_FALSE(dflash);
                          break;
                        default:
                          break;
                      }
                    }
                    for (const auto& line : advice.lines) {
                      EXPECT_LT(line.size(), 256u) << line;
                    }
                    // Every offered fix is a candidate, and every candidate
                    // left out fails some choice it would allow.
                    const auto candidates = kv_advice::candidate_fixes(
                        report_for(config, reservations), config.draft);
                    for (const auto& candidate : candidates.fixes) {
                      const bool offered = std::any_of(
                          advice.fixes.begin(),
                          advice.fixes.end(),
                          [&](const kv_advice::Fix& fix) {
                            return fix.kind == candidate.kind &&
                                fix.value == candidate.value;
                          });
                      if (offered) {
                        continue;
                      }
                      bool fails =
                          candidate.kind == kv_advice::FixKind::kTokenLimit &&
                          candidate.value < candidate.low;
                      for (const auto& choice :
                           kv_advice::fix_choices(config, candidate)) {
                        const auto after = kv_advice::check_startup(choice);
                        fails = fails || !after.admitted ||
                            (full_context && after.session_tokens != kContext);
                      }
                      EXPECT_TRUE(fails) << static_cast<int>(candidate.kind);
                    }
                    // What the text asks for, read back from the text alone,
                    // passes every capacity check.
                    int primary = 0;
                    for (const auto& choice :
                         choices_from_text(config, advice.lines, &primary)) {
                      const auto after =
                          kv_advice::check_startup(choice.config);
                      EXPECT_TRUE(after.admitted);
                      if (full_context) {
                        EXPECT_EQ(after.session_tokens, kContext);
                      }
                    }
                    EXPECT_EQ(primary, static_cast<int>(advice.fixes.size()));
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  // The grid reaches refusals of every kind, and most have a closed fix.
  EXPECT_GT(refused, 500);
  EXPECT_GT(fixed, refused / 2);
  std::cout << "advice closure: " << configs << " configs, " << refused
            << " refused, " << fixed << " with closed fixes, " << unfixable
            << " described only\n";
}

// The real CudaExecutor admits what the advice tells a user to change, on the
// backend's toy program (test/export_toy_decoder.py: 256 cells, a 64-token
// context). Only serving options apply to the same program.
TEST(KvAdviceToy, AdvisedServingFixesAdmitTheToyProgram) {
  const char* dir = std::getenv("ET_CUDA_BATCHING_TOY_DIR");
  if (!dir || !*dir) {
    GTEST_SKIP() << "Set ET_CUDA_BATCHING_TOY_DIR to the output of "
                    "backends/cuda/batching/test/export_toy_decoder.py";
  }
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    GTEST_SKIP() << "CUDA device required";
  }
  executorch::runtime::runtime_init();
  const std::string toy = std::string(dir) + "/dense";
  const auto create = [&](int sessions,
                          int tokens,
                          std::optional<cb::KvCapacityReport>* report) {
    cb::CudaExecutorConfig config;
    config.model_path = toy + "/model.pte";
    config.data_files = {toy + "/aoti_cuda_blob.ptd"};
    config.load_mode = executorch::extension::Module::LoadMode::File;
    config.max_sessions = sessions;
    config.max_session_tokens = tokens;
    config.initial_capacity = 16;
    config.options.cuda_graph_for_static_methods = false;
    config.capacity_report = [report](const cb::KvCapacityReport& r) {
      *report = r;
    };
    return cb::CudaExecutor::create(config).error();
  };
  struct Case {
    int sessions, tokens;
  };
  // 8 x 64 tokens on 256 cells; 65 sessions of 3 tokens; 65 and 256
  // automatic sessions, past the cell layout's 64.
  for (const Case c : {Case{8, 64}, Case{65, 3}, Case{65, 0}, Case{256, 0}}) {
    SCOPED_TRACE(std::to_string(c.sessions) + " x " + std::to_string(c.tokens));
    std::optional<cb::KvCapacityReport> report;
    ASSERT_EQ(
        create(c.sessions, c.tokens, &report),
        executorch::runtime::Error::InvalidArgument);
    ASSERT_TRUE(report);
    const auto advice = kv_advice::advise(*report);
    int primary = 0;
    int applied = 0;
    for (const auto& choice : choices_from_text(
             kv_advice::startup_config(*report), advice.lines, &primary)) {
      if (!choice.serving) {
        continue;
      }
      std::optional<cb::KvCapacityReport> after;
      EXPECT_EQ(
          create(
              choice.config.sessions, choice.config.requested_tokens, &after),
          executorch::runtime::Error::Ok)
          << choice.config.sessions << " x " << choice.config.requested_tokens;
      ++applied;
    }
    EXPECT_GT(applied, 0);
  }
}
