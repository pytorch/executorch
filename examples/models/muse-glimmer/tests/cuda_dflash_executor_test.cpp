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
#include <gtest/gtest.h>
#include <algorithm>
#include <array>

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

// 4 owners x 16-token blocks over a shared draft pool of `cells` cells.
kv_advice::DraftPool cell_pool(int cells) {
  return {/*sequence_layout=*/false, cells, /*max_owners=*/4, /*block=*/16};
}

cb::KvCapacityReport target_report(
    cb::KvAdmission outcome,
    int sessions,
    int tokens,
    std::int64_t cells) {
  cb::KvCapacityReport report;
  report.outcome = outcome;
  report.sessions = sessions;
  report.reservations = {4, 8, sessions - 12};
  report.session_tokens = tokens;
  report.context_length = 131072;
  report.cells = cells;
  report.session_limit = 64;
  report.facts = {53248, /*has_windowed_layers=*/true};
  return report;
}

} // namespace

TEST(KvAdvice, DraftBudgetFollowsTheExecutorsRule) {
  // (cells - 1 - owners x block) / sessions for a shared pool.
  EXPECT_EQ(kv_advice::draft_session_budget(16, cell_pool(131072)), 8187);
  EXPECT_EQ(kv_advice::draft_session_budget(16, cell_pool(32768)), 2043);
  // Each sequence's rows less the block for a private history.
  EXPECT_EQ(kv_advice::draft_session_budget(16, {true, 131072, 4, 16}), 131056);
}

TEST(KvAdvice, DraftAdmissionMatchesTheExecutorsCondition) {
  // Automatic: the smaller pool's budget, which must exceed the block.
  auto admitted = kv_advice::draft_admission(16, 0, 8191, cell_pool(131072));
  EXPECT_EQ(admitted.max_context, 8187);
  EXPECT_TRUE(admitted.fits);
  admitted = kv_advice::draft_admission(16, 0, 8191, cell_pool(32768));
  EXPECT_EQ(admitted.max_context, 2043);
  EXPECT_TRUE(admitted.fits);
  // A requested limit must fit the draft exactly.
  EXPECT_TRUE(
      kv_advice::draft_admission(16, 2043, 2043, cell_pool(32768)).fits);
  EXPECT_FALSE(
      kv_advice::draft_admission(16, 2044, 2044, cell_pool(32768)).fits);
  // Neither pool may leave the block or less.
  EXPECT_FALSE(kv_advice::draft_admission(16, 0, 16, cell_pool(131072)).fits);
  // (336 - 65) / 16 = 16 tokens, the block; (337 - 65) / 16 = 17.
  EXPECT_FALSE(kv_advice::draft_admission(16, 0, 8191, cell_pool(336)).fits);
  EXPECT_TRUE(kv_advice::draft_admission(16, 0, 8191, cell_pool(337)).fits);
}

TEST(KvAdvice, DraftShortfallForARequestedLimit) {
  const auto lines = kv_advice::draft_shortfall_lines(
      16, 8191, 8191, cell_pool(131072), {4, 8, 4});
  ASSERT_EQ(lines.size(), 2u);
  EXPECT_EQ(
      lines[0],
      "DFlash: 16 resident sessions (4 live + 8 prefix entries + 4 capture "
      "slots) x 8191 + 64 temporary draft-block cells + 1 padding = 131121 "
      "draft cells needed, 131072 available");
  EXPECT_EQ(
      lines[1],
      "Muse Glimmer: set --max_session_tokens (serve_batching: --max-context) "
      "to at most 8187, or re-export with --draft-max-cells >= 131121");
}

TEST(KvAdvice, DraftShortfallWhenTheDraftLeavesTheBlockOrLess) {
  // 300 cells leave (300 - 1 - 64) / 16 = 14 tokens, below the 16-token block.
  const auto lines =
      kv_advice::draft_shortfall_lines(16, 0, 8191, cell_pool(300), {});
  ASSERT_EQ(lines.size(), 2u);
  EXPECT_EQ(
      lines[0],
      "DFlash: 300 draft cells, less 64 temporary block cells and 1 padding, "
      "leave 14 tokens for each of 16 resident sessions, which must exceed "
      "the 16-token block");
  // floor(235 / 17) = 13 sessions of 17 tokens; 16 x 17 + 64 + 1 = 337.
  EXPECT_EQ(
      lines[1],
      "Muse Glimmer: reserve at most 13 sessions, or re-export with "
      "--draft-max-cells >= 337");
  EXPECT_FALSE(any_contains(lines, "--max_session_tokens"));
}

TEST(KvAdvice, DraftShortfallWhenTheTargetBudgetIsTooSmall) {
  auto lines =
      kv_advice::draft_shortfall_lines(4, 16, 16, cell_pool(131072), {});
  EXPECT_EQ(
      lines[0],
      "DFlash: a 16-token session budget must exceed the 16-token draft "
      "block");
  EXPECT_TRUE(any_contains(lines, "raise --max_session_tokens"));
  lines = kv_advice::draft_shortfall_lines(4, 0, 12, cell_pool(131072), {});
  EXPECT_TRUE(any_contains(lines, "larger --max-cells"));
}

TEST(KvAdvice, DraftShortfallOnASequenceLayout) {
  const auto lines =
      kv_advice::draft_shortfall_lines(4, 4096, 4096, {true, 2048, 4, 16}, {});
  ASSERT_EQ(lines.size(), 2u);
  EXPECT_EQ(
      lines[0],
      "DFlash: 4096 tokens per session exceed each draft sequence's 2048 rows "
      "less the 16-token block");
  EXPECT_EQ(
      lines[1],
      "Muse Glimmer: set --max_session_tokens (serve_batching: --max-context) "
      "to at most 2032");
}

TEST(KvAdvice, CellShortfallAdviceNamesTheFlags) {
  const auto lines = kv_advice::serving_advice(
      target_report(cb::KvAdmission::kCells, 16, 131072, 131072));
  ASSERT_EQ(lines.size(), 3u);
  EXPECT_EQ(
      lines[0],
      "Muse Glimmer: set --max_session_tokens (serve_batching: --max-context) "
      "to at most 8191, or lower --max_sessions, --prefix_cache_entries or "
      "--prefix_capture_slots");
  EXPECT_EQ(
      lines[1],
      "Muse Glimmer: or re-export with --max-cells >= 2097153 (104.0 GiB of "
      "K/V)");
  EXPECT_EQ(
      lines[2],
      "Muse Glimmer: for long contexts, re-export with --kv-layout sequence "
      "--max-kv-seqs >= 16");
}

TEST(KvAdvice, TargetAdviceIsCappedByTheDraftPool) {
  // The default draft pool (--max-cells) serves 8187, not the target's 8191.
  const auto report =
      target_report(cb::KvAdmission::kCells, 16, 131072, 131072);
  const auto lines = kv_advice::serving_advice(
      report, kv_advice::draft_session_budget(16, cell_pool(131072)));
  EXPECT_TRUE(any_contains(lines, "to at most 8187,")) << lines[0];
  EXPECT_FALSE(any_contains(lines, "8191"));
}

TEST(KvAdvice, SessionLimitAdviceNeverSuggestsATokenLimit) {
  auto report = target_report(cb::KvAdmission::kSessionLimit, 80, 8192, 131072);
  report.reservations = {16, 48, 16};
  const auto lines = kv_advice::serving_advice(report);
  ASSERT_EQ(lines.size(), 1u);
  EXPECT_EQ(
      lines[0],
      "Muse Glimmer: lower --max_sessions, --prefix_cache_entries or "
      "--prefix_capture_slots to total at most 64, or re-export with "
      "--kv-layout sequence --max-kv-seqs >= 80");
  report.sequence_layout = true;
  report.session_limit = 4;
  EXPECT_TRUE(any_contains(
      kv_advice::serving_advice(report), "re-export with --max-kv-seqs >= 80"));
}

TEST(KvAdvice, ZeroTokenBudgetAdviceFitsTheSessions) {
  cb::KvCapacityReport report =
      target_report(cb::KvAdmission::kCells, 16, 4, 8);
  report.sessions = 8;
  report.reservations = {8, 0, 0};
  EXPECT_EQ(
      kv_advice::serving_advice(report)[0],
      "Muse Glimmer: lower --max_sessions, --prefix_cache_entries or "
      "--prefix_capture_slots to total at most 1");
}

TEST(KvAdvice, AutomaticContextExplainsTheFullContext) {
  const auto report =
      target_report(cb::KvAdmission::kAutomatic, 16, 8191, 131072);
  const auto lines = kv_advice::serving_advice(report);
  ASSERT_EQ(lines.size(), 3u);
  EXPECT_EQ(
      lines[0],
      "Muse Glimmer: each session may use 8191 of the model's 131072 context "
      "tokens; fewer --max_sessions, --prefix_cache_entries or "
      "--prefix_capture_slots leave more");
  EXPECT_EQ(
      lines[1],
      "Muse Glimmer: the full context needs an export with --max-cells >= "
      "2097153 (104.0 GiB of K/V)");
  EXPECT_TRUE(any_contains(lines, "--kv-layout sequence --max-kv-seqs >= 16"));
  // With the draft pool the session gets less.
  EXPECT_TRUE(any_contains(
      kv_advice::serving_advice(report, 2043), "each session may use 2043 "));
  // A budget that already covers the context needs no advice.
  EXPECT_TRUE(
      kv_advice::serving_advice(
          target_report(cb::KvAdmission::kAutomatic, 16, 131072, 131072 * 17))
          .empty());
}

TEST(KvAdvice, NoBudgetAdvice) {
  EXPECT_EQ(
      kv_advice::serving_advice(
          target_report(cb::KvAdmission::kNoBudget, 16, 0, 8))[0],
      "Muse Glimmer: lower --max_sessions, --prefix_cache_entries or "
      "--prefix_capture_slots to total at most 7, or re-export with a larger "
      "--max-cells");
}
