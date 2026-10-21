/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/examples/models/muse-glimmer/runtime/cuda_dflash_executor.h>
#include <gtest/gtest.h>
#include <algorithm>

namespace df = executorch::extension::llm::cuda_dflash_detail;
using executorch::extension::llm::batching::Token;

TEST(CudaDFlash, FirstMismatchReturnsOnlyCorrection) {
  const auto burst = df::verify_greedy({11, 12, 13}, {21, 12, 13, 14});
  ASSERT_TRUE(burst);
  EXPECT_EQ(burst->tokens, std::vector<Token>({21}));
  EXPECT_EQ(burst->accepted, 0);
  EXPECT_FALSE(burst->bonus);
}

TEST(CudaDFlash, PartialAcceptanceStopsAtFirstMismatch) {
  const auto burst = df::verify_greedy({11, 12, 13}, {11, 22, 13, 14});
  ASSERT_TRUE(burst);
  EXPECT_EQ(burst->tokens, std::vector<Token>({11, 22}));
  EXPECT_EQ(burst->accepted, 1);
  EXPECT_FALSE(burst->bonus);
}

TEST(CudaDFlash, FullAcceptanceAddsOneBonus) {
  const auto burst = df::verify_greedy({11, 12, 13}, {11, 12, 13, 14});
  ASSERT_TRUE(burst);
  EXPECT_EQ(burst->tokens, std::vector<Token>({11, 12, 13, 14}));
  EXPECT_EQ(burst->accepted, 3);
  EXPECT_TRUE(burst->bonus);
}

TEST(CudaDFlash, InvalidVerificationShapeRejected) {
  EXPECT_FALSE(df::verify_greedy({}, {1}));
  EXPECT_FALSE(df::verify_greedy({1}, {}));
  EXPECT_FALSE(df::verify_greedy({1, 2}, {1, 2}));
  EXPECT_FALSE(df::verify_greedy({1}, {1, 2, 3}));
}

TEST(CudaDFlash, NewPreambleAnchorIsRetainedOnce) {
  // The input token (7) has already committed. The regular preamble predicts
  // 10, then verification consumes 10 and predicts the accepted/corrected tail.
  for (const auto& predictions : std::vector<std::vector<Token>>{
           {21, 12, 13, 14}, {11, 22, 13, 14}, {11, 12, 13, 14}}) {
    auto burst = df::verify_greedy({11, 12, 13}, predictions);
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
  const auto burst = df::verify_greedy({11, 12, 13}, {11, 12, 13, 14});
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

TEST(CudaDFlash, PackingRejectsDuplicateOwnersAndOversizedContext) {
  std::vector<std::uint8_t> hidden{1, 2};
  EXPECT_FALSE(df::pack_draft(
      {{4, 0, 70, &hidden, 1}, {4, 0, 80, &hidden, 1}}, 4, 99, 2, 64, 2));
  EXPECT_FALSE(df::pack_draft({{4, 0, 70, &hidden, 2}}, 4, 99, 2, 64, 2));
  EXPECT_FALSE(df::pack_draft({{4, 0, 70, &hidden, 1}}, 4, 99, 0, 64, 2));
}
