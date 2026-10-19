/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <cuda_runtime.h>
#include <executorch/examples/models/muse-glimmer/runtime/cuda_dflash_executor.h>
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

TEST(CudaDFlash, NewPreambleAnchorIsRetainedOnce) {
  // The input token (7) has already committed. The regular preamble predicts
  // 10, then verification consumes 10 and predicts the accepted/corrected tail.
  for (const auto& predictions : std::vector<std::vector<Token>>{
           {21, 12, 13, 14}, {11, 22, 13, 14}, {11, 12, 13, 14}}) {
    const auto accepted = static_cast<std::int64_t>(
        std::mismatch(predictions.begin(), predictions.begin() + 3,
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
      [](void* data) { cudaFree(data); });
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
