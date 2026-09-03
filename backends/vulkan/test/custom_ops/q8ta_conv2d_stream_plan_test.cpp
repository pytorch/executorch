// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q8taConv2d.h>

#include <gtest/gtest.h>

#include <limits>

namespace vkcompute {
namespace {

constexpr int64_t kEightMiB = 8 * 1024 * 1024;
constexpr int64_t kSixteenMiB = 16 * 1024 * 1024;

TEST(Q8taConv2dStreamPlanTest, SplitsFirstSceneXConvolution) {
  const auto plan = make_q8ta_conv2d_stream_plan(
      /*batch=*/60,
      /*flattened_kernel_size=*/576,
      /*out_height=*/20,
      /*out_width=*/26,
      kEightMiB);

  EXPECT_TRUE(plan.feasible);
  EXPECT_EQ(plan.aligned_out_width, 28);
  EXPECT_EQ(plan.rows_per_tile, 400);
  EXPECT_EQ(plan.num_tiles, 3);
  EXPECT_EQ(plan.scratch_bytes, 6451200);
}

TEST(Q8taConv2dStreamPlanTest, SplitsSecondSceneXConvolution) {
  const auto plan = make_q8ta_conv2d_stream_plan(
      /*batch=*/60,
      /*flattened_kernel_size=*/1152,
      /*out_height=*/10,
      /*out_width=*/13,
      kEightMiB);

  EXPECT_TRUE(plan.feasible);
  EXPECT_EQ(plan.aligned_out_width, 16);
  EXPECT_EQ(plan.rows_per_tile, 300);
  EXPECT_EQ(plan.num_tiles, 2);
  EXPECT_EQ(plan.scratch_bytes, 5529600);
}

TEST(Q8taConv2dStreamPlanTest, UsesOneTileWhenFullScratchFits) {
  const auto plan = make_q8ta_conv2d_stream_plan(
      /*batch=*/2,
      /*flattened_kernel_size=*/288,
      /*out_height=*/7,
      /*out_width=*/7,
      kEightMiB);

  EXPECT_TRUE(plan.feasible);
  EXPECT_EQ(plan.rows_per_tile, 14);
  EXPECT_EQ(plan.num_tiles, 1);
  EXPECT_EQ(plan.scratch_bytes, 32256);
}

TEST(Q8taConv2dStreamPlanTest, SelectsFullFitBelowProductionBudget) {
  const auto plan = make_q8ta_conv2d_stream_plan(
      /*batch=*/10,
      /*flattened_kernel_size=*/288,
      /*out_height=*/30,
      /*out_width=*/99,
      kSixteenMiB);

  EXPECT_TRUE(plan.feasible);
  EXPECT_EQ(plan.rows_per_tile, 300);
  EXPECT_EQ(plan.num_tiles, 1);
  EXPECT_EQ(plan.scratch_bytes, 8640000);
}

TEST(Q8taConv2dStreamPlanTest, SelectsStreamingAboveProductionBudget) {
  const auto plan = make_q8ta_conv2d_stream_plan(
      /*batch=*/10,
      /*flattened_kernel_size=*/576,
      /*out_height=*/30,
      /*out_width=*/99,
      kSixteenMiB);

  EXPECT_TRUE(plan.feasible);
  EXPECT_EQ(plan.rows_per_tile, 150);
  EXPECT_EQ(plan.num_tiles, 2);
  EXPECT_EQ(plan.scratch_bytes, 8640000);
}

TEST(Q8taConv2dStreamPlanTest, RejectsOneRowLargerThanBudget) {
  const auto plan = make_q8ta_conv2d_stream_plan(
      /*batch=*/1,
      /*flattened_kernel_size=*/16384,
      /*out_height=*/1,
      /*out_width=*/513,
      kEightMiB);

  EXPECT_FALSE(plan.feasible);
  EXPECT_EQ(plan.rows_per_tile, 0);
  EXPECT_EQ(plan.num_tiles, 0);
  EXPECT_EQ(plan.scratch_bytes, 0);
}

TEST(Q8taConv2dStreamPlanTest, RejectsOutputWidthAlignmentOverflow) {
  const auto plan = make_q8ta_conv2d_stream_plan(
      /*batch=*/1,
      /*flattened_kernel_size=*/1,
      /*out_height=*/1,
      /*out_width=*/std::numeric_limits<int64_t>::max(),
      std::numeric_limits<int64_t>::max());

  EXPECT_FALSE(plan.feasible);
}

TEST(Q8taConv2dStreamPlanTest, HandlesTileCountCeilDivisionAtShaderLimit) {
  const auto plan = make_q8ta_conv2d_stream_plan(
      /*batch=*/std::numeric_limits<int32_t>::max(),
      /*flattened_kernel_size=*/1,
      /*out_height=*/1,
      /*out_width=*/1,
      /*scratch_budget_bytes=*/8);

  EXPECT_TRUE(plan.feasible);
  EXPECT_EQ(plan.rows_per_tile, 2);
  EXPECT_EQ(plan.num_tiles, 1073741824);
  EXPECT_EQ(plan.scratch_bytes, 8);
}

TEST(Q8taConv2dStreamPlanTest, RejectsRowOffsetBeyondShaderIndexRange) {
  const auto plan = make_q8ta_conv2d_stream_plan(
      /*batch=*/std::numeric_limits<int64_t>::max(),
      /*flattened_kernel_size=*/1,
      /*out_height=*/1,
      /*out_width=*/1,
      std::numeric_limits<int64_t>::max());

  EXPECT_FALSE(plan.feasible);
}

} // namespace
} // namespace vkcompute
