/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Device-free unit tests for pure WebGPU utility math. The shared utility
// header also exposes device-taking helpers, but these tests do not call them.

#include <executorch/backends/webgpu/runtime/WebGPUUtils.h>
#include <executorch/backends/webgpu/runtime/ops/argmax/arg_reduce_multiwg_route.h>
#include <executorch/backends/webgpu/runtime/ops/sdpa_fd_decode/SdpaFdDecode.h>
#include <executorch/backends/webgpu/runtime/ops/sdpa_fd_decode/sdpa_fd_split_gqa2_f16_wgsl.h>

#include <gtest/gtest.h>

#include <string_view>

#include <limits>

using namespace executorch::backends::webgpu;

TEST(WebGPUUtils, DivUpDoesNotOverflowAtUint32Max) {
  constexpr uint32_t kMax = std::numeric_limits<uint32_t>::max();
  EXPECT_EQ(utils::div_up(kMax, 4u), 1073741824u);
  EXPECT_EQ(utils::div_up(kMax, kMax), 1u);
}

TEST(WebGPUUtils, DispatchGridStaysOneDimUnderCeiling) {
  utils::DispatchGrid g =
      utils::compute_dispatch_grid_from_limits(1000u, 256u, 65535u, "test");
  EXPECT_EQ(g.count_x, 1000u);
  EXPECT_EQ(g.count_y, 1u);
  EXPECT_EQ(g.stride_x, g.count_x * g.wg_size);
}

TEST(WebGPUUtils, DispatchGridPastCeilingIsNearSquareNotMaxedOut) {
  // total=65536, max_dim=65535: one workgroup past the 1D ceiling.
  // The old {max_dim, div_up(total,max_dim)} fold gave (65535, 2) = 131070
  // launched workgroups for 65536 needed (~100% overhead). A near-square
  // grid should launch close to the needed count instead.
  const uint32_t total = 65536u;
  const uint32_t max_dim = 65535u;
  utils::DispatchGrid g =
      utils::compute_dispatch_grid_from_limits(total, 256u, max_dim, "test");

  EXPECT_LE(static_cast<uint64_t>(g.count_x) * g.count_y, total + total / 10)
      << "near-square grid should launch within ~10% of the needed count, "
         "not ~2x it";
  // Grid must still cover every needed workgroup.
  EXPECT_GE(static_cast<uint64_t>(g.count_x) * g.count_y, total);
  // Not the old maxed-out-count_x behavior.
  EXPECT_NE(g.count_x, max_dim);
  EXPECT_EQ(g.stride_x, g.count_x * g.wg_size);
}

TEST(WebGPUUtils, DispatchGridExactSquareCase) {
  // total=65536 factors exactly as 256*256 — the near-square grid should
  // find this with zero waste.
  utils::DispatchGrid g =
      utils::compute_dispatch_grid_from_limits(65536u, 1u, 65535u, "test");
  EXPECT_EQ(g.count_x, 256u);
  EXPECT_EQ(g.count_y, 256u);
  EXPECT_EQ(static_cast<uint64_t>(g.count_x) * g.count_y, 65536u);
}

TEST(WebGPUUtils, DispatchGridThrowsPastCapacity) {
  // total > max_dim^2: even a near-square grid can't fit in the 2D ceiling.
  const uint32_t max_dim = 4u;
  EXPECT_THROW(
      utils::compute_dispatch_grid_from_limits(
          static_cast<uint32_t>(max_dim) * max_dim + 1u, 1u, max_dim, "test"),
      std::runtime_error);
}

TEST(WebGPUUtils, RowChunkingKeepsFittingRowsTogether) {
  const utils::RowChunking chunking =
      utils::compute_row_chunking(1024u, 256u, 10u, "test");
  EXPECT_EQ(chunking.rows_per_chunk, 4u);
  EXPECT_EQ(chunking.num_chunks, 3u);
}

TEST(WebGPUUtils, BindingSpecCarriesExplicitBufferOffset) {
  const utils::BindingSpec binding = {
      3u, WGPUBufferBindingType_Storage, nullptr, 64u, 256u};
  const WGPUBindGroupEntry entry = utils::make_bind_group_entry(binding);
  EXPECT_EQ(entry.binding, 3u);
  EXPECT_EQ(entry.buffer, nullptr);
  EXPECT_EQ(entry.size, 64u);
  EXPECT_EQ(entry.offset, 256u);
}

TEST(WebGPUUtils, BindingSpecDefaultsBufferOffsetToZero) {
  const utils::BindingSpec binding = {
      3u, WGPUBufferBindingType_Storage, nullptr, 64u};
  const WGPUBindGroupEntry entry = utils::make_bind_group_entry(binding);
  EXPECT_EQ(entry.offset, 0u);
}

TEST(WebGPUUtils, RowChunkingUsesOneChunkWhenAllRowsFit) {
  const utils::RowChunking chunking =
      utils::compute_row_chunking(1024u, 16u, 8u, "test");
  EXPECT_EQ(chunking.rows_per_chunk, 8u);
  EXPECT_EQ(chunking.num_chunks, 1u);
}

TEST(WebGPUUtils, RowChunkingRejectsInvalidArguments) {
  EXPECT_THROW(
      utils::compute_row_chunking(0u, 1u, 1u, "test"), std::runtime_error);
  EXPECT_THROW(
      utils::compute_row_chunking(1u, 0u, 1u, "test"), std::runtime_error);
  EXPECT_THROW(
      utils::compute_row_chunking(1u, 1u, 0u, "test"), std::runtime_error);
  EXPECT_THROW(
      utils::compute_row_chunking(1u, 2u, 1u, "test"), std::runtime_error);
}

TEST(WebGPUUtils, RowChunkingRejectsChunkCountsAboveUint32) {
  EXPECT_THROW(
      utils::compute_row_chunking(
          1u,
          1u,
          static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1u,
          "test"),
      std::runtime_error);
}

TEST(WebGPUUtils, ArgReduceRouteKeepsShortRowsOnGenericKernel) {
  EXPECT_EQ(select_arg_reduce_parts(1u, 4095u, 65535u), 0u);
  EXPECT_EQ(select_arg_reduce_parts(0u, 262144u, 65535u), 0u);
}

TEST(WebGPUUtils, ArgReduceRouteSelectsLongVocabularyRows) {
  EXPECT_EQ(select_arg_reduce_parts(1u, 4096u, 65535u), 4u);
  EXPECT_EQ(select_arg_reduce_parts(1u, 262144u, 65535u), 256u);
  EXPECT_EQ(
      select_arg_reduce_parts(1u, std::numeric_limits<uint32_t>::max(), 65535u),
      256u);
}

TEST(WebGPUUtils, ArgReduceRouteFailsClosedWhenScratchOrGridWouldOverflow) {
  EXPECT_EQ(select_arg_reduce_parts(4097u, 262144u, 65535u), 0u);
  EXPECT_EQ(select_arg_reduce_parts(1u, 262144u, 1u), 0u);
}

TEST(WebGPUUtils, ArgReduceResizeRejectsScratchGrowth) {
  EXPECT_TRUE(arg_reduce_partial_slots_fit(256u, 1u, 256u));
  EXPECT_TRUE(arg_reduce_partial_slots_fit(256u, 256u, 1u));
  EXPECT_FALSE(arg_reduce_partial_slots_fit(256u, 257u, 1u));
  EXPECT_FALSE(arg_reduce_partial_slots_fit(256u, 129u, 2u));
}

TEST(WebGPUUtils, QwenGqa2FdRouteRequiresEveryExactPredicate) {
  EXPECT_TRUE(is_qwen_gqa2_f16_fd_route(true, 16, 8, 128, 2));
  EXPECT_FALSE(is_qwen_gqa2_f16_fd_route(false, 16, 8, 128, 2));
  EXPECT_FALSE(is_qwen_gqa2_f16_fd_route(true, 15, 8, 128, 2));
  EXPECT_FALSE(is_qwen_gqa2_f16_fd_route(true, 16, 7, 128, 2));
  EXPECT_FALSE(is_qwen_gqa2_f16_fd_route(true, 16, 8, 64, 2));
  EXPECT_FALSE(is_qwen_gqa2_f16_fd_route(true, 16, 8, 128, 1));
}

TEST(WebGPUUtils, QwenGqa2FdRouteUsesTile128Schedule) {
  EXPECT_EQ(sdpa_fd_split_tile(true), 128u);
  EXPECT_EQ(sdpa_fd_num_splits(1u, true), 1u);
  EXPECT_EQ(sdpa_fd_num_splits(129u, true), 2u);
  EXPECT_EQ(sdpa_fd_num_splits(7937u, true), 63u);
  EXPECT_EQ(sdpa_fd_num_splits(8192u, true), 64u);
  EXPECT_EQ(sdpa_fd_num_splits(8193u, true), 65u);
  EXPECT_EQ(sdpa_fd_num_splits(8960u, true), 70u);
  EXPECT_EQ(sdpa_fd_num_splits(16385u, true), 128u);
}

TEST(WebGPUUtils, GenericFdFallbackKeepsTile64Schedule) {
  EXPECT_EQ(sdpa_fd_split_tile(false), 64u);
  EXPECT_EQ(sdpa_fd_num_splits(128u, false), 2u);
  EXPECT_EQ(sdpa_fd_num_splits(8192u, false), 128u);
  EXPECT_EQ(sdpa_fd_num_splits(8960u, false), 128u);
}

TEST(WebGPUUtils, QwenGqa2FdUsesOneSplitWorkgroupPerKvHead) {
  EXPECT_EQ(sdpa_fd_split_head_count(16u, 8u, true), 8u);
  EXPECT_EQ(sdpa_fd_split_head_count(16u, 8u, false), 16u);
}

TEST(WebGPUUtils, QwenGqa2FdRouteReentersAfterLongContext) {
  const uint32_t splits[] = {
      sdpa_fd_num_splits(128u, true),
      sdpa_fd_num_splits(8192u, true),
      sdpa_fd_num_splits(128u, true),
  };
  EXPECT_EQ(splits[0], 1u);
  EXPECT_EQ(splits[1], 64u);
  EXPECT_EQ(splits[2], 1u);
}

TEST(WebGPUUtils, QwenGqa2FdShaderLoadsEachVElementOnceForTheHeadPair) {
  const std::string_view source(kSdpaFdSplitGqa2F16WGSL);
  const std::string_view load = "t_v_cache[";
  const size_t first = source.find(load);
  ASSERT_NE(first, std::string_view::npos);
  EXPECT_EQ(source.find(load, first + 1), std::string_view::npos);
  EXPECT_EQ(source.find("atomic"), std::string_view::npos);
}
