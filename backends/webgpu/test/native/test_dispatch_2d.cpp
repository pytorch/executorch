/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Device-free unit test for the pure 2D workgroup-count fold that lifts the
// 65535 per-dim dispatch cap. Exercises the fold arithmetic only — no GPU.

#include <executorch/backends/webgpu/runtime/WebGPUGraph.h>
#include <executorch/backends/webgpu/runtime/WebGPUUtils.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>

using executorch::backends::webgpu::WebGPUDispatch;
using executorch::backends::webgpu::WebGPUGraph;
using executorch::backends::webgpu::utils::DispatchRange;
using executorch::backends::webgpu::utils::DispatchRouteRegistry;
using executorch::backends::webgpu::utils::fold_workgroup_count_2d;
using executorch::backends::webgpu::utils::WgCount;

namespace {

constexpr uint32_t kMax = 65535u;

// count <= max -> {count, 1}: the 1D fast path, byte-identical to the old path.
TEST(DispatchFold, FastPath1D) {
  for (uint32_t count : {1u, kMax - 1u, kMax}) {
    const WgCount got = fold_workgroup_count_2d(count, kMax, "test");
    EXPECT_EQ(got.x, count);
    EXPECT_EQ(got.y, 1u);
  }
}

// count > max -> near-square {x, y}: fits the per-dim cap, covers every
// workgroup, and stays near-square so few invocations are inactive (launched -
// count is O(sqrt(count)); a flat {max, div_up} split would idle up to ~half).
TEST(DispatchFold, NearSquareFold) {
  // Includes prefill-scale QK counts (Hq*ceil(S/4)*ceil(ctx/4)/wg) that fold:
  // 131072 = S=2048 (32*512*512/64); 2097152 = large-S stress.
  for (uint32_t count :
       {kMax + 1u, 2u * kMax, 2u * kMax + 1u, 131072u, 2097152u}) {
    const WgCount got = fold_workgroup_count_2d(count, kMax, "test");
    const uint64_t launched = static_cast<uint64_t>(got.x) * got.y;
    const uint32_t root =
        static_cast<uint32_t>(std::ceil(std::sqrt(static_cast<double>(count))));
    EXPECT_LE(got.x, kMax) << "count=" << count;
    EXPECT_LE(got.y, kMax) << "count=" << count;
    EXPECT_GE(launched, count) << "count=" << count;
    EXPECT_LT(launched - count, 2ull * root)
        << "count=" << count << " launched=" << launched;
  }
}

// count > max^2 needs a 3rd dispatch dimension -> throws (out of scope).
TEST(DispatchFold, ThrowsWhenNeeds3rdDimension) {
  EXPECT_ANY_THROW(fold_workgroup_count_2d(kMax * kMax + 1u, kMax, "test"));
}

void expect_grid(const WgCount& grid, uint32_t x, uint32_t y) {
  EXPECT_EQ(grid.x, x);
  EXPECT_EQ(grid.y, y);
}

void expect_grids_equal(
    const std::vector<WgCount>& actual,
    const std::vector<WgCount>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < actual.size(); i++) {
    EXPECT_EQ(actual[i].x, expected[i].x) << "index=" << i;
    EXPECT_EQ(actual[i].y, expected[i].y) << "index=" << i;
  }
}

TEST(DispatchRoute, SwitchesUnequalRangesAndRestoresBothDimensions) {
  std::vector<WgCount> grids(6, {101, 103});
  const std::vector<bool> compute(6, true);
  DispatchRouteRegistry registry;
  const size_t group = registry.register_group(
      grids.size(), {{1, 3}, {4, 5}}, [&](size_t i) { return compute[i]; });
  auto set_grid = [&](size_t i, WgCount grid) { grids[i] = grid; };

  registry.select(group, 0, {{3, 7}, {5, 11}}, set_grid);
  expect_grid(grids[0], 101, 103);
  expect_grid(grids[1], 3, 7);
  expect_grid(grids[2], 5, 11);
  expect_grid(grids[3], 101, 103);
  expect_grid(grids[4], 0, 0);
  expect_grid(grids[5], 101, 103);

  registry.select(group, 1, {{13, 17}}, set_grid);
  expect_grid(grids[1], 0, 0);
  expect_grid(grids[2], 0, 0);
  expect_grid(grids[4], 13, 17);

  registry.select(group, 0, {{19, 23}, {29, 31}}, set_grid);
  expect_grid(grids[1], 19, 23);
  expect_grid(grids[2], 29, 31);
  expect_grid(grids[4], 0, 0);
}

TEST(DispatchRoute, InvalidSelectionDoesNotMutateGrids) {
  std::vector<WgCount> grids = {{1, 2}, {3, 4}, {5, 6}};
  const std::vector<WgCount> original = grids;
  DispatchRouteRegistry registry;
  const size_t group = registry.register_group(
      grids.size(), {{0, 2}, {2, 3}}, [](size_t) { return true; });
  auto set_grid = [&](size_t i, WgCount grid) { grids[i] = grid; };

  EXPECT_ANY_THROW(registry.select(group, 2, {{7, 8}}, set_grid));
  expect_grids_equal(grids, original);
  EXPECT_ANY_THROW(registry.select(group, 0, {{7, 8}}, set_grid));
  expect_grids_equal(grids, original);
  EXPECT_ANY_THROW(registry.select(group, 0, {{0, 8}, {9, 10}}, set_grid));
  expect_grids_equal(grids, original);
  EXPECT_ANY_THROW(registry.select(group, 0, {{7, 0}, {9, 10}}, set_grid));
  expect_grids_equal(grids, original);
}

TEST(DispatchRoute, RejectsInvalidAndMultiplyOwnedRanges) {
  const auto all_compute = [](size_t) { return true; };
  DispatchRouteRegistry registry;
  // Each pairs the range under test with a valid second range so the group
  // clears the size < 2 short-circuit and the range-validation loop is actually
  // exercised: {2,1} inverted, {1,1} empty, {0,5} end past dispatch_count.
  EXPECT_ANY_THROW(registry.register_group(4, {{2, 1}, {3, 4}}, all_compute));
  EXPECT_ANY_THROW(registry.register_group(4, {{1, 1}, {3, 4}}, all_compute));
  EXPECT_ANY_THROW(registry.register_group(4, {{0, 5}, {3, 4}}, all_compute));
  EXPECT_ANY_THROW(registry.register_group(4, {{0, 2}, {1, 3}}, all_compute));

  const size_t first =
      registry.register_group(4, {{0, 1}, {1, 2}}, all_compute);
  EXPECT_EQ(first, 0);
  EXPECT_ANY_THROW(registry.register_group(4, {{1, 3}}, all_compute));

  DispatchRouteRegistry copy_registry;
  const std::vector<bool> compute = {true, false, true};
  EXPECT_ANY_THROW(copy_registry.register_group(
      compute.size(), {{0, 1}, {1, 2}}, [&](size_t i) { return compute[i]; }));
}

TEST(DispatchRoute, GraphRejectsCopyAndCrossGroupOwnership) {
  WebGPUGraph graph;
  for (size_t i = 0; i < 6; i++) {
    graph.add_dispatch(WebGPUDispatch{});
  }
  graph.add_buffer_copy(nullptr, nullptr, 0);

  const size_t group = graph.register_dispatch_route_group({{0, 1}, {1, 2}});
  EXPECT_EQ(group, 0);
  EXPECT_ANY_THROW(graph.register_dispatch_route_group({{1, 2}, {2, 3}}));
  EXPECT_ANY_THROW(graph.register_dispatch_route_group({{5, 6}, {6, 7}}));

  const size_t second = graph.register_dispatch_route_group({{2, 4}, {4, 5}});
  EXPECT_EQ(second, 1);

  graph.select_dispatch_route(group, 0, {{7, 11}});
  graph.select_dispatch_route(second, 0, {{13, 17}, {19, 23}});
  expect_grid(
      {graph.dispatch_at(0).workgroup_count_x,
       graph.dispatch_at(0).workgroup_count_y},
      7,
      11);
  expect_grid(
      {graph.dispatch_at(1).workgroup_count_x,
       graph.dispatch_at(1).workgroup_count_y},
      0,
      0);
  expect_grid(
      {graph.dispatch_at(2).workgroup_count_x,
       graph.dispatch_at(2).workgroup_count_y},
      13,
      17);
  expect_grid(
      {graph.dispatch_at(3).workgroup_count_x,
       graph.dispatch_at(3).workgroup_count_y},
      19,
      23);
  expect_grid(
      {graph.dispatch_at(4).workgroup_count_x,
       graph.dispatch_at(4).workgroup_count_y},
      0,
      0);

  graph.select_dispatch_route(group, 1, {{29, 31}});
  expect_grid(
      {graph.dispatch_at(2).workgroup_count_x,
       graph.dispatch_at(2).workgroup_count_y},
      13,
      17);
}

TEST(DispatchRoute, ExecuteRejectsHalfZeroGrid) {
  WebGPUGraph graph;
  WebGPUDispatch dispatch;
  dispatch.workgroup_count_x = 0;
  dispatch.workgroup_count_y = 1;
  graph.add_dispatch(dispatch);
  EXPECT_ANY_THROW(graph.execute({}));
}

TEST(DispatchRoute, RecordsAlternatesOnlyForDynamicEligibleGraphs) {
  using executorch::backends::webgpu::utils::should_record_q4gsw_dual_route;
  using executorch::backends::webgpu::utils::should_record_sdpa_dual_route;

  EXPECT_FALSE(should_record_q4gsw_dual_route(32, true, false));
  EXPECT_FALSE(should_record_q4gsw_dual_route(1, true, true));
  EXPECT_FALSE(should_record_q4gsw_dual_route(32, false, true));
  EXPECT_TRUE(should_record_q4gsw_dual_route(32, true, true));

  EXPECT_FALSE(should_record_sdpa_dual_route(true, false));
  EXPECT_FALSE(should_record_sdpa_dual_route(false, true));
  EXPECT_TRUE(should_record_sdpa_dual_route(true, true));
}

} // namespace
