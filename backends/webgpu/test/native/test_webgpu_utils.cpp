/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Device-free unit tests for the dispatch-grid math (WebGPUDispatchMath.h has
// zero WebGPU/Dawn dependency, unlike WebGPUUtils.h which needs a WGPUDevice
// for its other helpers).

#include <executorch/backends/webgpu/runtime/WebGPUDispatchMath.h>

#include <gtest/gtest.h>

using namespace executorch::backends::webgpu;

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
  utils::DispatchGrid g = utils::compute_dispatch_grid_from_limits(
      65536u, 1u, 65535u, "test");
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
