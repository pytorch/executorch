/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Device-free unit test for the pure 2D workgroup-count fold that lifts the
// 65535 per-dim dispatch cap. Exercises the fold arithmetic only — no GPU.

#include <executorch/backends/webgpu/runtime/WebGPUUtils.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>

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

} // namespace
