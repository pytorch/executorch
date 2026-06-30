/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/webgpu/runtime/WebGPUUtils.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>

using executorch::backends::webgpu::utils::fold_workgroup_count_2d;
using executorch::backends::webgpu::utils::WgCount;

namespace {

// Device-free unit test for the pure 2D workgroup-count fold that lifts the
// 65535 per-dim dispatch cap. Exercises the fold arithmetic only — no GPU.
int g_failures = 0;

void expect_fold(
    uint32_t count,
    uint32_t max_count,
    uint32_t want_x,
    uint32_t want_y) {
  WgCount got = fold_workgroup_count_2d(count, max_count, "test");
  bool ok = got.x == want_x && got.y == want_y &&
      static_cast<uint64_t>(got.x) * got.y >= count;
  printf(
      "%s fold(%u, max=%u) = {%u, %u} (want {%u, %u})\n",
      ok ? "PASS:" : "FAIL:",
      count,
      max_count,
      got.x,
      got.y,
      want_x,
      want_y);
  if (!ok) {
    g_failures++;
  }
}

void expect_throw(uint32_t count, uint32_t max_count) {
  bool threw = false;
  try {
    fold_workgroup_count_2d(count, max_count, "test");
  } catch (const std::exception&) {
    threw = true;
  }
  printf(
      "%s fold(%u, max=%u) throws (needs a 3rd dispatch dimension)\n",
      threw ? "PASS:" : "FAIL:",
      count,
      max_count);
  if (!threw) {
    g_failures++;
  }
}

} // namespace

int main() {
  const uint32_t kMax = 65535u;
  // 1D fast path: count <= max -> {count, 1}, byte-identical to the old path.
  expect_fold(1u, kMax, 1u, 1u);
  expect_fold(kMax - 1u, kMax, kMax - 1u, 1u);
  expect_fold(kMax, kMax, kMax, 1u);
  // Fold to 2D: count > max -> {max, div_up(count, max)}.
  expect_fold(kMax + 1u, kMax, kMax, 2u);
  expect_fold(2u * kMax, kMax, kMax, 2u);
  expect_fold(2u * kMax + 1u, kMax, kMax, 3u);
  // Prefill-scale QK counts (tiled grid = Hq*ceil(S/4)*ceil(ctx/4)/wg) that
  // exceed kMax and must fold.
  expect_fold(131072u, kMax, kMax, 3u); // S=2048: 32*512*512/64
  expect_fold(2097152u, kMax, kMax, 33u); // deep fold (large-S stress)
  // count > max^2 needs a 3rd dispatch dimension -> throws (out of scope).
  expect_throw(kMax * kMax + 1u, kMax);

  if (g_failures != 0) {
    printf("\nFAIL: %d dispatch_2d fold case(s) failed\n", g_failures);
    return 1;
  }
  printf("\nAll dispatch_2d fold tests passed\n");
  return 0;
}
