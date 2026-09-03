/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Pins the C++ patch-grid choice to the Python reference's, tie-break included.
//
// vision/preprocess.h::compute_grid_size must agree with
// vision_precompute.compute_grid_size for every image size, because the two
// disagreeing means the runner resizes to a different resolution and emits a
// different <|patch|> count than whatever tokenized the prompt.
//
// The Python side of the same table lives in
// tests/test_vision_precompute.py::GridReferenceParityTest. Keep them in sync:
// the C++ test catches C++ regressions, the Python one catches the reference
// moving out from under us (see the CPython note in vision/preprocess.h).

#include <executorch/examples/models/muse-glimmer/vision/preprocess.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <utility>

namespace {

using ::executorch::examples::muse_glimmer_vision::compute_grid_size;
using ::executorch::examples::muse_glimmer_vision::kCell;
using ::executorch::examples::muse_glimmer_vision::kMaxImageTokens;

struct GridCase {
  int32_t img_w;
  int32_t img_h;
  int32_t nph; // grid rows the Python reference picks
  int32_t npw; // grid cols the Python reference picks
};

// Grids produced by running the Python reference, NOT by running the code
// under test. Several are exact cost ties that the reference resolves via
// CPython set order, following no rule -- which is precisely why they are
// worth pinning.
constexpr GridCase kReferenceGrids[] = {
    // Ties. 1440x1440 and 225x225 are the two cases named in P2447507824.
    {1440, 1440, 52, 52}, // reference takes the LARGER grid
    {225, 225, 8, 8}, // reference takes the SMALLER grid
    {512, 512, 19, 19},
    {1024, 1024, 37, 37},
    {200, 203, 8, 8}, // ties despite not being square
    {24, 60, 3, 1}, // ties on ratios straddling the aspect (2 and 3 vs 2.5)
    // No tie: both sides are exact multiples of the 28px cell.
    {700, 700, 25, 25},
    {336, 336, 12, 12},
    {1000, 2500, 90, 36},
    // Token cap: the ideal grid is rescaled to sqrt(4096) = 64 exactly.
    {6000, 6000, 64, 64},
    // Smaller than one cell, so floor() is 0 and only survives because the
    // >=1 filter runs after the candidate set is built, not before.
    {1, 1, 1, 1},
    {27, 27, 1, 1},
    {28, 28, 1, 1},
    // Extreme aspect ratio.
    {1, 10000, 358, 1},
};

TEST(VisionGridParityTest, MatchesPythonReference) {
  for (const GridCase& grid_case : kReferenceGrids) {
    int32_t target_h = 0;
    int32_t target_w = 0;
    compute_grid_size(grid_case.img_w, grid_case.img_h, target_h, target_w);

    const std::pair<int32_t, int32_t> expected{
        grid_case.nph * kCell, grid_case.npw * kCell};
    EXPECT_EQ(std::make_pair(target_h, target_w), expected)
        << "image " << grid_case.img_w << "x" << grid_case.img_h;
  }
}

// The crux of P2447507824: these two are both exact ties and the reference
// resolves them in OPPOSITE directions, so no fixed "prefer larger" or
// "prefer smaller" rule can satisfy both. A regression to either rule breaks
// exactly one of these assertions.
TEST(VisionGridParityTest, TiesResolveInOppositeDirections) {
  int32_t takes_ceil_h = 0;
  int32_t takes_ceil_w = 0;
  compute_grid_size(1440, 1440, takes_ceil_h, takes_ceil_w);
  EXPECT_EQ(takes_ceil_h / kCell, 52) << "1440x1440 must take ceil, not 51";

  int32_t takes_floor_h = 0;
  int32_t takes_floor_w = 0;
  compute_grid_size(225, 225, takes_floor_h, takes_floor_w);
  EXPECT_EQ(takes_floor_h / kCell, 8) << "225x225 must take floor, not 9";
}

// Invariants solo.cpp relies on when it splices soft tokens into the prompt.
TEST(VisionGridParityTest, TargetsAreCellAlignedAndWithinCap) {
  for (int32_t w = 1; w <= 600; w += 7) {
    for (int32_t h = 1; h <= 600; h += 11) {
      int32_t target_h = 0;
      int32_t target_w = 0;
      compute_grid_size(w, h, target_h, target_w);

      ASSERT_GT(target_h, 0) << w << "x" << h;
      ASSERT_GT(target_w, 0) << w << "x" << h;
      ASSERT_EQ(target_h % kCell, 0) << w << "x" << h;
      ASSERT_EQ(target_w % kCell, 0) << w << "x" << h;
      ASSERT_LE(
          static_cast<int64_t>(target_h / kCell) * (target_w / kCell),
          kMaxImageTokens)
          << w << "x" << h;
    }
  }
}

} // namespace
