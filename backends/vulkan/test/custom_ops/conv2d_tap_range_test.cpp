// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <executorch/backends/vulkan/runtime/graph/ops/impl/convolution/conv2d/Conv2dTapRange.h>

#include <gtest/gtest.h>

#include <vector>

namespace vkcompute {
namespace {

bool reads_input(const Conv2dAxis& axis, const int64_t out, const int64_t tap) {
  const int64_t in = out * axis.stride - axis.padding + tap * axis.dilation;
  return in >= 0 && in < axis.input;
}

// Valid axes over strides, dilations, paddings and sizes, including outputs
// that read only padding.
std::vector<Conv2dAxis> sweep_axes() {
  std::vector<Conv2dAxis> axes;
  for (int64_t input : {1, 2, 5, 17, 33}) {
    for (int64_t kernel : {1, 3, 5}) {
      for (int64_t stride : {1, 2, 3}) {
        for (int64_t dilation : {1, 2, 7, 12, 36}) {
          for (int64_t padding : {0, 1, 4, 12, 36, 40}) {
            const int64_t span = input + 2 * padding - dilation * (kernel - 1);
            if (span >= 1) {
              axes.push_back(
                  {input,
                   (span - 1) / stride + 1,
                   kernel,
                   stride,
                   padding,
                   dilation});
            }
          }
        }
      }
    }
  }
  return axes;
}

std::array<Conv2dAxis, 2> square_axes(
    const int64_t size,
    const int64_t kernel,
    const int64_t padding,
    const int64_t dilation) {
  const int64_t out = size + 2 * padding - dilation * (kernel - 1);
  const Conv2dAxis axis{size, out, kernel, 1, padding, dilation};
  return {axis, axis};
}

TEST(Conv2dTapRangeTest, TapBoundsKeepEveryTapThatReadsInput) {
  for (const Conv2dAxis& axis : sweep_axes()) {
    for (int64_t first = 0; first < axis.output; ++first) {
      for (int64_t last = first; last < std::min(first + 3, axis.output);
           ++last) {
        const auto bounds = conv2d_tap_bounds(axis, first, last);
        for (int64_t tap = 0; tap < axis.kernel; ++tap) {
          bool reads = false;
          for (int64_t out = first; out <= last; ++out) {
            reads = reads || reads_input(axis, out, tap);
          }
          if (reads) {
            EXPECT_TRUE(tap >= bounds[0] && tap < bounds[1]);
          } else if (first == last) {
            EXPECT_FALSE(tap >= bounds[0] && tap < bounds[1]);
          }
        }
      }
    }
  }
}

TEST(Conv2dTapRangeTest, CountsMatchBruteForce) {
  for (const Conv2dAxis& axis : sweep_axes()) {
    uint64_t tile = 0;
    uint64_t global = 0;
    for (int64_t tap = 0; tap < axis.kernel; ++tap) {
      bool any = false;
      for (int64_t first = 0; first < axis.output; first += 2) {
        const bool kept = reads_input(axis, first, tap) ||
            (first + 1 < axis.output && reads_input(axis, first + 1, tap));
        tile += kept;
        any = any || kept;
      }
      global += any;
    }
    const Conv2dAxisTaps taps = count_conv2d_axis_taps(axis);
    EXPECT_EQ(taps.tile, tile);
    EXPECT_EQ(taps.global, global);
  }
}

TEST(Conv2dTapRangeTest, PicksModeByShareAndSize) {
  // 33x33 dilated 3x3 convolutions with 256 output channels.
  EXPECT_EQ(
      pick_conv2d_tap_range(square_axes(33, 3, 12, 12), 256),
      Conv2dTapRange::Tile);
  EXPECT_EQ(
      pick_conv2d_tap_range(square_axes(33, 3, 24, 24), 256),
      Conv2dTapRange::Tile);
  EXPECT_EQ(
      pick_conv2d_tap_range(square_axes(33, 3, 36, 36), 256),
      Conv2dTapRange::Global);
  // Too few tiles for per-tile bounds.
  EXPECT_EQ(
      pick_conv2d_tap_range(square_axes(33, 3, 12, 12), 64),
      Conv2dTapRange::None);
  EXPECT_EQ(
      pick_conv2d_tap_range(square_axes(17, 3, 8, 8), 128),
      Conv2dTapRange::None);
  EXPECT_EQ(
      pick_conv2d_tap_range(square_axes(17, 3, 8, 8), 512),
      Conv2dTapRange::Tile);
  EXPECT_EQ(
      pick_conv2d_tap_range(square_axes(17, 3, 24, 24), 256),
      Conv2dTapRange::Global);
  // Tile pruning skips less than the gain threshold beyond global pruning.
  EXPECT_EQ(
      pick_conv2d_tap_range(square_axes(33, 3, 6, 6), 256),
      Conv2dTapRange::None);
  // "Same" padding reads input at every tap of every tile.
  EXPECT_EQ(
      pick_conv2d_tap_range(square_axes(64, 3, 1, 1), 256),
      Conv2dTapRange::None);
  // 5x5 kernel.
  EXPECT_EQ(
      pick_conv2d_tap_range(square_axes(65, 5, 32, 16), 128),
      Conv2dTapRange::Tile);
  // Different width and height geometry, with stride 2 along the width.
  EXPECT_EQ(
      pick_conv2d_tap_range(
          {Conv2dAxis{37, 19, 5, 2, 20, 10}, Conv2dAxis{21, 21, 3, 1, 14, 14}},
          512),
      Conv2dTapRange::Tile);
}

} // namespace
} // namespace vkcompute
