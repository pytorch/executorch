/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace vkcompute {

/*
 * Tap pruning for direct convolutions with large padding or dilation, where
 * many kernel taps read only padding. Along one axis, output position o reads
 * input position o * stride - padding + tap * dilation for each tap, and the
 * input reads as zero outside [0, input).
 */

// One spatial axis (width or height) of a convolution.
struct Conv2dAxis {
  int64_t input;
  int64_t output;
  int64_t kernel;
  int64_t stride;
  int64_t padding;
  int64_t dilation;
};

// Taps skipped by a direct convolution: none, those reading only padding for
// every output position, or those reading only padding for every pixel of
// each 2x2 output tile. The values are the shaders' skip_taps_mode
// specialization constant.
enum class Conv2dTapRange : int32_t {
  Disabled = 0,
  Global = 1,
  Tile = 2,
};

// Below this many 2x2-pixel x 8-channel tiles, few resident rounds expose the
// slowest tiles and per-tile tap bounds do not pay. The value is the pointwise
// convolution underfill threshold, which counts tiles of the same footprint.
inline constexpr uint64_t kMinTileTapRangeInvocations = 4096;

// Per-tile bounds cost about 11% more than global bounds when both skip the
// same taps, and workgroups that straddle a padding border wait for their
// slowest tile. Tile pruning is used only when it skips at least this share of
// the tap work beyond global pruning; the threshold lies between the measured
// shapes that did not gain (0.22) and did (0.42).
inline constexpr uint64_t kMinTileTapRangeGainPercent = 30;

/*
 * Taps [begin, end) of one axis that read inside the input for at least one
 * output position in [first, last]; all other taps read only padding there.
 * Exact for a single position. The tiled shaders compute the same bounds.
 */
inline std::array<int64_t, 2> conv2d_tap_bounds(
    const Conv2dAxis& axis,
    const int64_t first,
    const int64_t last) {
  const auto floor_div = [](const int64_t n, const int64_t d) {
    return n / d - (n % d < 0);
  };
  const int64_t first_origin = first * axis.stride - axis.padding;
  const int64_t last_origin = last * axis.stride - axis.padding;
  return {
      std::clamp(
          -floor_div(last_origin, axis.dilation), int64_t(0), axis.kernel),
      std::clamp(
          floor_div(axis.input - 1 - first_origin, axis.dilation) + 1,
          int64_t(0),
          axis.kernel)};
}

// Taps of one axis that read inside the input: per 2-pixel output tile, summed
// over the tiles, and for the whole output.
struct Conv2dAxisTaps {
  uint64_t tile;
  uint64_t global;
};

inline Conv2dAxisTaps count_conv2d_axis_taps(const Conv2dAxis& axis) {
  std::vector<bool> global(axis.kernel, false);
  uint64_t tile = 0;
  for (int64_t first = 0; first < axis.output; first += 2) {
    std::vector<bool> kept(axis.kernel, false);
    for (int64_t o = first; o < std::min(first + 2, axis.output); ++o) {
      const auto bounds = conv2d_tap_bounds(axis, o, o);
      for (int64_t k = bounds[0]; k < bounds[1]; ++k) {
        kept[k] = true;
        global[k] = true;
      }
    }
    tile += std::count(kept.begin(), kept.end(), true);
  }
  return {tile, uint64_t(std::count(global.begin(), global.end(), true))};
}

/*
 * Picks tap pruning for a convolution computing 2x2- or 2x1-pixel tiles, from
 * the exact share of tap work each mode skips. axes holds the width and height
 * axes. Work is counted in 2x2-pixel x 8-channel tiles for every tile shape.
 */
inline Conv2dTapRange pick_conv2d_tap_range(
    const std::array<Conv2dAxis, 2>& axes,
    const int64_t output_channels) {
  const Conv2dAxisTaps x = count_conv2d_axis_taps(axes[0]);
  const Conv2dAxisTaps y = count_conv2d_axis_taps(axes[1]);
  const uint64_t tiles =
      uint64_t((axes[0].output + 1) / 2) * uint64_t((axes[1].output + 1) / 2);
  const uint64_t taps = uint64_t(axes[0].kernel) * uint64_t(axes[1].kernel);
  const uint64_t tap_work = tiles * taps;
  const uint64_t tile_kept = x.tile * y.tile;
  const uint64_t global_kept = tiles * x.global * y.global;
  if (tile_kept == tap_work) {
    return Conv2dTapRange::Disabled;
  }
  const uint64_t invocations = tiles * uint64_t((output_channels + 7) / 8);
  if (invocations >= kMinTileTapRangeInvocations &&
      100 * (global_kept - tile_kept) >=
          kMinTileTapRangeGainPercent * tap_work) {
    return Conv2dTapRange::Tile;
  }
  return global_kept < tap_work ? Conv2dTapRange::Global
                                : Conv2dTapRange::Disabled;
}

} // namespace vkcompute
