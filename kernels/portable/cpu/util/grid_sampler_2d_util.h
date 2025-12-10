/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {

// Ported from aten/src/ATen/native/GridSampler.h
// note that these need to be in the SAME ORDER as the enum in GridSampler.h
// as they are mapped to integer values (0, 1, 2) in this order
enum class GridSamplerInterpolation { Bilinear, Nearest, Bicubic };
enum class GridSamplerPadding { Zeros, Border, Reflection };

// Ported from aten/src/ATen/native/GridSampler.h
// Unnormalizes a coordinate from the -1 to +1 scale to its pixel index value,
// where we view each pixel as an area between (idx - 0.5) and (idx + 0.5).
// if align_corners: -1 and +1 get sent to the centers of the corner pixels
//     -1 --> 0
//     +1 --> (size - 1)
//     scale_factor = (size - 1) / 2
// if not align_corners: -1 and +1 get sent to the image edges
//     -1 --> -0.5
//     +1 --> (size - 1) + 0.5 == size - 0.5
//     scale_factor = size / 2
template <typename scalar_t>
inline scalar_t
grid_sampler_unnormalize(scalar_t coord, int64_t size, bool align_corners) {
  if (align_corners) {
    // unnormalize coord from [-1, 1] to [0, size - 1]
    return ((coord + 1) / 2) * (size - 1);
  } else {
    // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
    return ((coord + 1) * size - 1) / 2;
  }
}

// Ported from aten/src/ATen/native/GridSampler.h
// Clips coordinates to between 0 and clip_limit - 1
template <typename scalar_t>
inline scalar_t clip_coordinates(scalar_t in, int64_t clip_limit) {
  return std::min(
      static_cast<scalar_t>(clip_limit - 1),
      std::max(in, static_cast<scalar_t>(0)));
}

// Ported from aten/src/ATen/native/GridSampler.h
// Reflects coordinates until they fall between low and high (inclusive).
// The bounds are passed as twice their value so that half-integer values
// can be represented as ints.
template <typename scalar_t>
inline scalar_t
reflect_coordinates(scalar_t in, int64_t twice_low, int64_t twice_high) {
  if (twice_low == twice_high) {
    return static_cast<scalar_t>(0);
  }
  scalar_t min = static_cast<scalar_t>(twice_low) / 2;
  scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
  in = std::fabs(in - min);
  // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
  scalar_t extra = std::fmod(in, span);
  int flips = static_cast<int>(std::floor(in / span));
  if (flips % 2 == 0) {
    return extra + min;
  } else {
    return span - extra + min;
  }
}

// Ported from aten/src/ATen/native/GridSampler.h
// Computes the pixel source index value for a grid coordinate
template <typename scalar_t>
inline scalar_t grid_sampler_compute_source_index(
    scalar_t coord,
    int64_t size,
    GridSamplerPadding padding_mode,
    bool align_corners) {
  coord = grid_sampler_unnormalize(coord, size, align_corners);
  if (padding_mode == GridSamplerPadding::Border) {
    // clip coordinates to image borders
    coord = clip_coordinates(coord, size);
  } else if (padding_mode == GridSamplerPadding::Reflection) {
    // reflect coordinates by image borders
    if (align_corners) {
      coord = reflect_coordinates(coord, 0, 2 * (size - 1));
    } else {
      coord = reflect_coordinates(coord, -1, 2 * size - 1);
    }
    coord = clip_coordinates(coord, size);
  }
  return coord;
}

// Ported from aten/src/ATen/native/GridSampler.h
// Check if coordinates are within bounds [0, limit-1]
template <typename scalar_t>
inline bool within_bounds_2d(scalar_t h, scalar_t w, int64_t H, int64_t W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}

// Ported from aten/src/ATen/native/UpSample.h
// Cubic convolution function 1 (for points within 1 unit of the point)
template <typename scalar_t>
inline scalar_t cubic_convolution1(scalar_t x, scalar_t A) {
  return ((A + 2) * x - (A + 3)) * x * x + 1;
}

// Ported from aten/src/ATen/native/UpSample.h
// Cubic convolution function 2 (for points between 1 and 2 units from the
// point)
template <typename scalar_t>
inline scalar_t cubic_convolution2(scalar_t x, scalar_t A) {
  return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
}

// Ported from aten/src/ATen/native/UpSample.h
// Computes the 4 cubic interpolation coefficients for a given position t in [0,
// 1]
template <typename scalar_t>
inline void get_cubic_upsample_coefficients(scalar_t coeffs[4], scalar_t t) {
  // Standard bicubic interpolation uses alpha = -0.75
  scalar_t A = static_cast<scalar_t>(-0.75);

  scalar_t x1 = t;
  coeffs[0] = cubic_convolution2<scalar_t>(x1 + static_cast<scalar_t>(1.0), A);
  coeffs[1] = cubic_convolution1<scalar_t>(x1, A);

  scalar_t x2 = static_cast<scalar_t>(1.0) - t;
  coeffs[2] = cubic_convolution1<scalar_t>(x2, A);
  coeffs[3] = cubic_convolution2<scalar_t>(x2 + static_cast<scalar_t>(1.0), A);
}

// Ported from aten/src/ATen/native/UpSample.h
// Performs 1D cubic interpolation given 4 points and a position t in [0, 1]
template <typename scalar_t>
inline scalar_t
cubic_interp1d(scalar_t x0, scalar_t x1, scalar_t x2, scalar_t x3, scalar_t t) {
  scalar_t coeffs[4];
  get_cubic_upsample_coefficients<scalar_t>(coeffs, t);

  return x0 * coeffs[0] + x1 * coeffs[1] + x2 * coeffs[2] + x3 * coeffs[3];
}

// Argument checking and output tensor resizing for grid_sampler_2d
Error check_grid_sampler_2d_args_and_resize_out(
    const Tensor& input,
    const Tensor& grid,
    Tensor& out);

} // namespace executor
} // namespace torch