/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <c10/util/irange.h>

#include <executorch/kernels/portable/cpu/util/grid_sampler_2d_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using executorch::aten::ArrayRef;
using executorch::aten::SizesType;
using std::optional;

namespace {
template <typename CTYPE>
void grid_sample_2d_bilinear_kernel_impl_nchw(
    const Tensor& in,
    const Tensor& grid,
    GridSamplerPadding padding_mode,
    bool align_corners,
    Tensor& out) {
  const auto in_data = in.const_data_ptr<CTYPE>();
  auto out_data = out.mutable_data_ptr<CTYPE>();

  // Grid has shape [N, H_out, W_out, 2]
  // Last dimension contains (x, y) normalized coordinates in [-1, 1]
  const auto grid_data = grid.const_data_ptr<CTYPE>();

  const int64_t N = in.size(0);
  const int64_t C = in.size(1);
  const int64_t inp_H = in.size(2);
  const int64_t inp_W = in.size(3);

  const int64_t out_H = out.size(2);
  const int64_t out_W = out.size(3);

  // Process each batch
  for (const auto n : c10::irange(N)) {
    const auto grid_offset = n * grid.strides()[0];
    const auto in_batch_offset = n * in.strides()[0];
    const auto out_batch_offset = n * out.strides()[0];

    // Process each channel
    for (const auto c : c10::irange(C)) {
      const auto in_channel_offset = in_batch_offset + c * in.strides()[1];
      const auto out_channel_offset = out_batch_offset + c * out.strides()[1];

      // Process each output pixel
      for (const auto h : c10::irange(out_H)) {
        for (const auto w : c10::irange(out_W)) {
          // Get grid coordinates for this output position
          // grid[n, h, w] contains (x, y)
          const int64_t grid_idx =
              grid_offset + h * grid.strides()[1] + w * grid.strides()[2];
          const CTYPE x = grid_data[grid_idx];
          const CTYPE y = grid_data[grid_idx + grid.strides()[3]];

          // Compute source coordinates in pixel space
          const CTYPE ix = grid_sampler_compute_source_index(
              x, inp_W, padding_mode, align_corners);
          const CTYPE iy = grid_sampler_compute_source_index(
              y, inp_H, padding_mode, align_corners);

          // Get corner pixel coordinates
          const int64_t ix_nw = static_cast<int64_t>(std::floor(ix));
          const int64_t iy_nw = static_cast<int64_t>(std::floor(iy));
          const int64_t ix_ne = ix_nw + 1;
          const int64_t iy_ne = iy_nw;
          const int64_t ix_sw = ix_nw;
          const int64_t iy_sw = iy_nw + 1;
          const int64_t ix_se = ix_nw + 1;
          const int64_t iy_se = iy_nw + 1;

          // Get interpolation weights
          const CTYPE nw_weight = (ix_se - ix) * (iy_se - iy);
          const CTYPE ne_weight = (ix - ix_sw) * (iy_sw - iy);
          const CTYPE sw_weight = (ix_ne - ix) * (iy - iy_ne);
          const CTYPE se_weight = (ix - ix_nw) * (iy - iy_nw);

          // Compute output value for this channel
          CTYPE out_val = 0;

          // Add contribution from each corner if within bounds
          if (padding_mode == GridSamplerPadding::Zeros) {
            // For zeros padding, only sample if within bounds
            if (within_bounds_2d(iy_nw, ix_nw, inp_H, inp_W)) {
              out_val += in_data
                             [in_channel_offset + iy_nw * in.strides()[2] +
                              ix_nw * in.strides()[3]] *
                  nw_weight;
            }
            if (within_bounds_2d(iy_ne, ix_ne, inp_H, inp_W)) {
              out_val += in_data
                             [in_channel_offset + iy_ne * in.strides()[2] +
                              ix_ne * in.strides()[3]] *
                  ne_weight;
            }
            if (within_bounds_2d(iy_sw, ix_sw, inp_H, inp_W)) {
              out_val += in_data
                             [in_channel_offset + iy_sw * in.strides()[2] +
                              ix_sw * in.strides()[3]] *
                  sw_weight;
            }
            if (within_bounds_2d(iy_se, ix_se, inp_H, inp_W)) {
              out_val += in_data
                             [in_channel_offset + iy_se * in.strides()[2] +
                              ix_se * in.strides()[3]] *
                  se_weight;
            }
          } else {
            // For border/reflection padding, clip corner indices to valid range
            // Even though source coordinates are clipped, adding 1 can push
            // corners out of bounds
            const int64_t ix_nw_safe = clip_coordinates(ix_nw, inp_W);
            const int64_t iy_nw_safe = clip_coordinates(iy_nw, inp_H);
            const int64_t ix_ne_safe = clip_coordinates(ix_ne, inp_W);
            const int64_t iy_ne_safe = clip_coordinates(iy_ne, inp_H);
            const int64_t ix_sw_safe = clip_coordinates(ix_sw, inp_W);
            const int64_t iy_sw_safe = clip_coordinates(iy_sw, inp_H);
            const int64_t ix_se_safe = clip_coordinates(ix_se, inp_W);
            const int64_t iy_se_safe = clip_coordinates(iy_se, inp_H);
            out_val = in_data
                          [in_channel_offset + iy_nw_safe * in.strides()[2] +
                           ix_nw_safe * in.strides()[3]] *
                    nw_weight +
                in_data
                        [in_channel_offset + iy_ne_safe * in.strides()[2] +
                         ix_ne_safe * in.strides()[3]] *
                    ne_weight +
                in_data
                        [in_channel_offset + iy_sw_safe * in.strides()[2] +
                         ix_sw_safe * in.strides()[3]] *
                    sw_weight +
                in_data
                        [in_channel_offset + iy_se_safe * in.strides()[2] +
                         ix_se_safe * in.strides()[3]] *
                    se_weight;
          }

          // Write output in NCHW order
          const int64_t out_idx =
              out_channel_offset + h * out.strides()[2] + w * out.strides()[3];
          out_data[out_idx] = out_val;
        }
      }
    }
  }
}

template <typename CTYPE>
void grid_sample_2d_nearest_kernel_impl_nchw(
    const Tensor& in,
    const Tensor& grid,
    GridSamplerPadding padding_mode,
    bool align_corners,
    Tensor& out) {
  const auto in_data = in.const_data_ptr<CTYPE>();
  auto out_data = out.mutable_data_ptr<CTYPE>();

  // Grid has shape [N, H_out, W_out, 2]
  // Last dimension contains (x, y) normalized coordinates in [-1, 1]
  const auto grid_data = grid.const_data_ptr<CTYPE>();

  const int64_t N = in.size(0);
  const int64_t C = in.size(1);
  const int64_t inp_H = in.size(2);
  const int64_t inp_W = in.size(3);

  const int64_t out_H = out.size(2);
  const int64_t out_W = out.size(3);

  // Process each batch
  for (const auto n : c10::irange(N)) {
    const auto grid_offset = n * grid.strides()[0];
    const auto in_batch_offset = n * in.strides()[0];
    const auto out_batch_offset = n * out.strides()[0];

    // Process each channel
    for (const auto c : c10::irange(C)) {
      const auto in_channel_offset = in_batch_offset + c * in.strides()[1];
      const auto out_channel_offset = out_batch_offset + c * out.strides()[1];

      // Process each output pixel
      for (const auto h : c10::irange(out_H)) {
        for (const auto w : c10::irange(out_W)) {
          // Get grid coordinates for this output position
          // grid[n, h, w] contains (x, y)
          const int64_t grid_idx =
              grid_offset + h * grid.strides()[1] + w * grid.strides()[2];
          const CTYPE x = grid_data[grid_idx];
          const CTYPE y = grid_data[grid_idx + grid.strides()[3]];

          // Compute source coordinates in pixel space
          const CTYPE ix = grid_sampler_compute_source_index(
              x, inp_W, padding_mode, align_corners);
          const CTYPE iy = grid_sampler_compute_source_index(
              y, inp_H, padding_mode, align_corners);

          // Get nearest pixel coordinates
          // Use nearbyint (not round) to match ATen's rounding behavior.
          // nearbyint uses the current rounding mode (typically round-to-even),
          // which matches PyTorch's (ATen's) behavior. In contrast, round may
          // not always respect the rounding mode. See:
          // aten/src/ATen/native/GridSampler.cpp
          int64_t ix_nearest = static_cast<int64_t>(std::nearbyint(ix));
          int64_t iy_nearest = static_cast<int64_t>(std::nearbyint(iy));

          // Compute output value for this channel
          CTYPE out_val = 0;

          // Check bounds and sample
          if (padding_mode == GridSamplerPadding::Zeros) {
            // For zeros padding, only sample if within bounds
            if (within_bounds_2d(iy_nearest, ix_nearest, inp_H, inp_W)) {
              out_val = in_data
                  [in_channel_offset + iy_nearest * in.strides()[2] +
                   ix_nearest * in.strides()[3]];
            }
          } else {
            // For border/reflection padding, clip coordinates after rounding
            // Rounding can push coordinates out of bounds even after
            // grid_sampler_compute_source_index
            int64_t ix_clipped = clip_coordinates(ix_nearest, inp_W);
            int64_t iy_clipped = clip_coordinates(iy_nearest, inp_H);
            out_val = in_data
                [in_channel_offset + iy_clipped * in.strides()[2] +
                 ix_clipped * in.strides()[3]];
          }

          // Write output in NCHW order
          const int64_t out_idx =
              out_channel_offset + h * out.strides()[2] + w * out.strides()[3];
          out_data[out_idx] = out_val;
        }
      }
    }
  }
}

template <typename CTYPE>
void grid_sample_2d_bicubic_kernel_impl_nchw(
    const Tensor& in,
    const Tensor& grid,
    GridSamplerPadding padding_mode,
    bool align_corners,
    Tensor& out) {
  const auto in_data = in.const_data_ptr<CTYPE>();
  auto out_data = out.mutable_data_ptr<CTYPE>();

  // Grid has shape [N, H_out, W_out, 2]
  // Last dimension contains (x, y) normalized coordinates in [-1, 1]
  const auto grid_data = grid.const_data_ptr<CTYPE>();

  const int64_t N = in.size(0);
  const int64_t C = in.size(1);
  const int64_t inp_H = in.size(2);
  const int64_t inp_W = in.size(3);

  const int64_t out_H = out.size(2);
  const int64_t out_W = out.size(3);

  // Process each batch
  for (const auto n : c10::irange(N)) {
    const auto grid_offset = n * grid.strides()[0];
    const auto in_batch_offset = n * in.strides()[0];
    const auto out_batch_offset = n * out.strides()[0];

    // Process each channel
    for (const auto c : c10::irange(C)) {
      const auto in_channel_offset = in_batch_offset + c * in.strides()[1];
      const auto out_channel_offset = out_batch_offset + c * out.strides()[1];

      // Process each output pixel
      for (const auto h : c10::irange(out_H)) {
        for (const auto w : c10::irange(out_W)) {
          // Get grid coordinates for this output position
          // grid[n, h, w] contains (x, y)
          const int64_t grid_idx =
              grid_offset + h * grid.strides()[1] + w * grid.strides()[2];
          const CTYPE x = grid_data[grid_idx];
          const CTYPE y = grid_data[grid_idx + grid.strides()[3]];

          // Compute source coordinates in pixel space
          // For bicubic, we need raw unnormalized coordinates without padding
          // applied Padding is applied later when fetching individual pixels
          // from the 4x4 neighborhood
          CTYPE ix = grid_sampler_unnormalize(x, inp_W, align_corners);
          CTYPE iy = grid_sampler_unnormalize(y, inp_H, align_corners);

          // Get the integer part and fractional part
          int64_t ix_0 = static_cast<int64_t>(std::floor(ix));
          int64_t iy_0 = static_cast<int64_t>(std::floor(iy));
          CTYPE tx = ix - ix_0;
          CTYPE ty = iy - iy_0;

          // Bicubic interpolation uses a 4x4 grid of pixels
          // Get the 16 pixel coordinates
          int64_t ix_m1 = ix_0 - 1;
          int64_t ix_p1 = ix_0 + 1;
          int64_t ix_p2 = ix_0 + 2;

          int64_t iy_m1 = iy_0 - 1;
          int64_t iy_p1 = iy_0 + 1;
          int64_t iy_p2 = iy_0 + 2;

          // Helper lambda to safely get pixel value with bounds checking
          auto get_value_bounded = [&](int64_t iy, int64_t ix) -> CTYPE {
            if (padding_mode == GridSamplerPadding::Zeros) {
              if (within_bounds_2d(iy, ix, inp_H, inp_W)) {
                return in_data
                    [in_channel_offset + iy * in.strides()[2] +
                     ix * in.strides()[3]];
              }
              return static_cast<CTYPE>(0);
            } else if (padding_mode == GridSamplerPadding::Border) {
              // For border padding, clip coordinates to valid range
              int64_t iy_safe =
                  std::max(static_cast<int64_t>(0), std::min(iy, inp_H - 1));
              int64_t ix_safe =
                  std::max(static_cast<int64_t>(0), std::min(ix, inp_W - 1));
              return in_data
                  [in_channel_offset + iy_safe * in.strides()[2] +
                   ix_safe * in.strides()[3]];
            } else {
              // For reflection padding, reflect coordinates at boundaries
              CTYPE iy_reflected = static_cast<CTYPE>(iy);
              CTYPE ix_reflected = static_cast<CTYPE>(ix);

              if (align_corners) {
                iy_reflected =
                    reflect_coordinates(iy_reflected, 0, 2 * (inp_H - 1));
                ix_reflected =
                    reflect_coordinates(ix_reflected, 0, 2 * (inp_W - 1));
              } else {
                iy_reflected =
                    reflect_coordinates(iy_reflected, -1, 2 * inp_H - 1);
                ix_reflected =
                    reflect_coordinates(ix_reflected, -1, 2 * inp_W - 1);
              }

              // Clip to ensure we're in bounds (reflection + clip for safety)
              int64_t iy_safe =
                  static_cast<int64_t>(clip_coordinates(iy_reflected, inp_H));
              int64_t ix_safe =
                  static_cast<int64_t>(clip_coordinates(ix_reflected, inp_W));

              return in_data
                  [in_channel_offset + iy_safe * in.strides()[2] +
                   ix_safe * in.strides()[3]];
            }
          };

          // Get the 4x4 grid of pixels
          // For each row, interpolate in x-direction
          CTYPE coefficients[4];

          // Row -1
          CTYPE p0 = get_value_bounded(iy_m1, ix_m1);
          CTYPE p1 = get_value_bounded(iy_m1, ix_0);
          CTYPE p2 = get_value_bounded(iy_m1, ix_p1);
          CTYPE p3 = get_value_bounded(iy_m1, ix_p2);
          coefficients[0] = cubic_interp1d(p0, p1, p2, p3, tx);

          // Row 0
          p0 = get_value_bounded(iy_0, ix_m1);
          p1 = get_value_bounded(iy_0, ix_0);
          p2 = get_value_bounded(iy_0, ix_p1);
          p3 = get_value_bounded(iy_0, ix_p2);
          coefficients[1] = cubic_interp1d(p0, p1, p2, p3, tx);

          // Row +1
          p0 = get_value_bounded(iy_p1, ix_m1);
          p1 = get_value_bounded(iy_p1, ix_0);
          p2 = get_value_bounded(iy_p1, ix_p1);
          p3 = get_value_bounded(iy_p1, ix_p2);
          coefficients[2] = cubic_interp1d(p0, p1, p2, p3, tx);

          // Row +2
          p0 = get_value_bounded(iy_p2, ix_m1);
          p1 = get_value_bounded(iy_p2, ix_0);
          p2 = get_value_bounded(iy_p2, ix_p1);
          p3 = get_value_bounded(iy_p2, ix_p2);
          coefficients[3] = cubic_interp1d(p0, p1, p2, p3, tx);

          // Interpolate in y-direction
          CTYPE out_val = cubic_interp1d(
              coefficients[0],
              coefficients[1],
              coefficients[2],
              coefficients[3],
              ty);

          // Write output in NCHW order
          const int64_t out_idx =
              out_channel_offset + h * out.strides()[2] + w * out.strides()[3];
          out_data[out_idx] = out_val;
        }
      }
    }
  }
}

} // namespace

// Signatures are auto-generated, so disable pass-by-value lint.
// NOLINTBEGIN(facebook-hte-ConstantArgumentPassByValue,
// facebook-hte-ParameterMightThrowOnCopy)
Tensor& grid_sampler_2d_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& grid,
    int64_t interpolation_mode,
    int64_t padding_mode,
    bool align_corners,
    Tensor& out) {
  // Check arguments and resize output tensor
  ET_KERNEL_CHECK_MSG(
      ctx,
      check_grid_sampler_2d_args_and_resize_out(input, grid, out) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to validate arguments and resize output tensor");

  // Convert integer mode parameters to enums
  GridSamplerInterpolation mode =
      static_cast<GridSamplerInterpolation>(interpolation_mode);
  GridSamplerPadding padding = static_cast<GridSamplerPadding>(padding_mode);

  // Validate mode and padding values
  ET_KERNEL_CHECK(
      ctx,
      mode == GridSamplerInterpolation::Bilinear ||
          mode == GridSamplerInterpolation::Nearest ||
          mode == GridSamplerInterpolation::Bicubic,
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      padding == GridSamplerPadding::Zeros ||
          padding == GridSamplerPadding::Border ||
          padding == GridSamplerPadding::Reflection,
      InvalidArgument,
      out);

  // Dispatch to appropriate implementation based on dtype
  ET_SWITCH_REALHBF16_TYPES(
      input.scalar_type(), ctx, "grid_sampler_2d.out", CTYPE, [&]() {
        // Dispatch to appropriate interpolation mode
        switch (mode) {
          case GridSamplerInterpolation::Bilinear:
            grid_sample_2d_bilinear_kernel_impl_nchw<CTYPE>(
                input, grid, padding, align_corners, out);
            break;
          case GridSamplerInterpolation::Nearest:
            grid_sample_2d_nearest_kernel_impl_nchw<CTYPE>(
                input, grid, padding, align_corners, out);
            break;
          case GridSamplerInterpolation::Bicubic:
            grid_sample_2d_bicubic_kernel_impl_nchw<CTYPE>(
                input, grid, padding, align_corners, out);
            break;
        }
      });

  return out;
}
// NOLINTEND(facebook-hte-ConstantArgumentPassByValue,
// facebook-hte-ParameterMightThrowOnCopy)

} // namespace native
} // namespace executor
} // namespace torch