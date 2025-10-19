/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <c10/util/irange.h>
#include <algorithm>
#include <cmath>

#include <executorch/kernels/portable/cpu/util/upsample_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using executorch::aten::ArrayRef;
using executorch::aten::SizesType;
using std::optional;

namespace {

// Anti-aliasing filter matching PyTorch's implementation exactly
template <typename T>
inline T bilinear_aa_filter(T x) {
  x = std::abs(x);
  return (x < static_cast<T>(1.0)) ? (static_cast<T>(1.0) - x)
                                   : static_cast<T>(0.0);
}

// Compute anti-aliasing weights exactly matching PyTorch's algorithm
template <typename T>
void compute_aa_weights_for_pixel(
    int64_t output_idx,
    T scale,
    int64_t input_size,
    int64_t* indices,
    T* weights,
    int64_t* num_contributors) {
  // Use the provided scale directly instead of recalculating

  // PyTorch's center calculation for anti-aliasing
  // Always uses scale * (i + 0.5) for anti-aliasing, regardless of
  // align_corners
  const T center = scale * (output_idx + static_cast<T>(0.5));

  // PyTorch's support calculation for bilinear anti-aliasing
  // interp_size = 2 for bilinear, so base support = 1.0
  const T support = (scale >= static_cast<T>(1.0))
      ? (static_cast<T>(1.0) * scale)
      : static_cast<T>(1.0);

  // PyTorch's exact range calculation
  const int64_t xmin = std::max(
      static_cast<int64_t>(center - support + static_cast<T>(0.5)),
      static_cast<int64_t>(0));
  const int64_t xmax = std::min(
      static_cast<int64_t>(center + support + static_cast<T>(0.5)), input_size);

  *num_contributors = std::min(xmax - xmin, static_cast<int64_t>(4));

  // Ensure we have at least one contributor
  if (*num_contributors <= 0) {
    *num_contributors = 1;
    indices[0] = std::max(
        static_cast<int64_t>(0),
        std::min(static_cast<int64_t>(center), input_size - 1));
    weights[0] = static_cast<T>(1.0);
    // Clear unused weight slots
    for (int64_t j = 1; j < 4; ++j) {
      weights[j] = static_cast<T>(0.0);
    }
    return;
  }

  // PyTorch's weight computation
  T total_weight = static_cast<T>(0.0);
  const T invscale = (scale >= static_cast<T>(1.0))
      ? (static_cast<T>(1.0) / scale)
      : static_cast<T>(1.0);

  for (int64_t j = 0; j < *num_contributors; ++j) {
    int64_t x = xmin + j;
    // PyTorch's exact weight formula: (j + xmin - center + 0.5) * invscale
    T arg = (static_cast<T>(j) + static_cast<T>(xmin) - center +
             static_cast<T>(0.5)) *
        invscale;
    T weight = bilinear_aa_filter<T>(arg);
    indices[j] = x;
    weights[j] = weight;
    total_weight += weight;
  }

  // Normalize weights to sum to 1 (PyTorch does this)
  if (total_weight > static_cast<T>(0.0)) {
    for (int64_t j = 0; j < *num_contributors; ++j) {
      weights[j] /= total_weight;
    }
  } else {
    // Fallback: if total weight is 0, set equal weights
    T equal_weight = static_cast<T>(1.0) / static_cast<T>(*num_contributors);
    for (int64_t j = 0; j < *num_contributors; ++j) {
      weights[j] = equal_weight;
    }
  }

  // Clear unused weight slots
  for (int64_t j = *num_contributors; j < 4; ++j) {
    weights[j] = static_cast<T>(0.0);
  }
}

template <typename CTYPE>
void upsample_bilinear2d_aa_kernel_impl(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    bool align_corners,
    const float scale_h,
    const float scale_w,
    Tensor& out) {
  const auto in_data = in.const_data_ptr<CTYPE>();
  auto out_data = out.mutable_data_ptr<CTYPE>();

  const bool is_nchw =
      is_contiguous_dim_order(in.dim_order().data(), in.dim_order().size());

  if (is_nchw) {
    // NCHW layout
    for (int64_t n = 0; n < out.size(0); ++n) {
      for (int64_t c = 0; c < out.size(1); ++c) {
        const auto in_plane =
            in_data + (n * in.size(1) + c) * in.size(2) * in.size(3);
        auto out_plane =
            out_data + (n * out.size(1) + c) * out.size(2) * out.size(3);

        for (int64_t oh = 0; oh < out.size(2); ++oh) {
          // Compute height weights for this output row
          int64_t h_indices[4];
          float h_weights[4];
          int64_t h_num_contributors;
          compute_aa_weights_for_pixel<float>(
              oh,
              scale_h,
              in.size(2),
              h_indices,
              h_weights,
              &h_num_contributors);

          for (int64_t ow = 0; ow < out.size(3); ++ow) {
            // Compute width weights for this output column
            int64_t w_indices[4];
            float w_weights[4];
            int64_t w_num_contributors;
            compute_aa_weights_for_pixel<float>(
                ow,
                scale_w,
                in.size(3),
                w_indices,
                w_weights,
                &w_num_contributors);

            CTYPE value = 0;

            // Apply anti-aliased interpolation
            for (int64_t ih_idx = 0; ih_idx < h_num_contributors; ++ih_idx) {
              int64_t ih = h_indices[ih_idx];
              float h_weight = h_weights[ih_idx];

              for (int64_t iw_idx = 0; iw_idx < w_num_contributors; ++iw_idx) {
                int64_t iw = w_indices[iw_idx];
                float w_weight = w_weights[iw_idx];

                value += in_plane[ih * in.size(3) + iw] * h_weight * w_weight;
              }
            }

            out_plane[oh * out.size(3) + ow] = value;
          }
        }
      }
    }
  } else {
    // NHWC layout
    for (int64_t n = 0; n < out.size(0); ++n) {
      const auto in_batch = in_data + n * in.size(1) * in.size(2) * in.size(3);
      auto out_batch = out_data + n * out.size(1) * out.size(2) * out.size(3);

      for (int64_t oh = 0; oh < out.size(2); ++oh) {
        // Compute height weights for this output row
        int64_t h_indices[4];
        float h_weights[4];
        int64_t h_num_contributors;
        compute_aa_weights_for_pixel<float>(
            oh, scale_h, in.size(2), h_indices, h_weights, &h_num_contributors);

        for (int64_t ow = 0; ow < out.size(3); ++ow) {
          // Compute width weights for this output column
          int64_t w_indices[4];
          float w_weights[4];
          int64_t w_num_contributors;
          compute_aa_weights_for_pixel<float>(
              ow,
              scale_w,
              in.size(3),
              w_indices,
              w_weights,
              &w_num_contributors);

          for (int64_t c = 0; c < out.size(1); ++c) {
            CTYPE value = 0;

            // Apply anti-aliased interpolation
            for (int64_t ih_idx = 0; ih_idx < h_num_contributors; ++ih_idx) {
              int64_t ih = h_indices[ih_idx];
              float h_weight = h_weights[ih_idx];

              for (int64_t iw_idx = 0; iw_idx < w_num_contributors; ++iw_idx) {
                int64_t iw = w_indices[iw_idx];
                float w_weight = w_weights[iw_idx];

                value += in_batch[(ih * in.size(3) + iw) * in.size(1) + c] *
                    h_weight * w_weight;
              }
            }

            out_batch[(oh * out.size(3) + ow) * out.size(1) + c] = value;
          }
        }
      }
    }
  }
}

} // namespace

// Check function for anti-aliased bilinear upsampling
bool check_upsample_bilinear2d_aa_args(
    const Tensor& in,
    const executorch::aten::OptionalArrayRef<int64_t>& output_size,
    const bool align_corners,
    const executorch::aten::OptionalArrayRef<double>& scale_factors,
    Tensor& out) {
  // Use the same checks as regular bilinear upsampling
  return check_upsample_bilinear2d_args(
      in, output_size, align_corners, scale_factors, out);
}

// Main entry point for anti-aliased bilinear upsampling
Tensor& _upsample_bilinear2d_aa_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const executorch::aten::ArrayRef<int64_t> output_size,
    bool align_corners,
    const std::optional<double> scale_h,
    const std::optional<double> scale_w,
    Tensor& out) {
  // Preconditions (checked in check_..._args):
  //  In and out tensors have same dtype.
  //  In and out tensors are rank 4 and have same dim[0] and dim[1].
  //  In and out tensors are NHWC or NCHW dim order.

  // Custom validation for our specific interface (ArrayRef + optional
  // individual scales)
  ET_KERNEL_CHECK(ctx, in.dim() == 4, InvalidArgument, out);
  ET_KERNEL_CHECK(ctx, out.dim() == 4, InvalidArgument, out);
  ET_KERNEL_CHECK(
      ctx, in.scalar_type() == out.scalar_type(), InvalidArgument, out);
  ET_KERNEL_CHECK(ctx, output_size.size() == 2, InvalidArgument, out);
  ET_KERNEL_CHECK(
      ctx, output_size[0] > 0 && output_size[1] > 0, InvalidArgument, out);

  // Ensure output tensor has correct dimensions
  ET_KERNEL_CHECK(
      ctx, out.size(0) == in.size(0), InvalidArgument, out); // batch
  ET_KERNEL_CHECK(
      ctx, out.size(1) == in.size(1), InvalidArgument, out); // channels
  ET_KERNEL_CHECK(
      ctx, out.size(2) == output_size[0], InvalidArgument, out); // height
  ET_KERNEL_CHECK(
      ctx, out.size(3) == output_size[1], InvalidArgument, out); // width

  // Compute final scales - use provided scales if available, otherwise compute
  // from sizes
  double final_scale_h, final_scale_w;
  if (scale_h.has_value() && scale_w.has_value()) {
    final_scale_h = scale_h.value();
    final_scale_w = scale_w.value();
  } else {
    // Compute scales from input/output sizes
    final_scale_h =
        static_cast<double>(output_size[0]) / static_cast<double>(in.size(2));
    final_scale_w =
        static_cast<double>(output_size[1]) / static_cast<double>(in.size(3));
  }

  const auto kernel_scale_h = area_pixel_compute_scale<double>(
      in.sizes()[2], out.sizes()[2], align_corners, final_scale_h);
  const auto kernel_scale_w = area_pixel_compute_scale<double>(
      in.sizes()[3], out.sizes()[3], align_corners, final_scale_w);

  ET_SWITCH_REALHBF16_TYPES(
      in.scalar_type(), ctx, "_upsample_bilinear2d_aa.out", CTYPE, [&]() {
        upsample_bilinear2d_aa_kernel_impl<CTYPE>(
            ctx, in, align_corners, kernel_scale_h, kernel_scale_w, out);
      });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
