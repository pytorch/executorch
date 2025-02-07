/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/upsample_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using executorch::aten::ArrayRef;
using executorch::aten::optional;
using executorch::aten::SizesType;

namespace {
template <typename CTYPE>
void upsample_bilinear2d_kernel_impl(
    const Tensor& in,
    bool align_corners,
    const float scale_h,
    const float scale_w,
    Tensor& out) {
  const auto in_data = in.const_data_ptr<CTYPE>();
  auto out_data = out.mutable_data_ptr<CTYPE>();

  auto in_plane = in_data;
  for (auto n = 0; n < out.size(0); n++) {
    for (auto c = 0; c < out.size(1); c++) {
      for (auto h = 0; h < out.size(2); h++) {
        // Compute source index and weights.
        int64_t in_h1, in_h2;
        float weight_h, inv_weight_h;

        compute_source_index_and_lambda(
            in_h1,
            in_h2,
            weight_h,
            inv_weight_h,
            scale_h,
            h,
            in.sizes()[2],
            out.sizes()[2],
            align_corners);

        for (auto w = 0; w < out.size(3); w++) {
          int64_t in_w1, in_w2;
          float weight_w, inv_weight_w;

          compute_source_index_and_lambda(
              in_w1,
              in_w2,
              weight_w,
              inv_weight_w,
              scale_w,
              w,
              in.sizes()[3],
              out.sizes()[3],
              align_corners);

          const auto top_left =
              in_plane[in_h1 * in.strides()[2] + in_w1 * in.strides()[3]];
          const auto top_right =
              in_plane[in_h1 * in.strides()[2] + in_w2 * in.strides()[3]];
          const auto bottom_left =
              in_plane[in_h2 * in.strides()[2] + in_w1 * in.strides()[3]];
          const auto bottom_right =
              in_plane[in_h2 * in.strides()[2] + in_w2 * in.strides()[3]];

          const auto top = top_left * weight_w + top_right * inv_weight_w;
          const auto bottom =
              bottom_left * weight_w + bottom_right * inv_weight_w;
          const auto val = top * weight_h + bottom * inv_weight_h;

          *out_data = val;
          out_data++;
        }
      }

      in_plane += in.strides()[1];
    }
  }
}
} // namespace

// Signatures are auto-generated, so disable pass-by-value lint.
// NOLINTBEGIN(facebook-hte-ConstantArgumentPassByValue,
// facebook-hte-ParameterMightThrowOnCopy)
Tensor& upsample_bilinear2d_vec_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const executorch::aten::OptionalArrayRef<int64_t> output_size,
    bool align_corners,
    const executorch::aten::OptionalArrayRef<double> scale_factors,
    Tensor& out) {
  // Preconditions (checked in check_..._args):
  //  In and out tensors have same dtype.
  //  In and out tensors are rank 4 and have same dim[0] and dim[1].
  //  In and out tensors are default dim order (NCHW).
  ET_KERNEL_CHECK(
      ctx,
      check_upsample_bilinear2d_args(
          in, output_size, align_corners, scale_factors, out),
      InvalidArgument,
      out);

  double scale_h, scale_w;

  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_upsample_2d(
          in, output_size, scale_factors, scale_h, scale_w, out) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor");

  const auto kernel_scale_h = area_pixel_compute_scale<double>(
      in.sizes()[2], out.sizes()[2], align_corners, scale_h);
  const auto kernel_scale_w = area_pixel_compute_scale<double>(
      in.sizes()[3], out.sizes()[3], align_corners, scale_w);

  ET_SWITCH_REALHBF16_TYPES(
      in.scalar_type(), ctx, "upsample_bilinear2d.out", CTYPE, [&]() {
        upsample_bilinear2d_kernel_impl<CTYPE>(
            in, align_corners, kernel_scale_h, kernel_scale_w, out);
      });

  return out;
}
// NOLINTEND(facebook-hte-ConstantArgumentPassByValue,
// facebook-hte-ParameterMightThrowOnCopy)

} // namespace native
} // namespace executor
} // namespace torch
