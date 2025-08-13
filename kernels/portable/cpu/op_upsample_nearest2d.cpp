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
using executorch::aten::SizesType;
using std::optional;

namespace {
template <typename CTYPE>
void upsample_nearest2d_kernel_impl_nchw(
    const Tensor& in,
    const float scale_h,
    const float scale_w,
    Tensor& out) {
  const auto in_data = in.const_data_ptr<CTYPE>();
  auto out_data = out.mutable_data_ptr<CTYPE>();

  auto in_plane = in_data;
  for (auto n = 0; n < out.size(0); n++) {
    for (auto c = 0; c < out.size(1); c++) {
      for (auto h = 0; h < out.size(2); h++) {
        for (auto w = 0; w < out.size(3); w++) {
          const auto in_h =
              nearest_neighbor_compute_source_index(scale_h, h, in.sizes()[2]);
          const auto in_w =
              nearest_neighbor_compute_source_index(scale_w, w, in.sizes()[3]);

          *out_data = in_plane[in_h * in.strides()[2] + in_w * in.strides()[3]];
          out_data++;
        }
      }

      in_plane += in.strides()[1];
    }
  }
}

template <typename CTYPE>
void upsample_nearest2d_kernel_impl_nhwc(
    const Tensor& in,
    const float scale_h,
    const float scale_w,
    Tensor& out) {
  auto in_data = in.const_data_ptr<CTYPE>();
  auto out_data = out.mutable_data_ptr<CTYPE>();

  for (auto n = 0; n < out.size(0); n++) {
    for (auto h = 0; h < out.size(2); h++) {
      const auto in_h =
          nearest_neighbor_compute_source_index(scale_h, h, in.sizes()[2]);
      for (auto w = 0; w < out.size(3); w++) {
        const auto in_w =
            nearest_neighbor_compute_source_index(scale_w, w, in.sizes()[3]);
        for (auto c = 0; c < out.size(1); c++) {
          *out_data = in_data
              [in_h * in.strides()[2] + in_w * in.strides()[3] +
               c * in.strides()[1]];
          out_data++;
        }
      }
    }

    in_data += in.strides()[0];
  }
}

template <typename CTYPE>
void upsample_nearest2d_kernel_impl(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const float scale_h,
    const float scale_w,
    Tensor& out) {
  if (is_contiguous_dim_order(in.dim_order().data(), in.dim_order().size())) {
    upsample_nearest2d_kernel_impl_nchw<CTYPE>(in, scale_h, scale_w, out);
  } else if (is_channels_last_dim_order(
                 in.dim_order().data(), in.dim_order().size())) {
    upsample_nearest2d_kernel_impl_nhwc<CTYPE>(in, scale_h, scale_w, out);
  } else {
    // Shouldn't be reachable because of args checks, but just in case.
    ET_LOG(Error, "Unsupported dim order");
    ctx.fail(Error::InvalidArgument);
  }
}
} // namespace

Tensor& upsample_nearest2d_vec_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const executorch::aten::OptionalArrayRef<int64_t> output_size,
    const executorch::aten::OptionalArrayRef<double> scale_factors,
    Tensor& out) {
  // Preconditions (checked in check_..._args):
  //  In and out tensors have same dtype.
  //  In and out tensors are rank 4 and have same dim[0] and dim[1].
  //  In and out tensors are default dim order (NCHW).
  ET_KERNEL_CHECK(
      ctx,
      check_upsample_nearest2d_args(in, output_size, scale_factors, out),
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
      in.sizes()[2], out.sizes()[2], false, scale_h);
  const auto kernel_scale_w = area_pixel_compute_scale<double>(
      in.sizes()[3], out.sizes()[3], false, scale_w);

  ET_SWITCH_REALHBF16_TYPES(
      in.scalar_type(), ctx, "upsample_nearest2d.out", CTYPE, [&]() {
        upsample_nearest2d_kernel_impl<CTYPE>(
            ctx, in, kernel_scale_h, kernel_scale_w, out);
      });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
