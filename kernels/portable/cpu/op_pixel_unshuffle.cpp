/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {
namespace {

template <typename CTYPE>
void pixel_unshuffle_impl(
    const Tensor& in,
    int64_t downscale_factor,
    Tensor& out) {
  const CTYPE* const in_data = in.const_data_ptr<CTYPE>();
  CTYPE* const out_data = out.mutable_data_ptr<CTYPE>();

  const auto leading_dims = getLeadingDims(in, in.dim() - 3);
  const auto channels = out.size(in.dim() - 3);
  const auto height = out.size(in.dim() - 2);
  const auto width = out.size(in.dim() - 1);

  const auto S = downscale_factor;
  const auto sub_channels = channels / (S * S);

  // output strides
  const auto stride_n = channels * height * width;
  const auto stride_c = S * S * height * width;
  const auto stride_s1 = S * height * width;
  const auto stride_s2 = height * width;
  const auto stride_h = width;

  // input tensor shape of [n, c, h, s1, w, s2]
  // output tensor shape of [n, c, s1, s2, h, w]
  size_t i = 0;
  for (size_t n = 0; n < leading_dims; n++) {
    for (size_t c = 0; c < sub_channels; c++) {
      for (size_t h = 0; h < height; h++) {
        for (size_t s1 = 0; s1 < S; s1++) {
          for (size_t w = 0; w < width; w++) {
            for (size_t s2 = 0; s2 < S; s2++) {
              size_t output_offset = n * stride_n + c * stride_c +
                  s1 * stride_s1 + s2 * stride_s2 + h * stride_h + w;
              out_data[output_offset] = in_data[i++];
            }
          }
        }
      }
    }
  }
}

} // namespace

using SizesType = exec_aten::SizesType;
using Tensor = exec_aten::Tensor;

Tensor& pixel_unshuffle_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    int64_t downscale_factor,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      check_pixel_unshuffle_args(in, downscale_factor, out),
      InvalidArgument,
      out);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  Tensor::SizesType expected_out_size[kTensorDimensionLimit];
  size_t expected_out_dim = 0;
  get_pixel_unshuffle_out_target_size(
      in, downscale_factor, expected_out_size, &expected_out_dim);

  // Make sure the output tensor is the right size.
  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {expected_out_size, expected_out_dim}) == Error::Ok,
      InvalidArgument,
      out);

  constexpr auto name = "pixel_unshuffle.out";

  const auto in_type = out.scalar_type();
  // in and out must be the same dtype
  ET_SWITCH_ALL_TYPES(in_type, ctx, name, CTYPE, [&]() {
    pixel_unshuffle_impl<CTYPE>(in, downscale_factor, out);
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
