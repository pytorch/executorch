/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <c10/util/irange.h>

#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {
namespace {

void pixel_unshuffle_impl(
    const Tensor& in,
    int64_t downscale_factor,
    Tensor& out) {
  const char* const in_data =
      reinterpret_cast<const char*>(in.const_data_ptr());
  char* const out_data = reinterpret_cast<char*>(out.mutable_data_ptr());
  const auto elem_size = in.element_size();

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
  for (const auto n : c10::irange(leading_dims)) {
    for (const auto c : c10::irange(sub_channels)) {
      for (const auto h : c10::irange(height)) {
        for (const auto s1 : c10::irange(S)) {
          for (const auto w : c10::irange(width)) {
            for (const auto s2 : c10::irange(S)) {
              size_t output_offset = n * stride_n + c * stride_c +
                  s1 * stride_s1 + s2 * stride_s2 + h * stride_h + w;
              std::memcpy(
                  out_data + output_offset * elem_size,
                  in_data + i * elem_size,
                  elem_size);
              i++;
            }
          }
        }
      }
    }
  }
}

} // namespace

using SizesType = executorch::aten::SizesType;
using Tensor = executorch::aten::Tensor;

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

  pixel_unshuffle_impl(in, downscale_factor, out);

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
