/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>

#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;
using IntArrayRef = executorch::aten::ArrayRef<int64_t>;

namespace {

inline int64_t
adaptive_start_index(int64_t out_idx, int64_t out_size, int64_t in_size) {
  return static_cast<int64_t>(
      std::floor(static_cast<float>(out_idx * in_size) / out_size));
}

inline int64_t
adaptive_end_index(int64_t out_idx, int64_t out_size, int64_t in_size) {
  return static_cast<int64_t>(
      std::ceil(static_cast<float>((out_idx + 1) * in_size) / out_size));
}

} // namespace

Tensor& _adaptive_avg_pool2d_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    IntArrayRef output_size,
    Tensor& out) {
  ET_KERNEL_CHECK(
      ctx,
      check_adaptive_avg_pool2d_args(in, output_size, out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ET_KERNEL_CHECK(ctx, tensor_is_default_dim_order(in), InvalidArgument, out);

  size_t output_ndim = 0;
  executorch::aten::SizesType output_sizes[kTensorDimensionLimit];
  get_adaptive_avg_pool2d_out_target_size(
      in, output_size, output_sizes, &output_ndim);

  ET_KERNEL_CHECK(
      ctx,
      output_size_is_valid({output_sizes, output_ndim}, 2),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {output_sizes, output_ndim}) == Error::Ok,
      InvalidArgument,
      out);

  ScalarType in_type = in.scalar_type();

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  static constexpr const char op_name[] = "_adaptive_avg_pool2d.out";

  ET_SWITCH_FLOATHBF16_TYPES_AND(Long, in_type, ctx, op_name, CTYPE, [&]() {
    const CTYPE* const in_ptr = in.const_data_ptr<CTYPE>();
    CTYPE* const out_ptr = out.mutable_data_ptr<CTYPE>();

    const size_t ndim = in.dim();
    const int64_t in_H = in.size(ndim - 2);
    const int64_t in_W = in.size(ndim - 1);
    const int64_t out_H = output_size[0];
    const int64_t out_W = output_size[1];

    const size_t channels = in.size(ndim - 3);
    const size_t batch_size = ndim == 4 ? in.size(0) : 1;

    const size_t in_plane_size = in_H * in_W;
    const size_t out_plane_size = out_H * out_W;

    for (size_t b = 0; b < batch_size; ++b) {
      for (size_t c = 0; c < channels; ++c) {
        const size_t plane_idx = b * channels + c;
        const CTYPE* plane_in = in_ptr + plane_idx * in_plane_size;
        CTYPE* plane_out = out_ptr + plane_idx * out_plane_size;

        for (int64_t oh = 0; oh < out_H; ++oh) {
          int64_t ih0 = adaptive_start_index(oh, out_H, in_H);
          int64_t ih1 = adaptive_end_index(oh, out_H, in_H);

          for (int64_t ow = 0; ow < out_W; ++ow) {
            int64_t iw0 = adaptive_start_index(ow, out_W, in_W);
            int64_t iw1 = adaptive_end_index(ow, out_W, in_W);

            float sum = 0;
            for (int64_t ih = ih0; ih < ih1; ++ih) {
              for (int64_t iw = iw0; iw < iw1; ++iw) {
                sum += plane_in[ih * in_W + iw];
              }
            }

            int64_t count = (ih1 - ih0) * (iw1 - iw0);
            plane_out[oh * out_W + ow] =
                static_cast<CTYPE>(sum / static_cast<float>(count));
          }
        }
      }
    }
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
