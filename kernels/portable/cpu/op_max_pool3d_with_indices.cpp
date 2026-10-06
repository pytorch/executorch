/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <c10/util/irange.h>
#include <limits>
#include <tuple>

#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;
using IntArrayRef = executorch::aten::ArrayRef<int64_t>;

namespace {

// Pools one {D, H, W} volume. torch reports the index of the maximum as its
// flat offset inside that volume, not inside the whole tensor.
template <typename CTYPE>
void max_pool3d_volume(
    const CTYPE* const in_ptr,
    const executorch::aten::StridesType* in_strides,
    const executorch::aten::SizesType* in_spatial_sizes,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    CTYPE* const out_ptr,
    int64_t* const indices_ptr,
    const executorch::aten::StridesType* out_strides,
    const executorch::aten::SizesType* out_spatial_sizes,
    const size_t leading_in_offset,
    const size_t leading_out_offset) {
  const int64_t in_D = in_spatial_sizes[0];
  const int64_t in_H = in_spatial_sizes[1];
  const int64_t in_W = in_spatial_sizes[2];

  for (const auto out_d : c10::irange(out_spatial_sizes[0])) {
    for (const auto out_h : c10::irange(out_spatial_sizes[1])) {
      for (const auto out_w : c10::irange(out_spatial_sizes[2])) {
        const int64_t out_coord[3] = {out_d, out_h, out_w};

        CTYPE accum = std::numeric_limits<CTYPE>::has_infinity
            ? -std::numeric_limits<CTYPE>::infinity()
            : std::numeric_limits<CTYPE>::lowest();
        int64_t accum_idx = 0;
        bool accum_initialized = false;

        for (const auto k_d : c10::irange(val_at(kernel_size, 0))) {
          for (const auto k_h : c10::irange(val_at(kernel_size, 1))) {
            for (const auto k_w : c10::irange(val_at(kernel_size, 2))) {
              const int64_t k_coord[3] = {k_d, k_h, k_w};
              int64_t in_coord[3];
              bool in_bounds = true;
              for (const auto i : c10::irange(3)) {
                const int64_t kernel_i = val_at(kernel_size, i);
                const int64_t stride_i =
                    val_at(stride, i, /*default_value=*/kernel_i);
                const int64_t padding_i =
                    val_at(padding, i, /*default_value=*/0);
                const int64_t dilation_i = val_at(dilation, i);
                in_coord[i] = out_coord[i] * stride_i +
                    k_coord[i] * dilation_i - padding_i;
                if (in_coord[i] < 0 || in_coord[i] >= in_spatial_sizes[i]) {
                  in_bounds = false;
                  break;
                }
              }
              if (!in_bounds) {
                continue;
              }

              const size_t in_idx = leading_in_offset +
                  in_coord[0] * in_strides[0] + in_coord[1] * in_strides[1] +
                  in_coord[2] * in_strides[2];
              const CTYPE in_val = in_ptr[in_idx];
              const int64_t flat_idx = in_coord[0] * in_H * in_W +
                  in_coord[1] * in_W + in_coord[2];
              if (!accum_initialized || in_val > accum) {
                accum = in_val;
                accum_idx = flat_idx;
                accum_initialized = true;
              }
            }
          }
        }
        (void)in_D;

        const size_t out_idx = leading_out_offset +
            out_coord[0] * out_strides[0] + out_coord[1] * out_strides[1] +
            out_coord[2] * out_strides[2];
        out_ptr[out_idx] = accum;
        indices_ptr[out_idx] = accum_idx;
      }
    }
  }
}

} // namespace

std::tuple<Tensor&, Tensor&> max_pool3d_with_indices_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    Tensor& out,
    Tensor& indices) {
  std::tuple<Tensor&, Tensor&> ret_val(out, indices);

  ET_KERNEL_CHECK(
      ctx,
      check_max_pool3d_with_indices_args(
          in, kernel_size, stride, padding, dilation, ceil_mode, out, indices),
      InvalidArgument,
      ret_val);

  size_t output_ndim = 0;
  // @lint-ignore CLANGTIDY facebook-hte-CArray
  executorch::aten::SizesType output_sizes[kTensorDimensionLimit];
  get_max_pool3d_with_indices_out_target_size(
      in,
      kernel_size,
      stride,
      padding,
      dilation,
      ceil_mode,
      output_sizes,
      &output_ndim);

  ET_KERNEL_CHECK(
      ctx,
      output_size_is_valid({output_sizes, output_ndim}, 3),
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {output_sizes, output_ndim}) == Error::Ok,
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(indices, {output_sizes, output_ndim}) == Error::Ok,
      InvalidArgument,
      ret_val);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  executorch::aten::StridesType in_strides[kTensorDimensionLimit];
  dim_order_to_stride_nocheck(
      in.sizes().data(), in.dim_order().data(), in.sizes().size(), in_strides);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  executorch::aten::StridesType out_strides[kTensorDimensionLimit];
  dim_order_to_stride_nocheck(
      out.sizes().data(),
      out.dim_order().data(),
      out.sizes().size(),
      out_strides);

  const size_t spatial_offset = in.dim() - 3;
  const size_t channel_dim = spatial_offset - 1;
  const size_t batch_size = in.dim() == 5 ? in.size(0) : 1;
  const size_t channels = in.size(channel_dim);

  // @lint-ignore CLANGTIDY facebook-hte-CArray
  ET_DEFINE_OPERATOR_NAME(name, "max_pool3d_with_indices.out");

  ET_SWITCH_REALHBF16_TYPES(in.scalar_type(), ctx, name, CTYPE, [&]() {
    const CTYPE* const in_ptr = in.const_data_ptr<CTYPE>();
    CTYPE* const out_ptr = out.mutable_data_ptr<CTYPE>();
    int64_t* const indices_ptr = indices.mutable_data_ptr<int64_t>();

    for (const auto batch : c10::irange(batch_size)) {
      for (const auto channel : c10::irange(channels)) {
        size_t leading_in_offset = channel * in_strides[channel_dim];
        size_t leading_out_offset = channel * out_strides[channel_dim];
        if (in.dim() == 5) {
          leading_in_offset += batch * in_strides[0];
          leading_out_offset += batch * out_strides[0];
        }
        max_pool3d_volume<CTYPE>(
            in_ptr,
            in_strides + spatial_offset,
            in.sizes().data() + spatial_offset,
            kernel_size,
            stride,
            padding,
            dilation,
            out_ptr,
            indices_ptr,
            out_strides + spatial_offset,
            out.sizes().data() + spatial_offset,
            leading_in_offset,
            leading_out_offset);
      }
    }
  });

  return ret_val;
}

} // namespace native
} // namespace executor
} // namespace torch
