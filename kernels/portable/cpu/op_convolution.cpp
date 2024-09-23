/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>

#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;
using IntArrayRef = exec_aten::ArrayRef<int64_t>;
using SizesArrayRef = exec_aten::ArrayRef<exec_aten::SizesType>;
using DimOrderArrayRef = exec_aten::ArrayRef<exec_aten::DimOrderType>;
using StridesArrayRef = exec_aten::ArrayRef<exec_aten::StridesType>;

namespace {

/**
 * Computes 2D convolution out results for a given group and channel. The
 * computation can be thought of as a stencil computation: we iterate over an
 * in of size in_C_per_group x in_H x in_W, with a stencil of size
 * in_C_per_group x in_H x in_W, to compute an out channel of size 1 x out_H x
 * out_W.
 */
template <typename CTYPE, typename CTYPE_BIAS>
void conv2d_impl(
    const CTYPE* const in_ptr,
    SizesArrayRef in_sizes,
    StridesArrayRef in_strides,
    const CTYPE* const w_ptr,
    SizesArrayRef w_sizes,
    StridesArrayRef w_strides,
    const CTYPE_BIAS* const bias_ptr,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    const int64_t groups,
    CTYPE* const out_ptr,
    SizesArrayRef out_sizes,
    StridesArrayRef out_strides,
    const size_t batch,
    const size_t group,
    const size_t out_c,
    bool transposed) {
  size_t in_C = in_sizes[1];
  size_t out_C = out_sizes[1];

  size_t out_H = out_sizes[2];
  size_t in_H = in_sizes[2];
  size_t w_H = w_sizes[2];

  size_t out_W = out_sizes[3];
  size_t in_W = in_sizes[3];
  size_t w_W = w_sizes[3];

  size_t in_C_per_group = in_C / groups;
  size_t in_c_start = group * in_C_per_group;

  size_t out_C_per_group = out_C / groups;
  size_t out_c_start = group * out_C_per_group;

  exec_aten::SizesType in_coord[kTensorDimensionLimit];
  in_coord[0] = batch;
  exec_aten::SizesType out_coord[kTensorDimensionLimit];
  out_coord[0] = batch;
  out_coord[1] = out_c;
  exec_aten::SizesType w_coord[kTensorDimensionLimit];

  const int64_t stride_y = val_at(stride, 0);
  const int64_t padding_y = val_at(padding, 0, /*default_value=*/0);
  const int64_t dilation_y = val_at(dilation, 0);
  const int64_t stride_x = val_at(stride, 1);
  const int64_t padding_x = val_at(padding, 1, /*default_value=*/0);
  const int64_t dilation_x = val_at(dilation, 1);

  if (!transposed) {
    w_coord[0] = out_c;
    // Compute 2D output region
    for (size_t out_y = 0; out_y < out_H; ++out_y) {
      out_coord[2] = out_y;
      for (size_t out_x = 0; out_x < out_W; ++out_x) {
        out_coord[3] = out_x;

        CTYPE accum = 0.0f;
        for (size_t in_c = in_c_start; in_c < in_c_start + in_C_per_group;
             ++in_c) {
          in_coord[1] = in_c;
          w_coord[1] = in_c - in_c_start;

          for (size_t w_y = 0; w_y < w_H; ++w_y) {
            w_coord[2] = w_y;

            size_t in_y = stride_y * out_y + dilation_y * w_y - padding_y;
            in_coord[2] = in_y;
            // Only proceed if input y coordinate is within bounds
            if (in_y >= 0 && in_y < in_H) {
              for (size_t w_x = 0; w_x < w_W; ++w_x) {
                w_coord[3] = w_x;

                size_t in_x = stride_x * out_x + dilation_x * w_x - padding_x;
                in_coord[3] = in_x;

                // Only proceed if input x coordinate is within bounds
                if (in_x >= 0 && in_x < in_W) {
                  size_t in_idx =
                      calculate_linear_index(in_coord, in_strides.data(), 4);
                  CTYPE in_val = in_ptr[in_idx];

                  size_t w_idx =
                      calculate_linear_index(w_coord, w_strides.data(), 4);
                  CTYPE w_val = w_ptr[w_idx];

                  accum += in_val * w_val;
                }
              }
            }
          }
        }

        if (bias_ptr != nullptr) {
          accum += convert<CTYPE, CTYPE_BIAS>(bias_ptr[out_c]);
        }
        size_t out_idx =
            calculate_linear_index(out_coord, out_strides.data(), 4);
        out_ptr[out_idx] = accum;
      }
    }
  } else { // transposed convolution
    w_coord[1] = out_c - out_c_start;

    for (size_t in_y = 0; in_y < in_H; ++in_y) {
      in_coord[2] = in_y;

      for (size_t in_x = 0; in_x < in_W; ++in_x) {
        in_coord[3] = in_x;

        for (size_t in_c = in_c_start; in_c < in_c_start + in_C_per_group;
             ++in_c) {
          in_coord[1] = in_c;

          size_t in_idx =
              calculate_linear_index(in_coord, in_strides.data(), 4);
          CTYPE in_val = in_ptr[in_idx];

          w_coord[0] = in_c;
          for (size_t w_y = 0; w_y < w_H; ++w_y) {
            w_coord[2] = w_y;
            size_t out_y = stride_y * in_y + dilation_y * w_y - padding_y;
            out_coord[2] = out_y;

            // Only proceed if output y coordinate is within bounds
            if (out_y >= 0 && out_y < out_H) {
              for (size_t w_x = 0; w_x < w_W; ++w_x) {
                w_coord[3] = w_x;
                size_t out_x = stride_x * in_x + dilation_x * w_x - padding_x;
                out_coord[3] = out_x;

                // Only proceed if output x coordinate is within bounds
                if (out_x >= 0 && out_x < out_W) {
                  size_t w_idx =
                      calculate_linear_index(w_coord, w_strides.data(), 4);
                  CTYPE w_val = w_ptr[w_idx];

                  size_t out_idx =
                      calculate_linear_index(out_coord, out_strides.data(), 4);

                  out_ptr[out_idx] += in_val * w_val;
                }
              }
            }
          }
        }
      }
    }
  }
}

template <typename CTYPE, typename CTYPE_BIAS>
void convolution_wrapper(
    const Tensor& in,
    const Tensor& weight,
    const exec_aten::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    int64_t groups,
    Tensor& out) {
  SizesArrayRef in_sizes = in.sizes();
  SizesArrayRef weight_sizes = weight.sizes();
  SizesArrayRef out_sizes = out.sizes();

  DimOrderArrayRef in_dim_order = in.dim_order();
  DimOrderArrayRef weight_dim_order = weight.dim_order();
  DimOrderArrayRef out_dim_order = out.dim_order();

  IntArrayRef stride_ = stride;
  IntArrayRef padding_ = padding;
  IntArrayRef dilation_ = dilation;

  // Define arrays for modified sizes, etc. which will potentially be used
  exec_aten::SizesType in_sizes_arr[kTensorDimensionLimit];
  exec_aten::DimOrderType in_dim_order_arr[kTensorDimensionLimit];
  size_t in_ndim;
  exec_aten::SizesType weight_sizes_arr[kTensorDimensionLimit];
  exec_aten::DimOrderType weight_dim_order_arr[kTensorDimensionLimit];
  size_t weight_ndim;
  exec_aten::SizesType out_sizes_arr[kTensorDimensionLimit];
  exec_aten::DimOrderType out_dim_order_arr[kTensorDimensionLimit];
  size_t out_ndim;

  int64_t stride_arr[2];
  int64_t padding_arr[2];
  int64_t dilation_arr[2];

  // If in has a dim of 3, then a 1D convolution will be performed. A 1D
  // convolution is equivalent to a 2D convolution where the height dim of
  // all tensors is 1, and stride = 1, padding = 0, and dilation = 1 for
  // the height dimension. Therefore the tensor sizes are unsqueezed and
  // the stride, padding, and dilation are adjusted so that a 2D
  // convolution implementation can be used.
  if (in.dim() == 3) {
    get_unsqueezed_sizes(in, 2, in_sizes_arr, in_ndim);
    in_sizes = {in_sizes_arr, in_ndim};
    get_unsqueezed_dim_order(in, 2, in_dim_order_arr);
    in_dim_order = {in_dim_order_arr, in_ndim};

    get_unsqueezed_sizes(weight, 2, weight_sizes_arr, weight_ndim);
    weight_sizes = {weight_sizes_arr, weight_ndim};
    get_unsqueezed_dim_order(weight, 2, weight_dim_order_arr);
    weight_dim_order = {weight_dim_order_arr, weight_ndim};

    get_unsqueezed_sizes(out, 2, out_sizes_arr, out_ndim);
    out_sizes = {out_sizes_arr, out_ndim};
    get_unsqueezed_dim_order(out, 2, out_dim_order_arr);
    out_dim_order = {out_dim_order_arr, out_ndim};

    stride_arr[0] = 1;
    stride_arr[1] = stride[0];
    stride_ = {stride_arr, 2};

    padding_arr[0] = 0;
    padding_arr[1] = padding[0];
    padding_ = {padding_arr, 2};

    dilation_arr[0] = 1;
    if (dilation.size() > 0) {
      dilation_arr[1] = dilation[0];
    } else {
      dilation_arr[1] = 1;
    }
    dilation_ = {dilation_arr, 2};
  }

  exec_aten::StridesType in_strides[kTensorDimensionLimit];
  dim_order_to_stride_nocheck(
      in_sizes.data(), in_dim_order.data(), in_sizes.size(), in_strides);

  exec_aten::StridesType weight_strides[kTensorDimensionLimit];
  dim_order_to_stride_nocheck(
      weight_sizes.data(),
      weight_dim_order.data(),
      weight_sizes.size(),
      weight_strides);

  exec_aten::StridesType out_strides[kTensorDimensionLimit];
  dim_order_to_stride_nocheck(
      out_sizes.data(), out_dim_order.data(), out_sizes.size(), out_strides);

  CTYPE* const out_ptr = out.mutable_data_ptr<CTYPE>();
  const CTYPE* const in_ptr = in.const_data_ptr<CTYPE>();
  const CTYPE* const w_ptr = weight.const_data_ptr<CTYPE>();
  const CTYPE_BIAS* const bias_ptr =
      bias.has_value() ? bias.value().const_data_ptr<CTYPE_BIAS>() : nullptr;

  size_t out_N = out.size(0);
  size_t out_C = out.size(1);
  size_t out_C_per_group = out_C / groups;

  if (transposed) {
    // For transposed convolution, we need to initialized the output before we
    // can accumulate into it.
    if (bias_ptr == nullptr) {
      // If bias is not present, we need to initialize the output to 0
      memset(out_ptr, 0, out.nbytes());
    } else {
      // If bias is present, we initialize the output to the bias value
      for (size_t out_ix = 0; out_ix < out.numel(); ++out_ix) {
        out_ptr[out_ix] = convert<CTYPE, CTYPE_BIAS>(
            bias_ptr[(out_ix / out_strides[1]) % out_C]);
      }
    }
  }

  for (size_t batch = 0; batch < out_N; ++batch) {
    for (size_t group = 0; group < groups; ++group) {
      // Align channel offset based on the group
      size_t out_c_start = group * out_C_per_group;
      // Populate all the out channels in the group
      for (size_t out_c = out_c_start; out_c < out_c_start + out_C_per_group;
           ++out_c) {
        conv2d_impl(
            in_ptr,
            in_sizes,
            {in_strides, 4},
            w_ptr,
            weight_sizes,
            {weight_strides, 4},
            bias_ptr,
            stride_,
            padding_,
            dilation_,
            groups,
            out_ptr,
            out_sizes,
            {out_strides, 4},
            batch,
            group,
            out_c,
            transposed);
      }
    }
  }
}

} // namespace

Tensor& convolution_out(
    KernelRuntimeContext& ctx,
    const Tensor& in,
    const Tensor& weight,
    const exec_aten::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      check_convolution_args(
          in,
          weight,
          bias,
          stride,
          padding,
          dilation,
          transposed,
          output_padding,
          groups,
          out),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  size_t output_ndim = 0;
  exec_aten::SizesType output_sizes[kTensorDimensionLimit];
  get_convolution_out_target_size(
      in,
      weight,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups,
      output_sizes,
      &output_ndim);

  ET_KERNEL_CHECK(
      ctx,
      output_size_is_valid({output_sizes, output_ndim}, in.dim() - 2),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, {output_sizes, output_ndim}) == Error::Ok,
      InvalidArgument,
      out);

  if (out.numel() == 0) {
    return out;
  }

  ScalarType in_type = in.scalar_type();
  ScalarType bias_type = in_type;
  if (bias.has_value()) {
    bias_type = bias.value().scalar_type();
  }

  constexpr auto name = "convolution.out";

  ET_SWITCH_REALH_TYPES(in_type, ctx, name, CTYPE, [&]() {
    ET_SWITCH_REALHB_TYPES(bias_type, ctx, name, CTYPE_BIAS, [&]() {
      convolution_wrapper<CTYPE, CTYPE_BIAS>(
          in, weight, bias, stride, padding, dilation, transposed, groups, out);
    });
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
