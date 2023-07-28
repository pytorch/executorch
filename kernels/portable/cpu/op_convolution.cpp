/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>

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
 * Extracts a value at index i from an int array. If the array length is 1, then
 * the first element will be returned regardless of what i is requested to
 * simulate broadcasting.
 */
inline int64_t val_at(IntArrayRef array, size_t i) {
  if (array.size() == 1) {
    return array[0];
  } else if (array.size() > 1) {
    return array[i];
  } else {
    ET_CHECK_MSG(false, "Attempted to retrieve from an empty array!");
  }
}

inline void get_unsqueezed_sizes(
    const Tensor& t,
    int64_t unsqueeze_dim,
    exec_aten::SizesType* sizes_arr,
    size_t& ndim) {
  ndim = t.dim() + 1;
  for (int d = 0; d < unsqueeze_dim; ++d) {
    sizes_arr[d] = t.size(d);
  }
  sizes_arr[unsqueeze_dim] = 1;
  for (int d = (unsqueeze_dim + 1); d < ndim; d++) {
    sizes_arr[d] = t.size(d - 1);
  }
}

inline void get_unsqueezed_dim_order(
    const Tensor& t,
    exec_aten::DimOrderType unsqueeze_dim,
    exec_aten::DimOrderType* dim_order_arr) {
  int offset = 0;
  for (int i = 0; i < t.dim(); ++i) {
    exec_aten::DimOrderType dim = t.dim_order()[i];
    if (dim == unsqueeze_dim) {
      dim_order_arr[i] = dim;
      dim_order_arr[i + 1] = dim + 1;
      offset = 1;
    } else {
      dim_order_arr[i + offset] = dim > unsqueeze_dim ? dim + 1 : dim;
    }
  }
  return;
}

/**
 * Computes 2D convolution out results for a given group and channel. The
 * computation can be thought of as a stencil computation: we iterate over an
 * in of size in_C_per_group x in_H x in_W, with a stencil of size
 * in_C_per_group x in_H x in_W, to compute an out channel of size 1 x out_H x
 * out_W.
 */
template <typename CTYPE>
void conv2d_impl(
    const CTYPE* const in_ptr,
    SizesArrayRef in_sizes,
    StridesArrayRef in_strides,
    const CTYPE* const w_ptr,
    SizesArrayRef w_sizes,
    StridesArrayRef w_strides,
    const CTYPE* const bias_ptr,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    const int64_t groups,
    CTYPE* const out_ptr,
    SizesArrayRef out_sizes,
    StridesArrayRef out_strides,
    const size_t batch,
    const size_t group,
    const size_t out_c) {
  size_t in_C = in_sizes[1];

  size_t out_H = out_sizes[2];
  size_t in_H = in_sizes[2];
  size_t w_H = w_sizes[2];

  size_t out_W = out_sizes[3];
  size_t in_W = in_sizes[3];
  size_t w_W = w_sizes[3];

  size_t in_C_per_group = in_C / groups;
  size_t in_c_start = group * in_C_per_group;

  exec_aten::SizesType in_coord[kTensorDimensionLimit];
  in_coord[0] = batch;
  exec_aten::SizesType out_coord[kTensorDimensionLimit];
  out_coord[0] = batch;
  out_coord[1] = out_c;
  exec_aten::SizesType w_coord[kTensorDimensionLimit];
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

          int64_t dilation_y = 1;
          if (dilation.size() > 0) {
            dilation_y = val_at(dilation, 0);
          }
          int64_t stride_y = val_at(stride, 0);
          int64_t padding_y = val_at(padding, 0);
          size_t in_y = stride_y * out_y + dilation_y * w_y - padding_y;
          in_coord[2] = in_y;
          // Only proceed if input y coordinate is within bounds
          if (in_y >= 0 && in_y < in_H) {
            for (size_t w_x = 0; w_x < w_W; ++w_x) {
              w_coord[3] = w_x;

              int64_t dilation_x = 1;
              if (dilation.size() > 0) {
                dilation_x = val_at(dilation, 0);
              }
              int64_t stride_x = val_at(stride, 1);
              int64_t padding_x = val_at(padding, 1);
              size_t in_x = stride_x * out_x + dilation_x * w_x - padding_x;
              in_coord[3] = in_x;

              // Only proceed if input  coordinate is within bounds
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
        accum += bias_ptr[out_c];
      }
      size_t out_idx = calculate_linear_index(out_coord, out_strides.data(), 4);
      out_ptr[out_idx] = accum;
    }
  }
}

template <typename CTYPE>
void convolution_wrapper(
    const Tensor& in,
    const Tensor& weight,
    const exec_aten::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    Tensor& out) {
  size_t out_N = in.size(0);
  size_t out_C = weight.size(0);

  // Compute the number of in and out channels in each group
  size_t out_C_per_group = out_C / groups;

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
  ET_CHECK(
      dim_order_to_stride(
          in_sizes.data(), in_dim_order.data(), in_sizes.size(), in_strides) ==
      Error::Ok);

  exec_aten::StridesType weight_strides[kTensorDimensionLimit];
  ET_CHECK(
      dim_order_to_stride(
          weight_sizes.data(),
          weight_dim_order.data(),
          weight_sizes.size(),
          weight_strides) == Error::Ok);

  exec_aten::StridesType out_strides[kTensorDimensionLimit];
  ET_CHECK(
      dim_order_to_stride(
          out_sizes.data(),
          out_dim_order.data(),
          out_sizes.size(),
          out_strides) == Error::Ok);

  CTYPE* const out_ptr = out.data_ptr<CTYPE>();
  const CTYPE* const in_ptr = in.data_ptr<CTYPE>();
  const CTYPE* const w_ptr = weight.data_ptr<CTYPE>();
  const CTYPE* const bias_ptr =
      bias.has_value() ? bias.value().data_ptr<CTYPE>() : nullptr;

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
            out_c);
      }
    }
  }
}

void get_conv_output_size(
    const Tensor& in,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    exec_aten::SizesType* sizes_arr,
    size_t& dim) {
  dim = in.dim();

  sizes_arr[0] = in.size(0);
  sizes_arr[1] = weight.size(0);
  for (size_t d = 2; d < in.dim(); ++d) {
    int64_t dilation_val = 1;
    if (dilation.size() > 1) {
      dilation_val = val_at(dilation, d - 2);
    }
    int64_t padding_val = val_at(padding, d - 2);
    int64_t stride_val = val_at(stride, d - 2);

    int64_t kernel_len = dilation_val * (weight.size(d) - 1) + 1;
    sizes_arr[d] =
        (in.size(d) + (2 * padding_val) - kernel_len) / stride_val + 1;
  }
}

void check_preconditions(
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
  ET_CHECK_SAME_DTYPE3(in, weight, out);

  ET_CHECK_DEFAULT_OR_CHANNELSLAST_DIMORDER(in);
  ET_CHECK_DEFAULT_OR_CHANNELSLAST_DIMORDER(weight);
  ET_CHECK_DEFAULT_OR_CHANNELSLAST_DIMORDER(out);

  ET_CHECK(in.dim() >= 3 && in.dim() < 5);
  ET_CHECK(in.dim() == weight.dim());
  ET_CHECK(in.dim() == out.dim());

  if (bias.has_value()) {
    ET_CHECK(bias.value().dim() == 1);
    ET_CHECK(bias.value().size(0) == weight.size(0));
  }

  ET_CHECK(padding.size() > 0 && padding.size() <= in.dim() - 2);
  ET_CHECK(stride.size() > 0 && stride.size() <= in.dim() - 2);
  if (dilation.size() > 0) {
    ET_CHECK(dilation.size() <= in.dim() - 2);
  }
  // input channels must be evenly divisible by groups
  ET_CHECK(in.size(1) % groups == 0);

  ET_CHECK_MSG(!transposed, "transposed convolution not supported yet!");
  if (output_padding.size() > 0) {
    ET_CHECK(dilation.size() <= in.dim() - 2);
  }
}

} // namespace

Tensor& convolution_out(
    RuntimeContext& ctx,
    const Tensor& in,
    const Tensor& weight,
    const exec_aten::optional<Tensor>& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    __ET_UNUSED bool transposed,
    __ET_UNUSED IntArrayRef output_padding,
    int64_t groups,
    Tensor& out) {
  (void)ctx;

  check_preconditions(
      in,
      weight,
      bias,
      stride,
      padding,
      dilation,
      transposed,
      output_padding,
      groups,
      out);

  size_t output_ndim = 0;
  exec_aten::SizesType output_sizes[kTensorDimensionLimit];
  get_conv_output_size(
      in, weight, stride, padding, dilation, output_sizes, output_ndim);

  Error err = resize_tensor(out, {output_sizes, output_ndim});
  ET_CHECK_MSG(err == Error::Ok, "Could not resize output");

  ET_SWITCH_REAL_TYPES(in.scalar_type(), ctx, "convolution", CTYPE, [&]() {
    convolution_wrapper<CTYPE>(
        in, weight, bias, stride, padding, dilation, groups, out);
  });

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
