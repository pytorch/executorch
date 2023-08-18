/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>

#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>

namespace torch {
namespace executor {

using Tensor = exec_aten::Tensor;

int64_t val_at(IntArrayRef array, size_t i, int64_t default_val) {
  if (array.size() == 1) {
    return array[0];
  } else if (array.size() > 1) {
    return array[i];
  } else {
    return default_val;
  }
}

bool int_array_all_ge(IntArrayRef array, int64_t val) {
  for (size_t i = 0; i < array.size(); ++i) {
    if (array[i] < val) {
      ET_LOG(
          Error,
          "Expected array[%zu] > %" PRId64 ", found %" PRId64,
          i,
          val,
          array[i]);
      return false;
    }
  }
  return true;
}

bool stride_is_valid(IntArrayRef stride, size_t kernel_ndim) {
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      stride.size() > 0 && stride.size() <= kernel_ndim,
      "Expected stride to have size between 1 and %zu inclusive "
      "but got %zd",
      kernel_ndim,
      stride.size());
  ET_LOG_AND_RETURN_IF_FALSE(int_array_all_ge(stride, 1));
  return true;
}

bool padding_is_valid(
    IntArrayRef padding,
    IntArrayRef kernel_size,
    size_t kernel_ndim,
    bool enforce_half_kernel) {
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      padding.size() > 0 && padding.size() <= kernel_ndim,
      "Expected padding to have size between 1 and %zu inclusive "
      "but got %zd",
      kernel_ndim,
      padding.size());
  ET_LOG_AND_RETURN_IF_FALSE(int_array_all_ge(padding, 0));

  if (enforce_half_kernel) {
    // Padding must be at most half of kernel size.
    for (size_t i = 0; i < padding.size(); i++) {
      if (padding[i] > val_at(kernel_size, i) / 2) {
        ET_LOG(
            Error,
            "Padding should be at most half of kernel size, "
            "but got padding[%zu] = %" PRId64 " > kernel_size[%zu] = %" PRId64,
            i,
            padding[i],
            i,
            val_at(kernel_size, i));
        return false;
      }
    }
  }
  return true;
}

bool dilation_is_valid(IntArrayRef dilation, size_t kernel_ndim) {
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      dilation.size() > 0 && dilation.size() <= kernel_ndim,
      "Expected dilation to have size between 1 and %zu inclusive "
      "but got %zd",
      kernel_ndim,
      dilation.size());
  ET_LOG_AND_RETURN_IF_FALSE(int_array_all_ge(dilation, 1));
  return true;
}

bool output_size_is_valid(
    exec_aten::ArrayRef<exec_aten::SizesType> output_size) {
  bool valid = true;
  for (size_t i = 0; i < output_size.size(); i++) {
    if (output_size[i] <= 0) {
      valid = false;
    }
  }
  if (!valid) {
    ET_LOG(
        Error,
        "The provided combination of input and kernel parameters "
        "produces an invalid output size:");
    for (size_t d = 0; d < output_size.size(); ++d) {
      ET_LOG(
          Error, "    size(%zu): %zu", d, static_cast<size_t>(output_size[d]));
    }
  }
  return valid;
}

void get_unsqueezed_sizes(
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

void get_unsqueezed_dim_order(
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

void calculate_kernel_output_sizes(
    const Tensor& in,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    exec_aten::SizesType* out_sizes,
    bool ceil_mode) {
  size_t dim_offset = in.dim() - kernel_size.size();
  for (size_t d = 0; d < kernel_size.size(); ++d) {
    int64_t dilation_val = 1;
    if (dilation.size() > 1) {
      dilation_val = val_at(dilation, d);
    }
    int64_t padding_val = val_at(padding, d, /*default=*/0);
    int64_t stride_val = val_at(stride, d);

    int64_t kernel_len = dilation_val * (val_at(kernel_size, d) - 1) + 1;
    if (ceil_mode) {
      out_sizes[d + dim_offset] =
          std::ceil(
              static_cast<float>(
                  in.size(d + dim_offset) + (2 * padding_val) - kernel_len) /
              static_cast<float>(stride_val)) +
          1;
    } else {
      out_sizes[d + dim_offset] =
          (in.size(d + dim_offset) + (2 * padding_val) - kernel_len) /
              stride_val +
          1;
    }
  }
}

bool check_convolution_args(
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
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, weight, out));

  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_default_or_channels_last_dim_order(in));
  ET_LOG_AND_RETURN_IF_FALSE(
      tensor_is_default_or_channels_last_dim_order(weight));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_default_or_channels_last_dim_order(out));

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      in.dim() == 3 || in.dim() == 4,
      "Expect input tensor to be 3-D or 4-D, but got, %zu.",
      static_cast<size_t>(in.dim()));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(weight, in.dim()));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(out, in.dim()));

  if (bias.has_value()) {
    ET_LOG_AND_RETURN_IF_FALSE(tensor_is_rank(bias.value(), 1));
    ET_LOG_AND_RETURN_IF_FALSE(
        tensors_have_same_size_at_dims(bias.value(), 0, weight, 0));
  }

  int64_t kernel_size[2];
  size_t kernel_ndim = 2;
  if (weight.dim() == 3) {
    kernel_size[0] = weight.size(2);
    kernel_ndim = 1;
  } else {
    kernel_size[0] = weight.size(2);
    kernel_size[1] = weight.size(3);
  }
  ET_LOG_AND_RETURN_IF_FALSE(stride_is_valid(stride, kernel_ndim));
  ET_LOG_AND_RETURN_IF_FALSE(
      padding_is_valid(padding, {kernel_size, kernel_ndim}, kernel_ndim));
  if (dilation.size() > 0) {
    ET_LOG_AND_RETURN_IF_FALSE(dilation_is_valid(dilation, kernel_ndim));
  }

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      in.size(1) % groups == 0,
      "groups %" PRId64 " is not divisible by in.size(1) = %zd",
      groups,
      in.size(1));

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      !transposed, "transposed convolution not supported yet.");

  return true;
}

void get_convolution_out_target_size(
    const Tensor& in,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    exec_aten::SizesType* out_sizes,
    size_t* out_ndim) {
  *out_ndim = in.dim();

  out_sizes[0] = in.size(0);
  out_sizes[1] = weight.size(0);

  int64_t kernel_size[2];
  size_t kernel_ndim = 2;
  if (weight.dim() == 3) {
    kernel_size[0] = weight.size(2);
    kernel_ndim = 1;
  } else {
    kernel_size[0] = weight.size(2);
    kernel_size[1] = weight.size(3);
  }
  calculate_kernel_output_sizes(
      in, {kernel_size, kernel_ndim}, stride, padding, dilation, out_sizes);
}

} // namespace executor
} // namespace torch
