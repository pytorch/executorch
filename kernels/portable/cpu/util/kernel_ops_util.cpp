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

namespace {

bool param_array_is_valid(
    const char* name,
    IntArrayRef array,
    int64_t min_val,
    size_t length,
    bool allow_empty) {
  auto size = array.size();
  if (allow_empty) {
    ET_LOG_MSG_AND_RETURN_IF_FALSE(
        size == 0 || size == 1 || size == length,
        "Expected %s to have size 0, 1 or %zu but got %zd",
        name,
        length,
        size);
  } else {
    ET_LOG_MSG_AND_RETURN_IF_FALSE(
        size == 1 || size == length,
        "Expected %s to have size 1 or %zu but got %zd",
        name,
        length,
        size);
  }
  ET_LOG_AND_RETURN_IF_FALSE(int_array_all_ge(array, min_val));
  return true;
}

} // namespace

int64_t val_at(IntArrayRef array, size_t i, int64_t default_value) {
  if (array.size() == 1) {
    return array[0];
  } else if (array.size() > 1) {
    return array[i];
  } else {
    return default_value;
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

bool kernel_size_is_valid(IntArrayRef kernel_size, size_t kernel_ndim) {
  return param_array_is_valid(
      "kernel_size",
      kernel_size,
      /*min_val=*/1,
      kernel_ndim,
      /*allow_empty=*/false);
}

bool stride_is_valid(IntArrayRef stride, size_t kernel_ndim, bool allow_empty) {
  return param_array_is_valid(
      "stride", stride, /*min_val=*/1, kernel_ndim, allow_empty);
}

bool padding_is_valid(
    IntArrayRef padding,
    IntArrayRef kernel_size,
    size_t kernel_ndim,
    bool enforce_half_kernel) {
  bool valid = param_array_is_valid(
      "padding", padding, /*min_val=*/0, kernel_ndim, /*allow_empty=*/false);
  if (!valid) {
    return false;
  }

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
  return param_array_is_valid(
      "dilation", dilation, /*min_val=*/1, kernel_ndim, /*allow_empty=*/false);
}

bool output_size_is_valid(
    exec_aten::ArrayRef<exec_aten::SizesType> output_size,
    size_t kernel_ndim) {
  bool valid = true;
  size_t out_dim = output_size.size();
  for (size_t i = 0; i < out_dim - kernel_ndim; i++) {
    if (output_size[i] < 0) {
      valid = false;
    }
  }
  for (size_t i = out_dim - kernel_ndim; i < out_dim; i++) {
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

int64_t _kernel_output_size_helper(
    size_t inputSize,
    int64_t kernelSize,
    int64_t pad,
    int64_t stride,
    int64_t dilation,
    bool ceil_mode) {
  int64_t numerator = inputSize + 2 * pad - dilation * (kernelSize - 1) - 1 +
      (ceil_mode ? stride - 1 : 0);
  int64_t outputSize = numerator / stride + 1;
  if (ceil_mode) {
    // ensure that the last pooling starts inside the image
    // needed to avoid problems in ceil mode
    if ((outputSize - 1) * stride >= inputSize + pad) {
      --outputSize;
    }
  }
  return outputSize;
}

void calculate_kernel_output_sizes(
    const Tensor& in,
    size_t kernel_ndim,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    exec_aten::SizesType* out_sizes,
    bool ceil_mode) {
  for (size_t i = 0; i < kernel_ndim; ++i) {
    auto dim = in.dim() - (kernel_ndim - i);
    int64_t k = val_at(kernel_size, i);
    int64_t s = val_at(stride, i, /*default_value=*/k);
    int64_t d = val_at(dilation, i, /*default_value=*/1);
    int64_t p = val_at(padding, i, /*default_value=*/0);

    out_sizes[dim] =
        _kernel_output_size_helper(in.size(dim), k, p, s, d, ceil_mode);
  }
}

bool check_avg_pool2d_args(
    const Tensor& in,
    const IntArrayRef kernel_size,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const bool ceil_mode,
    const bool count_include_pad,
    const exec_aten::optional<int64_t>& divisor_override,
    const Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));

  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_default_or_channels_last_dim_order(in));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_default_or_channels_last_dim_order(out));

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      (in.dim() == 3 && in.size(0) > 0 && in.size(1) > 0 && in.size(2) > 0) ||
          (in.dim() == 4 && in.size(1) > 0 && in.size(2) > 0 && in.size(3) > 0),
      "Expected 3D or 4D (batch mode) tensor with optional 0 dim batch size for input");

  ET_LOG_AND_RETURN_IF_FALSE(
      kernel_size_is_valid(kernel_size, /*kernel_ndim=*/2));
  ET_LOG_AND_RETURN_IF_FALSE(
      stride_is_valid(kernel_size, /*kernel_ndim=*/2, /*allow_empty=*/true));
  ET_LOG_AND_RETURN_IF_FALSE(padding_is_valid(
      padding, kernel_size, /*kernel_ndim=*/2, /*enforce_half_kernel=*/true));

  if (divisor_override.has_value()) {
    ET_LOG_MSG_AND_RETURN_IF_FALSE(
        divisor_override.value() != 0,
        "divisor_override must be non-zero, but found %" PRId64,
        divisor_override.value());
  }

  return true;
}

void get_avg_pool2d_out_target_size(
    const Tensor& in,
    const IntArrayRef kernel_size,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const bool ceil_mode,
    exec_aten::SizesType* const out_sizes,
    size_t* const out_ndim) {
  *out_ndim = in.dim();

  // Batch dim is optional, so in can be either 3 or 4 dim.
  if (in.dim() == 4) {
    out_sizes[0] = in.size(0);
    out_sizes[1] = in.size(1);
  } else {
    out_sizes[0] = in.size(0);
  }

  calculate_kernel_output_sizes(
      in, 2, kernel_size, stride, padding, {}, out_sizes, ceil_mode);
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
  ET_LOG_AND_RETURN_IF_FALSE(
      stride_is_valid(stride, kernel_ndim, /*allow_empty=*/false));
  ET_LOG_AND_RETURN_IF_FALSE(
      padding_is_valid(padding, {kernel_size, kernel_ndim}, kernel_ndim));
  ET_LOG_AND_RETURN_IF_FALSE(dilation_is_valid(dilation, kernel_ndim));

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
  out_sizes[1] = in.size(1) == 0 ? 0 : weight.size(0);

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
      in,
      kernel_ndim,
      {kernel_size, kernel_ndim},
      stride,
      padding,
      dilation,
      out_sizes,
      false);
}

bool check_max_pool2d_with_indices_args(
    const Tensor& in,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    Tensor& out,
    Tensor& indices) {
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      indices.scalar_type() == ScalarType::Long,
      "Expected indices to have type of Long, but found %s",
      toString(indices.scalar_type()));

  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_default_or_channels_last_dim_order(in));
  ET_LOG_AND_RETURN_IF_FALSE(tensor_is_default_or_channels_last_dim_order(out));

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      (in.dim() == 3 && in.size(0) > 0 && in.size(1) > 0 && in.size(2) > 0) ||
          (in.dim() == 4 && in.size(1) > 0 && in.size(2) > 0 && in.size(3) > 0),
      "Expected 3D or 4D (batch mode) tensor with optional 0 dim batch size for input");

  ET_LOG_AND_RETURN_IF_FALSE(
      kernel_size_is_valid(kernel_size, /*kernel_ndim=*/2));
  ET_LOG_AND_RETURN_IF_FALSE(
      stride_is_valid(kernel_size, /*kernel_ndim=*/2, /*allow_empty=*/true));
  ET_LOG_AND_RETURN_IF_FALSE(padding_is_valid(
      padding, kernel_size, /*kernel_ndim=*/2, /*enforce_half_kernel=*/true));
  ET_LOG_AND_RETURN_IF_FALSE(dilation_is_valid(kernel_size, /*kernel_ndim=*/2));

  return true;
}

void get_max_pool2d_with_indices_out_target_size(
    const Tensor& in,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool ceil_mode,
    exec_aten::SizesType* out_sizes,
    size_t* out_ndim) {
  *out_ndim = in.dim();

  // Batch dim is optional, so in can be either 3 or 4 dim.
  if (in.dim() == 4) {
    out_sizes[0] = in.size(0);
    out_sizes[1] = in.size(1);
  } else {
    out_sizes[0] = in.size(0);
  }

  calculate_kernel_output_sizes(
      in, 2, kernel_size, stride, padding, dilation, out_sizes, ceil_mode);
}

} // namespace executor
} // namespace torch
