/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstring>

#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

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

bool output_padding_is_valid(
    IntArrayRef output_padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    size_t kernel_ndim) {
  ET_LOG_AND_RETURN_IF_FALSE(param_array_is_valid(
      "output_padding",
      output_padding,
      /*min_val=*/0,
      kernel_ndim,
      /*allow_empty=*/false));

  for (size_t i = 0; i < kernel_ndim; i++) {
    const int64_t op_i = val_at(output_padding, i);
    const int64_t s_i = val_at(stride, i);
    const int64_t d_i = val_at(dilation, i);
    ET_LOG_MSG_AND_RETURN_IF_FALSE(
        op_i < s_i || op_i < d_i,
        "output padding must be smaller than either stride or dilation");
  }
  return true;
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
    bool ceil_mode,
    bool transposed,
    int64_t output_padding) {
  if (transposed) {
    return (inputSize - 1) * stride - 2 * pad + dilation * (kernelSize - 1) +
        output_padding + 1;
  }
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
    bool ceil_mode,
    bool transposed,
    IntArrayRef output_padding) {
  for (size_t i = 0; i < kernel_ndim; ++i) {
    auto dim = in.dim() - (kernel_ndim - i);
    int64_t k = val_at(kernel_size, i);
    int64_t s = val_at(stride, i, /*default_value=*/k);
    int64_t d = val_at(dilation, i, /*default_value=*/1);
    int64_t p = val_at(padding, i, /*default_value=*/0);
    int64_t op =
        transposed ? val_at(output_padding, i, /*default_value=*/0) : 0;

    out_sizes[dim] = _kernel_output_size_helper(
        in.size(dim), k, p, s, d, ceil_mode, transposed, op);
  }
}

bool check_arange_args(double start, double end, double step, Tensor& out) {
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      out.dim() == 1,
      "out should be a 1-d tensor, but got a %zu-d tensor",
      out.dim());

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      (step > 0 && (end >= start)) || (step < 0 && (end <= start)),
      "upper bound and larger bound inconsistent with step sign");

  return true;
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
    const Tensor& out) {
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
    ET_LOG_MSG_AND_RETURN_IF_FALSE(
        bias.value().size(0) == transposed ? groups * weight.size(1)
                                           : weight.size(0),
        "bias length must equal number of output channels, but got %zd",
        bias.value().size(0));
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
  if (transposed) {
    ET_LOG_AND_RETURN_IF_FALSE(
        output_padding_is_valid(output_padding, stride, dilation, kernel_ndim));
  }

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      weight.size(0) >= groups,
      "Given groups=%" PRId64 ", expected weight to be at least %" PRId64
      " at dimension 0, but got weight.size(0) = %zd instead",
      groups,
      groups,
      weight.size(0));
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      weight.size(0) % groups == 0,
      "Given groups=%" PRId64 ", expected weight to be divisible by %" PRId64
      " at dimension 0, but got weight.size(0) = %zd instead",
      groups,
      groups,
      weight.size(0));

  if (!transposed) {
    ET_LOG_MSG_AND_RETURN_IF_FALSE(
        in.size(1) == groups * weight.size(1),
        "Given groups=%" PRId64
        " and weight.size(1) = %zd, expected input to have %" PRId64
        " channels, but got %zd",
        groups,
        weight.size(1),
        groups * weight.size(1),
        in.size(1));
  } else {
    ET_LOG_MSG_AND_RETURN_IF_FALSE(
        in.size(1) == weight.size(0),
        "input channels must match weight.size(0) in transposed convolution");
  }

  return true;
}

void get_convolution_out_target_size(
    const Tensor& in,
    const Tensor& weight,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    bool transposed,
    IntArrayRef output_padding,
    int64_t groups,
    exec_aten::SizesType* out_sizes,
    size_t* out_ndim) {
  *out_ndim = in.dim();

  // batch dim
  out_sizes[0] = in.size(0);

  // channel dim
  if (!transposed) {
    out_sizes[1] = in.size(1) == 0 ? 0 : weight.size(0);
  } else {
    out_sizes[1] = in.size(1) == 0 ? 0 : groups * weight.size(1);
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
  calculate_kernel_output_sizes(
      in,
      kernel_ndim,
      {kernel_size, kernel_ndim},
      stride,
      padding,
      dilation,
      out_sizes,
      false,
      transposed,
      output_padding);
}

bool check_cumsum_args(
    const Tensor& in,
    int64_t dim,
    optional<ScalarType> dtype,
    Tensor& out) {
  ET_LOG_AND_RETURN_IF_FALSE(dim_is_valid(dim, in.dim()));

  if (dtype.has_value()) {
    ET_LOG_AND_RETURN_IF_FALSE(dtype.value() == out.scalar_type());
  }

  return true;
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

bool check_masked_fill_args(
    const Tensor& in,
    const Tensor& mask,
    const Scalar& value,
    Tensor& out) {
  (void)value;

  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));
  ET_LOG_AND_RETURN_IF_FALSE(mask.scalar_type() == ScalarType::Bool);

  return true;
}

bool check_constant_pad_args(
    const Tensor& in,
    IntArrayRef pad,
    const Scalar& value,
    Tensor& out) {
  (void)value;

  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(in, out));

  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_rank(in, out));

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      pad.size() % 2 == 0, "Padding array must be a multiple of 2");

  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      pad.size() / 2 <= in.dim(), "Padding array contains too many elements");

  return true;
}

Error resize_constant_pad_output(
    const Tensor& in,
    IntArrayRef pad,
    Tensor& out) {
  Tensor::SizesType expected_output_size[kTensorDimensionLimit];

  int pad_i = in.dim() - 1;
  for (size_t i = 0; i < in.dim(); ++i, --pad_i) {
    expected_output_size[i] = in.size(i);
    if (pad_i >= 0 && pad_i < pad.size() / 2) {
      expected_output_size[i] += pad[2 * pad_i] + pad[2 * pad_i + 1];
    }
  }

  ArrayRef<Tensor::SizesType> output_size{
      expected_output_size, static_cast<size_t>(in.dim())};
  auto error = resize_tensor(out, output_size);

  return error;
}

bool check_embedding_args(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& out) {
  // Ensure weight is 2-D. It could be empty.
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      weight.dim() == 2, "weight.dim() %zd != 2", weight.dim());

  // Ensure out is k+1 dimension tensor where k is the indices.dim()
  // out's first k dimension shall be same as indices, and the last dim shall
  // equal weight's last dim
  ET_LOG_MSG_AND_RETURN_IF_FALSE(
      out.dim() == indices.dim() + 1,
      "out.dim() %zd != indices.dim() %zd + 1",
      out.dim(),
      indices.dim());

  // Ensure dtype is the same for out and weight
  ET_LOG_AND_RETURN_IF_FALSE(tensors_have_same_dtype(weight, out));

  return true;
}

Error resize_embedding_output(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& out) {
  Tensor::SizesType expected_output_size[kTensorDimensionLimit];
  for (size_t i = 0; i < indices.dim(); i++) {
    expected_output_size[i] = indices.size(i);
  }
  const size_t embedding_dim = weight.size(1);
  expected_output_size[out.dim() - 1] = embedding_dim;

  ArrayRef<Tensor::SizesType> output_size{
      expected_output_size, static_cast<size_t>(out.dim())};

  return resize_tensor(out, output_size);
}

bool check_alpha_type(
    const ScalarType alpha_type,
    const ScalarType common_type) {
  // Verify that alpha type is compatible with common type,
  // as used by ops such as add and sub.
  ET_LOG_AND_RETURN_IF_FALSE(
      canCast(alpha_type, common_type) ||
      (common_type == ScalarType::Bool && isIntegralType(alpha_type, true)));

  return true;
}

} // namespace executor
} // namespace torch
