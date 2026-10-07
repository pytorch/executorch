/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2025-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <climits>

#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>

#include "cortex_m_ops_common.h"

namespace cortex_m {
namespace native {

using KernelRuntimeContext = torch::executor::KernelRuntimeContext;

namespace {
constexpr int64_t kConvDim = 4;

bool validate_conv2d_arguments(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const Tensor& output,
    const Int64ArrayRef& stride,
    const Int64ArrayRef& padding,
    const Int64ArrayRef& dilation,
    const Tensor& requantize_multipliers,
    const Tensor& requantize_shifts,
    const Int64ArrayRef filter_shape,
    const bool packed_weights,
    ActivationLayout layout) {
  if (input.dim() != kConvDim ||
      weight.dim() != (packed_weights ? 1 : kConvDim) ||
      output.dim() != kConvDim) {
    ET_LOG(Error, "quantized_conv2d_out: invalid activation or weight rank");
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (layout == ActivationLayout::NHWCLogical) {
    if (!executorch::runtime::is_contiguous_dim_order(
            input.dim_order().data(), input.dim_order().size()) ||
        !executorch::runtime::is_contiguous_dim_order(
            output.dim_order().data(), output.dim_order().size())) {
      ET_LOG(
          Error,
          "quantized_conv2d_nhwc_out: input and output must have contiguous dim_order");
      context.fail(Error::InvalidArgument);
      return false;
    }
  } else if (
      !is_channels_last_tensor(input) || !is_channels_last_tensor(output)) {
    ET_LOG(
        Error,
        "quantized_conv2d_out: input and output must have channels_last dim_order");
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (input.scalar_type() != ScalarType::Char ||
      output.scalar_type() != ScalarType::Char) {
    ET_LOG(Error, "quantized_conv2d_out: input and output must be int8");
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (weight.scalar_type() != ScalarType::Char) {
    ET_LOG(Error, "quantized_conv2d_out: weight must be int8");
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (bias.has_value() && bias.value().scalar_type() != ScalarType::Int) {
    ET_LOG(Error, "quantized_conv2d_out: bias must be int32 if provided");
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (stride.size() != 2 || (padding.size() != 2 && padding.size() != 4) ||
      dilation.size() != 2) {
    ET_LOG(
        Error,
        "quantized_conv2d_out: stride/dilation must have length 2; padding must have length 2 or 4");
    context.fail(Error::InvalidArgument);
    return false;
  }

  const int64_t out_channels =
      output.size(layout == ActivationLayout::NHWCLogical ? 3 : 1);
  if (requantize_multipliers.dim() != 1 || requantize_shifts.dim() != 1 ||
      requantize_multipliers.scalar_type() != ScalarType::Int ||
      requantize_shifts.scalar_type() != ScalarType::Int ||
      requantize_multipliers.numel() != out_channels ||
      requantize_shifts.numel() != out_channels) {
    ET_LOG(
        Error,
        "quantized_conv2d_out: per-channel params must match output channels (%zd)",
        out_channels);
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (filter_shape.size() != kConvDim) {
    ET_LOG(Error, "Convolution filter_shape must have four dimensions");
    context.fail(Error::InvalidArgument);
    return false;
  }
  if (packed_weights) {
#if !ARM_NN_ENABLE_AOT || !defined(__ARM_FEATURE_MVE) || \
    !(__ARM_FEATURE_MVE & 1)
    ET_LOG(Error, "Packed convolution requires AOT-enabled CMSIS-NN and MVE");
    context.fail(Error::InvalidArgument);
    return false;
#else
    if (!bias.has_value() || bias->dim() != 1 ||
        bias->numel() != out_channels ||
        !executorch::runtime::is_contiguous_dim_order(
            weight.dim_order().data(), weight.dim_order().size())) {
      ET_LOG(Error, "Invalid packed convolution metadata or effective bias");
      context.fail(Error::InvalidArgument);
      return false;
    }
    if (std::any_of(filter_shape.begin(), filter_shape.end(), [](int64_t size) {
          return size <= 0 || size > INT32_MAX;
        })) {
      ET_LOG(Error, "Invalid logical convolution filter dimensions");
      context.fail(Error::InvalidArgument);
      return false;
    }
    const int64_t input_channels =
        input.size(layout == ActivationLayout::NHWCLogical ? 3 : 1);
    if (input_channels <= 0 || input_channels % filter_shape[3] != 0 ||
        filter_shape[0] != out_channels ||
        out_channels % (input_channels / filter_shape[3]) != 0) {
      ET_LOG(Error, "Invalid packed convolution channel dimensions");
      context.fail(Error::InvalidArgument);
      return false;
    }
    const cmsis_nn_dims dims{
        static_cast<int32_t>(filter_shape[0]),
        static_cast<int32_t>(filter_shape[1]),
        static_cast<int32_t>(filter_shape[2]),
        static_cast<int32_t>(filter_shape[3])};
    const int32_t packed_size = arm_convolve_s8_packed_get_filter_size(
        &dims, static_cast<int32_t>(input_channels / filter_shape[3]));
    if (packed_size <= 0 || weight.numel() != packed_size) {
      ET_LOG(Error, "Invalid packed convolution weight size");
      context.fail(Error::InvalidArgument);
      return false;
    }
#endif
  } else {
    for (size_t index = 0; index < kConvDim; ++index) {
      if (weight.size(index) != filter_shape[index]) {
        ET_LOG(Error, "Unpacked weights must match logical filter dimensions");
        context.fail(Error::InvalidArgument);
        return false;
      }
    }
  }

  return true;
}
} // namespace

// cppcheck-suppress unusedFunction
static Tensor& quantized_conv2d_out_impl(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const Int64ArrayRef stride,
    const Int64ArrayRef padding,
    const Int64ArrayRef dilation,
    const int64_t input_offset,
    const int64_t output_offset,
    const Tensor& requantize_multipliers,
    const Tensor& requantize_shifts,
    const int64_t activation_min,
    const int64_t activation_max,
    const Tensor& scratch,
    const std::optional<Int64ArrayRef> optional_filter_shape,
    const bool packed_weights,
    ActivationLayout layout,
    Tensor& out) {
  int64_t unpacked_filter_shape[kConvDim] = {};
  if (!optional_filter_shape.has_value()) {
    if (packed_weights) {
      ET_LOG(Error, "Packed convolution requires filter_shape");
      context.fail(Error::InvalidArgument);
      return out;
    }
    if (weight.dim() == kConvDim) {
      for (size_t index = 0; index < kConvDim; ++index) {
        unpacked_filter_shape[index] = weight.size(index);
      }
    }
  }
  const Int64ArrayRef filter_shape = optional_filter_shape.value_or(
      Int64ArrayRef(unpacked_filter_shape, kConvDim));
  if (!validate_conv2d_arguments(
          context,
          input,
          weight,
          bias,
          out,
          stride,
          padding,
          dilation,
          requantize_multipliers,
          requantize_shifts,
          filter_shape,
          packed_weights,
          layout)) {
    return out;
  }

  const int32_t batch = static_cast<int32_t>(input.size(0));
  const int32_t input_channels = static_cast<int32_t>(
      input.size(layout == ActivationLayout::NHWCLogical ? 3 : 1));
  const int32_t input_height = static_cast<int32_t>(
      input.size(layout == ActivationLayout::NHWCLogical ? 1 : 2));
  const int32_t input_width = static_cast<int32_t>(
      input.size(layout == ActivationLayout::NHWCLogical ? 2 : 3));

  const int32_t kernel_output_channels = static_cast<int32_t>(filter_shape[0]);
  const int32_t kernel_height = static_cast<int32_t>(filter_shape[1]);
  const int32_t kernel_width = static_cast<int32_t>(filter_shape[2]);
  const int32_t kernel_input_channels = static_cast<int32_t>(filter_shape[3]);

  const int32_t output_channels = static_cast<int32_t>(
      out.size(layout == ActivationLayout::NHWCLogical ? 3 : 1));
  const int32_t output_height = static_cast<int32_t>(
      out.size(layout == ActivationLayout::NHWCLogical ? 1 : 2));
  const int32_t output_width = static_cast<int32_t>(
      out.size(layout == ActivationLayout::NHWCLogical ? 2 : 3));

  const int32_t input_offset_val = static_cast<int32_t>(input_offset);
  const int32_t output_offset_val = static_cast<int32_t>(output_offset);
  const int32_t activation_min_val = static_cast<int32_t>(activation_min);
  const int32_t activation_max_val = static_cast<int32_t>(activation_max);

  const cmsis_nn_dims input_dims{
      batch, input_height, input_width, input_channels};
  const cmsis_nn_dims filter_dims{
      kernel_output_channels,
      kernel_height,
      kernel_width,
      kernel_input_channels};
  const cmsis_nn_dims output_dims{
      batch, output_height, output_width, output_channels};
  const cmsis_nn_dims bias_dims{1, 1, 1, output_channels};

  cmsis_nn_conv_params conv_params{};
  conv_params.input_offset = input_offset_val;
  conv_params.output_offset = output_offset_val;
  conv_params.stride.h = static_cast<const int32_t>(stride[0]);
  conv_params.stride.w = static_cast<const int32_t>(stride[1]);
  // Trailing padding is encoded in the planned output dimensions.
  conv_params.padding.h = static_cast<const int32_t>(padding[0]);
  conv_params.padding.w = static_cast<const int32_t>(padding[1]);
  conv_params.dilation.h = static_cast<const int32_t>(dilation[0]);
  conv_params.dilation.w = static_cast<const int32_t>(dilation[1]);
  conv_params.activation.min = activation_min_val;
  conv_params.activation.max = activation_max_val;

  cmsis_nn_per_channel_quant_params quant_params;
  quant_params.multiplier = requantize_multipliers.data_ptr<int32_t>();
  quant_params.shift = requantize_shifts.data_ptr<int32_t>();

  const int8_t* input_data = input.const_data_ptr<int8_t>();
  const int8_t* weight_data = weight.const_data_ptr<int8_t>();
  int8_t* output_data = out.mutable_data_ptr<int8_t>();
  const int32_t* bias_data =
      bias.has_value() ? bias.value().const_data_ptr<int32_t>() : nullptr;
#if ARM_NN_ENABLE_AOT
  if (packed_weights) {
    conv_params.filter_format = ARM_NN_FILTER_FORMAT_PACKED_4X16;
    conv_params.bias_format = ARM_NN_BIAS_FORMAT_EFFECTIVE;
  }
#endif

  cmsis_nn_context cmsis_context;
  cmsis_context.buf = nullptr;
  cmsis_context.size = scratch.nbytes();
  if (cmsis_context.size > 0) {
    cmsis_context.buf = scratch.mutable_data_ptr<int8_t>();
  }

#ifdef CORTEX_M_ENABLE_RUNTIME_CHECKS
  const int32_t runtime_buffer_bytes = arm_convolve_wrapper_s8_get_buffer_size(
      &conv_params, &input_dims, &filter_dims, &output_dims);
  if (runtime_buffer_bytes < 0) {
    ET_LOG(
        Error, "quantized_conv2d_out: CMSIS-NN buffer size calculation failed");
    context.fail(Error::Internal);
    return out;
  }
  if (scratch.nbytes() != static_cast<size_t>(runtime_buffer_bytes)) {
    ET_LOG(
        Error,
        "quantized_conv2d_out: scratch buffer size incorrect - actual: (%d) needed: (%d)",
        static_cast<int>(scratch.nbytes()),
        static_cast<int>(runtime_buffer_bytes));
    context.fail(Error::Internal);
    return out;
  }
#endif

  const arm_cmsis_nn_status status = arm_convolve_wrapper_s8(
      &cmsis_context,
      &conv_params,
      &quant_params,
      &input_dims,
      input_data,
      &filter_dims,
      weight_data,
      &bias_dims,
      bias_data,
      &output_dims,
      output_data);

  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(
        Error,
        "quantized_conv2d_out: arm_convolve_wrapper_s8 failed with status %d",
        status);
    context.fail(Error::Internal);
  }

  return out;
}

// cppcheck-suppress unusedFunction
Tensor& quantized_conv2d_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const Int64ArrayRef stride,
    const Int64ArrayRef padding,
    const Int64ArrayRef dilation,
    const int64_t input_offset,
    const int64_t output_offset,
    const Tensor& requantize_multipliers,
    const Tensor& requantize_shifts,
    const int64_t activation_min,
    const int64_t activation_max,
    const Tensor& scratch,
    const std::optional<Int64ArrayRef> filter_shape,
    const bool packed_weights,
    Tensor& out) {
  return quantized_conv2d_out_impl(
      context,
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      input_offset,
      output_offset,
      requantize_multipliers,
      requantize_shifts,
      activation_min,
      activation_max,
      scratch,
      filter_shape,
      packed_weights,
      ActivationLayout::NCHWLogical,
      out);
}

// cppcheck-suppress unusedFunction
Tensor& quantized_conv2d_nhwc_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const Int64ArrayRef stride,
    const Int64ArrayRef padding,
    const Int64ArrayRef dilation,
    const int64_t input_offset,
    const int64_t output_offset,
    const Tensor& requantize_multipliers,
    const Tensor& requantize_shifts,
    const int64_t activation_min,
    const int64_t activation_max,
    const Tensor& scratch,
    const std::optional<Int64ArrayRef> filter_shape,
    const bool packed_weights,
    Tensor& out) {
  return quantized_conv2d_out_impl(
      context,
      input,
      weight,
      bias,
      stride,
      padding,
      dilation,
      input_offset,
      output_offset,
      requantize_multipliers,
      requantize_shifts,
      activation_min,
      activation_max,
      scratch,
      filter_shape,
      packed_weights,
      ActivationLayout::NHWCLogical,
      out);
}

} // namespace native
} // namespace cortex_m
