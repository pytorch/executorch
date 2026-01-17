/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cortex_m_ops_common.h"

extern "C" {
#include "arm_nnfunctions.h"
}

namespace cortex_m {
namespace native {

using KernelRuntimeContext = torch::executor::KernelRuntimeContext;

namespace {
constexpr int64_t kConvTransposeDim = 4;

bool validate_transpose_conv2d_arguments(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weight,
    const torch::executor::optional<Tensor>& bias,
    const Tensor& output,
    const IntArrayRef& stride,
    const IntArrayRef& padding,
    const IntArrayRef& output_padding,
    const IntArrayRef& dilation,
    const Tensor& requantize_multipliers,
    const Tensor& requantize_shifts) {
  if (input.dim() != kConvTransposeDim || weight.dim() != kConvTransposeDim ||
      output.dim() != kConvTransposeDim) {
    ET_LOG(Error, "quantized_transpose_conv2d_out: tensors must be 4-D");
    context.fail(Error::InvalidArgument);
    return false;
  }

  // Check for channels_last dim_order (NHWC: 0, 2, 3, 1)
  // Skip check if channels == 1, as dim_order is ambiguous in that case
  constexpr executorch::aten::DimOrderType kChannelsLastDimOrder[] = {
      0, 2, 3, 1};
  executorch::aten::ArrayRef<executorch::aten::DimOrderType>
      channels_last_order(kChannelsLastDimOrder, 4);

  if (input.size(1) > 1 && input.dim_order() != channels_last_order) {
    ET_LOG(
        Error,
        "quantized_transpose_conv2d_out: input must have channels_last dim_order (NHWC)");
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (output.size(1) > 1 && output.dim_order() != channels_last_order) {
    ET_LOG(
        Error,
        "quantized_transpose_conv2d_out: output must have channels_last dim_order (NHWC)");
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (input.scalar_type() != ScalarType::Char ||
      output.scalar_type() != ScalarType::Char) {
    ET_LOG(
        Error, "quantized_transpose_conv2d_out: input and output must be int8");
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (weight.scalar_type() != ScalarType::Char) {
    ET_LOG(Error, "quantized_transpose_conv2d_out: weight must be int8");
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (bias.has_value() && bias.value().scalar_type() != ScalarType::Int) {
    ET_LOG(
        Error,
        "quantized_transpose_conv2d_out: bias must be int32 if provided");
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (stride.size() != 2 || padding.size() != 2 || output_padding.size() != 2 ||
      dilation.size() != 2) {
    ET_LOG(
        Error,
        "quantized_transpose_conv2d_out: stride, padding, output_padding, and dilation must have length 2");
    context.fail(Error::InvalidArgument);
    return false;
  }

  // Reject non-zero output_padding - CMSIS-NN does not support this parameter
  if (output_padding[0] != 0 || output_padding[1] != 0) {
    ET_LOG(
        Error,
        "quantized_transpose_conv2d_out: output_padding is not supported by CMSIS-NN. "
        "Only output_padding=(0,0) is allowed. Got output_padding=(%lld,%lld)",
        static_cast<long long>(output_padding[0]),
        static_cast<long long>(output_padding[1]));
    context.fail(Error::NotSupported);
    return false;
  }

  // Reject dilation != (1,1) - CMSIS-NN produces different results than PyTorch
  if (dilation[0] != 1 || dilation[1] != 1) {
    ET_LOG(
        Error,
        "quantized_transpose_conv2d_out: dilation != (1,1) produces incorrect results with CMSIS-NN. "
        "Only dilation=(1,1) is supported. Got dilation=(%lld,%lld)",
        static_cast<long long>(dilation[0]),
        static_cast<long long>(dilation[1]));
    context.fail(Error::NotSupported);
    return false;
  }

  // Reject grouped convolutions - CMSIS-NN does not support groups > 1
  if (input.size(1) != weight.size(3)) {
    ET_LOG(
        Error,
        "quantized_transpose_conv2d_out: grouped convolutions not supported");
    context.fail(Error::NotSupported);
    return false;
  }

  const int64_t out_channels = output.size(1);
  if (requantize_multipliers.size(0) != out_channels ||
      requantize_shifts.size(0) != out_channels) {
    ET_LOG(
        Error,
        "quantized_transpose_conv2d_out: per-channel params must match output channels (%zd)",
        out_channels);
    context.fail(Error::InvalidArgument);
    return false;
  }

  return true;
}
} // namespace

Tensor& quantized_transpose_conv2d_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weight,
    const torch::executor::optional<Tensor>& bias,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef output_padding,
    const IntArrayRef dilation,
    const int64_t input_offset,
    const int64_t output_offset,
    const Tensor& requantize_multipliers,
    const Tensor& requantize_shifts,
    const int64_t activation_min,
    const int64_t activation_max,
    Tensor& out) {
  if (!validate_transpose_conv2d_arguments(
          context,
          input,
          weight,
          bias,
          out,
          stride,
          padding,
          output_padding,
          dilation,
          requantize_multipliers,
          requantize_shifts)) {
    return out;
  }

  const int32_t batch = static_cast<int32_t>(input.size(0));
  const int32_t input_channels = static_cast<int32_t>(input.size(1));
  const int32_t input_height = static_cast<int32_t>(input.size(2));
  const int32_t input_width = static_cast<int32_t>(input.size(3));

  const int32_t kernel_output_channels = static_cast<int32_t>(weight.size(0));
  const int32_t kernel_height = static_cast<int32_t>(weight.size(1));
  const int32_t kernel_width = static_cast<int32_t>(weight.size(2));
  const int32_t kernel_input_channels = static_cast<int32_t>(weight.size(3));

  const int32_t output_channels = static_cast<int32_t>(out.size(1));
  const int32_t output_height = static_cast<int32_t>(out.size(2));
  const int32_t output_width = static_cast<int32_t>(out.size(3));

  if (kernel_output_channels != output_channels) {
    ET_LOG(
        Error,
        "quantized_transpose_conv2d_out: weight output channels (%d) != output channels (%d)",
        kernel_output_channels,
        output_channels);
    context.fail(Error::InvalidArgument);
    return out;
  }

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

  // Setup transposed convolution parameters
  cmsis_nn_transpose_conv_params transpose_conv_params;
  transpose_conv_params.input_offset = input_offset_val;
  transpose_conv_params.output_offset = output_offset_val;
  transpose_conv_params.stride.h = static_cast<const int32_t>(stride[0]);
  transpose_conv_params.stride.w = static_cast<const int32_t>(stride[1]);
  transpose_conv_params.padding.h = static_cast<const int32_t>(padding[0]);
  transpose_conv_params.padding.w = static_cast<const int32_t>(padding[1]);
  // padding_offsets corresponds to output_padding in PyTorch
  transpose_conv_params.padding_offsets.h =
      static_cast<const int32_t>(output_padding[0]);
  transpose_conv_params.padding_offsets.w =
      static_cast<const int32_t>(output_padding[1]);
  transpose_conv_params.dilation.h = static_cast<const int32_t>(dilation[0]);
  transpose_conv_params.dilation.w = static_cast<const int32_t>(dilation[1]);
  transpose_conv_params.activation.min = activation_min_val;
  transpose_conv_params.activation.max = activation_max_val;

  cmsis_nn_per_channel_quant_params quant_params;
  quant_params.multiplier = requantize_multipliers.data_ptr<int32_t>();
  quant_params.shift = requantize_shifts.data_ptr<int32_t>();

  const int8_t* input_data = input.const_data_ptr<int8_t>();
  const int8_t* weight_data = weight.const_data_ptr<int8_t>();
  int8_t* output_data = out.mutable_data_ptr<int8_t>();
  const int32_t* bias_data =
      bias.has_value() ? bias.value().const_data_ptr<int32_t>() : nullptr;

  cmsis_nn_context cmsis_context;
  cmsis_context.buf = nullptr;
  cmsis_context.size = 0;

  cmsis_nn_context output_context;
  output_context.buf = nullptr;
  output_context.size = 0;

  const int32_t buffer_bytes = arm_transpose_conv_s8_get_buffer_size(
      &transpose_conv_params, &input_dims, &filter_dims, &output_dims);
  auto buffer_or_error = context.allocate_temp(
      static_cast<size_t>(buffer_bytes), alignof(int16_t));
  if (!buffer_or_error.ok()) {
    ET_LOG(
        Error,
        "quantized_transpose_conv2d_out: failed to allocate scratch buffer (%d bytes, error %d)",
        buffer_bytes,
        static_cast<int>(buffer_or_error.error()));
    context.fail(buffer_or_error.error());
    return out;
  }
  cmsis_context.buf = buffer_or_error.get();
  cmsis_context.size = buffer_bytes;

  const int32_t output_buffer_bytes =
      arm_transpose_conv_s8_get_reverse_conv_buffer_size(
          &transpose_conv_params, &input_dims, &filter_dims);
  auto output_buffer_or_error = context.allocate_temp(
      static_cast<size_t>(output_buffer_bytes), alignof(int16_t));
  if (!output_buffer_or_error.ok()) {
    ET_LOG(
        Error,
        "quantized_transpose_conv2d_out: failed to allocate output scratch buffer (%d bytes, error %d)",
        output_buffer_bytes,
        static_cast<int>(output_buffer_or_error.error()));
    context.fail(output_buffer_or_error.error());
    return out;
  }
  output_context.buf = output_buffer_or_error.get();
  output_context.size = output_buffer_bytes;

  const arm_cmsis_nn_status status = arm_transpose_conv_wrapper_s8(
      &cmsis_context,
      &output_context,
      &transpose_conv_params,
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
        "quantized_transpose_conv2d_out: arm_transpose_conv_wrapper_s8 failed with status %d",
        status);
    context.fail(Error::Internal);
  }

  return out;
}

} // namespace native
} // namespace cortex_m
