/*
 * Copyright 2025-2026 Arm Limited and/or its affiliates.
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
constexpr int64_t kConvDim = 4;

bool validate_conv2d_arguments(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weight,
    const torch::executor::optional<Tensor>& bias,
    const Tensor& output,
    const IntArrayRef& stride,
    const IntArrayRef& padding,
    const IntArrayRef& dilation,
    const Tensor& requantize_multipliers,
    const Tensor& requantize_shifts) {
  if (input.dim() != kConvDim || weight.dim() != kConvDim ||
      output.dim() != kConvDim) {
    ET_LOG(Error, "quantized_conv2d_out: tensors must be 4-D");
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
        "quantized_conv2d_out: input must have channels_last dim_order (NHWC)");
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (output.size(1) > 1 && output.dim_order() != channels_last_order) {
    ET_LOG(
        Error,
        "quantized_conv2d_out: output must have channels_last dim_order (NHWC)");
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

  if (stride.size() != 2 || padding.size() != 2 || dilation.size() != 2) {
    ET_LOG(
        Error,
        "quantized_conv2d_out: stride, padding, and dilation must have length 2");
    context.fail(Error::InvalidArgument);
    return false;
  }

  const int64_t out_channels = output.size(1);
  if (requantize_multipliers.size(0) != out_channels ||
      requantize_shifts.size(0) != out_channels) {
    ET_LOG(
        Error,
        "quantized_conv2d_out: per-channel params must match output channels (%zd)",
        out_channels);
    context.fail(Error::InvalidArgument);
    return false;
  }

  return true;
}
} // namespace

Tensor& quantized_conv2d_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& scratch,
    const Tensor& weight,
    const torch::executor::optional<Tensor>& bias,
    const IntArrayRef stride,
    const IntArrayRef padding,
    const IntArrayRef dilation,
    const int64_t input_offset,
    const int64_t output_offset,
    const Tensor& requantize_multipliers,
    const Tensor& requantize_shifts,
    const int64_t activation_min,
    const int64_t activation_max,
    Tensor& out) {
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
  const cmsis_nn_dims upscale_dims{1, 1, 1, 1};

  cmsis_nn_conv_params conv_params;
  conv_params.input_offset = input_offset_val;
  conv_params.output_offset = output_offset_val;
  conv_params.stride.h = static_cast<const int32_t>(stride[0]);
  conv_params.stride.w = static_cast<const int32_t>(stride[1]);
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

  cmsis_nn_context cmsis_context;
  cmsis_context.buf = nullptr;
  cmsis_context.size = 0;

  const size_t buffer_bytes = static_cast<size_t>(
      arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims));

#if 1
  cmsis_context.buf = scratch.mutable_data_ptr<int8_t>();
  cmsis_context.size = buffer_bytes;

  if (scratch.nbytes() != buffer_bytes) {
    ET_LOG(
        Error,
        "quantized_dw_conv2d_out: scratch buffer size incorrect - actual: (%d) needed: (%d)",
        static_cast<int>(scratch.nbytes()),
        static_cast<int>(buffer_bytes));
    return out;
  }

#else

  if (buffer_bytes > 0) {
    auto buffer_or_error =
        context.allocate_temp(buffer_bytes, alignof(int16_t));
    if (!buffer_or_error.ok()) {
      if (buffer_or_error.error() != Error::NotFound) {
        ET_LOG(
            Error,
            "quantized_conv2d_out: failed to allocate scratch buffer (%d)",
            static_cast<int>(buffer_or_error.error()));
        context.fail(buffer_or_error.error());
        return out;
      }
    } else {
      cmsis_context.buf = buffer_or_error.get();
      cmsis_context.size = buffer_bytes;
    }
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
        "quantized_conv2d_out: arm_convolve_s8 failed with status %d",
        status);
    context.fail(Error::Internal);
  }

  return out;
}

} // namespace native
} // namespace cortex_m
