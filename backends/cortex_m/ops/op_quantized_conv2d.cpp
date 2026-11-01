/*
 * Copyright 2025-2026 Arm Limited and/or its affiliates.
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/core/exec_aten/util/dim_order_util.h>

#include <algorithm>

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
    ActivationLayout layout) {
  if (input.dim() != kConvDim || output.dim() != kConvDim) {
    ET_LOG(Error, "quantized_conv2d_out: input and output must be 4-D");
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
  if (requantize_multipliers.size(0) != out_channels ||
      requantize_shifts.size(0) != out_channels) {
    ET_LOG(
        Error,
        "quantized_conv2d_out: per-channel params must match output channels (%d)",
        static_cast<int>(out_channels));
    context.fail(Error::InvalidArgument);
    return false;
  }

  return true;
}

std::optional<cmsis_nn_dims> int8_filter_dims(
    KernelRuntimeContext& context,
    const Tensor& weight) {
  if (weight.dim() != kConvDim) {
    ET_LOG(Error, "quantized_conv2d_out: weight must be 4-D");
    context.fail(Error::InvalidArgument);
    return std::nullopt;
  }
  return cmsis_nn_dims{
      static_cast<int32_t>(weight.size(0)),
      static_cast<int32_t>(weight.size(1)),
      static_cast<int32_t>(weight.size(2)),
      static_cast<int32_t>(weight.size(3))};
}

std::optional<cmsis_nn_dims> packed_int4_filter_dims(
    KernelRuntimeContext& context,
    const Tensor& weight,
    const Int64ArrayRef weight_sizes) {
  if (weight.dim() != 1 || weight_sizes.size() != kConvDim) {
    ET_LOG(
        Error,
        "quantized_conv2d_s4_out: weight must be 1-D and weight_sizes must have length 4");
    context.fail(Error::InvalidArgument);
    return std::nullopt;
  }
  for (const int64_t size : weight_sizes) {
    if (size <= 0) {
      ET_LOG(Error, "quantized_conv2d_s4_out: weight_sizes must be positive");
      context.fail(Error::InvalidArgument);
      return std::nullopt;
    }
  }
  const int64_t count =
      weight_sizes[0] * weight_sizes[1] * weight_sizes[2] * weight_sizes[3];
  if (weight.numel() != (count + 1) / 2) {
    ET_LOG(
        Error,
        "quantized_conv2d_s4_out: weight must hold %d int4 values, two per byte",
        static_cast<int>(count));
    context.fail(Error::InvalidArgument);
    return std::nullopt;
  }
  return cmsis_nn_dims{
      static_cast<int32_t>(weight_sizes[0]),
      static_cast<int32_t>(weight_sizes[1]),
      static_cast<int32_t>(weight_sizes[2]),
      static_cast<int32_t>(weight_sizes[3])};
}

#ifndef CMSIS_NN_NO_INT4
// On MVE, arm_convolve_1_x_n_s4 misplaces packed weights when the input channel
// count is odd, can read unaligned rows when it is not a multiple of 4, and
// fails when its edge loops cannot split the padding. True for the shapes
// arm_convolve_wrapper_s4 would send there in those cases, which
// arm_convolve_s4 handles. Mirrored in passes/scratch_buffer_sizes.py.
bool use_generic_s4_kernel(
    const cmsis_nn_conv_params& params,
    const cmsis_nn_dims& input,
    const cmsis_nn_dims& filter,
    const cmsis_nn_dims& output) {
  const bool is_1x1 = params.padding.w == 0 && params.padding.h == 0 &&
      filter.w == 1 && filter.h == 1 && params.dilation.w == 1 &&
      params.dilation.h == 1;
  const bool is_1_x_n = input.h == 1 && filter.h == 1 &&
      params.dilation.w == 1 && (params.stride.w * input.c) % 4 == 0;
  if (is_1x1 || !is_1_x_n) {
    return false;
  }
  if (input.c % 4 != 0) {
    return true;
  }
  const int32_t stride = params.stride.w;
  const int32_t pad = params.padding.w;
  const int32_t total_pad = (output.w - 1) * stride + filter.w - input.w;
  const int32_t asym_pad = total_pad % 2;
  if (pad * 2 + asym_pad != total_pad) {
    return true;
  }
  const int32_t right_pad_outputs = pad + asym_pad != 0
      ? std::max<int32_t>(1, (pad + asym_pad + stride - 1) / stride)
      : 0;
  const int32_t left_pad_outputs =
      pad != 0 ? std::max<int32_t>(1, (pad + stride - 1) / stride) : 0;
  return left_pad_outputs + right_pad_outputs > output.w;
}

[[maybe_unused]] int32_t convolve_s4_get_buffer_size(
    const cmsis_nn_conv_params& params,
    const cmsis_nn_dims& input,
    const cmsis_nn_dims& filter,
    const cmsis_nn_dims& output) {
  if (!use_generic_s4_kernel(params, input, filter, output)) {
    return arm_convolve_wrapper_s4_get_buffer_size(
        &params, &input, &filter, &output);
  }
  // Dilation keeps the wrapper off its 1x1 and 1xN paths, and the
  // arm_convolve_s4 buffer size does not depend on it.
  cmsis_nn_conv_params generic_params = params;
  generic_params.dilation.w = 2;
  return arm_convolve_wrapper_s4_get_buffer_size(
      &generic_params, &input, &filter, &output);
}

arm_cmsis_nn_status convolve_s4(
    const cmsis_nn_context* ctx,
    const cmsis_nn_conv_params* params,
    const cmsis_nn_per_channel_quant_params* quant_params,
    const cmsis_nn_dims* input_dims,
    const int8_t* input_data,
    const cmsis_nn_dims* filter_dims,
    const int8_t* filter_data,
    const cmsis_nn_dims* bias_dims,
    const int32_t* bias_data,
    const cmsis_nn_dims* output_dims,
    int8_t* output_data) {
  auto* convolve =
      use_generic_s4_kernel(*params, *input_dims, *filter_dims, *output_dims)
      ? arm_convolve_s4
      : arm_convolve_wrapper_s4;
  return convolve(
      ctx,
      params,
      quant_params,
      input_dims,
      input_data,
      filter_dims,
      filter_data,
      bias_dims,
      bias_data,
      output_dims,
      output_data);
}
#endif
} // namespace

template <bool kPackedInt4>
static Tensor& quantized_conv2d_out_impl(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weight,
    const Int64ArrayRef weight_sizes,
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
    ActivationLayout layout,
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
          requantize_shifts,
          layout)) {
    return out;
  }
  const auto maybe_filter_dims = kPackedInt4
      ? packed_int4_filter_dims(context, weight, weight_sizes)
      : int8_filter_dims(context, weight);
  if (!maybe_filter_dims.has_value()) {
    return out;
  }
  const cmsis_nn_dims& filter_dims = maybe_filter_dims.value();

  const int32_t batch = static_cast<int32_t>(input.size(0));
  const int32_t input_channels = static_cast<int32_t>(
      input.size(layout == ActivationLayout::NHWCLogical ? 3 : 1));
  const int32_t input_height = static_cast<int32_t>(
      input.size(layout == ActivationLayout::NHWCLogical ? 1 : 2));
  const int32_t input_width = static_cast<int32_t>(
      input.size(layout == ActivationLayout::NHWCLogical ? 2 : 3));

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
  const cmsis_nn_dims output_dims{
      batch, output_height, output_width, output_channels};
  const cmsis_nn_dims bias_dims{1, 1, 1, output_channels};
  const cmsis_nn_dims upscale_dims{1, 1, 1, 1};

  cmsis_nn_conv_params conv_params;
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

  auto* convolve = arm_convolve_wrapper_s8;
  if constexpr (kPackedInt4) {
    // The int4 kernels have no grouped path.
    if (filter_dims.c != input_channels || filter_dims.n != output_channels) {
      ET_LOG(
          Error,
          "quantized_conv2d_s4_out: weight_sizes (%d out, %d in) must match the output and input channels (%d, %d)",
          static_cast<int>(filter_dims.n),
          static_cast<int>(filter_dims.c),
          static_cast<int>(output_channels),
          static_cast<int>(input_channels));
      context.fail(Error::InvalidArgument);
      return out;
    }
#if defined(ARM_MATH_MVEI)
    // The MVE s4 kernels de-interleave rows of 32 or more values with VLD2,
    // which faults on rows that do not start on a 4-byte boundary.
    // Mirrors _int4_kernels_safe in passes/aten_to_cortex_m_pass.py.
    const int32_t row_length = filter_dims.c * filter_dims.h * filter_dims.w;
    if (row_length >= 32 && row_length % 4 != 0) {
      ET_LOG(
          Error,
          "quantized_conv2d_s4_out: on MVE, rows of 32 or more values need a multiple of 4 values (%d)",
          static_cast<int>(row_length));
      context.fail(Error::NotSupported);
      return out;
    }
#endif
#ifdef CMSIS_NN_NO_INT4
    ET_LOG(
        Error,
        "quantized_conv2d_s4_out: CMSIS-NN was built without its int4 kernels");
    context.fail(Error::NotSupported);
    return out;
#else
    convolve = convolve_s4;
#endif
  }

  cmsis_nn_context cmsis_context;
  cmsis_context.buf = nullptr;
  cmsis_context.size = scratch.nbytes();
  if (cmsis_context.size > 0) {
    cmsis_context.buf = scratch.mutable_data_ptr<int8_t>();
  }

#ifdef CORTEX_M_ENABLE_RUNTIME_CHECKS
  int32_t runtime_buffer_bytes = 0;
  if constexpr (kPackedInt4) {
#ifndef CMSIS_NN_NO_INT4
    runtime_buffer_bytes = convolve_s4_get_buffer_size(
        conv_params, input_dims, filter_dims, output_dims);
#endif
  } else {
    runtime_buffer_bytes = arm_convolve_wrapper_s8_get_buffer_size(
        &conv_params, &input_dims, &filter_dims, &output_dims);
  }
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

  const arm_cmsis_nn_status status = convolve(
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
        "quantized_conv2d_out: CMSIS-NN convolution failed with status %d",
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
    Tensor& out) {
  return quantized_conv2d_out_impl<false>(
      context,
      input,
      weight,
      {},
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
    Tensor& out) {
  return quantized_conv2d_out_impl<false>(
      context,
      input,
      weight,
      {},
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
      ActivationLayout::NHWCLogical,
      out);
}

// cppcheck-suppress unusedFunction
Tensor& quantized_conv2d_s4_out(
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
    const Int64ArrayRef weight_sizes,
    const Tensor& scratch,
    Tensor& out) {
  return quantized_conv2d_out_impl<true>(
      context,
      input,
      weight,
      weight_sizes,
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
      ActivationLayout::NCHWLogical,
      out);
}

// cppcheck-suppress unusedFunction
Tensor& quantized_conv2d_s4_nhwc_out(
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
    const Int64ArrayRef weight_sizes,
    const Tensor& scratch,
    Tensor& out) {
  return quantized_conv2d_out_impl<true>(
      context,
      input,
      weight,
      weight_sizes,
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
      ActivationLayout::NHWCLogical,
      out);
}

} // namespace native
} // namespace cortex_m
