/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2025-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cortex_m_ops_common.h"

namespace cortex_m {
namespace native {
using KernelRuntimeContext = torch::executor::KernelRuntimeContext;

namespace {
// Runs `fully_connected`, a CMSIS-NN fully connected call bound to its
// requantization parameters, on the op's tensors.
template <typename FullyConnected>
Tensor& quantized_linear_impl(
    const char* op_name,
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weights,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& kernel_sum,
    const int64_t input_offset,
    const int64_t filter_offset,
    const int64_t output_offset,
    const int64_t activation_max,
    const int64_t activation_min,
    FullyConnected fully_connected,
    Tensor& out) {
  ET_LOG(Debug, "%s: called", op_name);

  const int8_t* input_data = input.const_data_ptr<int8_t>();
  const int8_t* weight_data = weights.const_data_ptr<int8_t>();
  const int32_t* bias_data =
      bias.has_value() ? bias.value().const_data_ptr<int32_t>() : nullptr;
  int32_t* kernel_sum_data =
      kernel_sum.has_value() ? kernel_sum.value().data_ptr<int32_t>() : nullptr;
  int8_t* output_data = out.mutable_data_ptr<int8_t>();

  cmsis_nn_context ctx;
  ctx.size = 0; // Not used in CMSIS-NN
  ctx.buf = kernel_sum_data;

  // Setup CMSIS-NN parameters
  cmsis_nn_fc_params fc_params;
  fc_params.input_offset = static_cast<int32_t>(input_offset);
  fc_params.filter_offset = static_cast<int32_t>(filter_offset);
  fc_params.output_offset = static_cast<int32_t>(output_offset);
  fc_params.activation.min = static_cast<int32_t>(activation_min);
  fc_params.activation.max = static_cast<int32_t>(activation_max);

  auto in_feat = input.size(input.dim() - 1);
  auto out_feat = out.size(out.dim() - 1);
  auto batches = 1;
  for (size_t i = 0; i < input.dim() - 1; i++) {
    batches *= input.size(i);
  }
  ET_LOG(
      Debug,
      "in features: %d, out_features: %d, batches: %d, kernel_sum_size: %d",
      in_feat,
      out_feat,
      batches,
      kernel_sum.has_value() ? kernel_sum.value().numel() : 0);
  ET_LOG(
      Debug,
      "kernel_sum[0]: %d, kernel_sum[1]: %d",
      kernel_sum_data != nullptr ? kernel_sum_data[0] : -1,
      kernel_sum_data != nullptr ? kernel_sum_data[1] : -1);
  cmsis_nn_dims input_dims = {batches, 1, 1, in_feat};
  cmsis_nn_dims filter_dims = {in_feat, 1, 1, out_feat};
  cmsis_nn_dims bias_dims = {1, 1, 1, out_feat};
  cmsis_nn_dims output_dims = {batches, 1, 1, out_feat};

  const arm_cmsis_nn_status status = fully_connected(
      &ctx,
      &fc_params,
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
        "%s: CMSIS-NN failed with status [%d]",
        op_name,
        status);
    context.fail(Error::Internal);
    return out;
  }

  return out;
}
} // namespace

// cppcheck-suppress unusedFunction
Tensor& quantized_linear_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weights,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& kernel_sum,
    const int64_t input_offset,
    const int64_t filter_offset,
    const int64_t output_offset,
    const Int64ArrayRef requantize_multipliers,
    const Int64ArrayRef requantize_shifts,
    const int64_t activation_max,
    const int64_t activation_min,
    Tensor& out) {
  const cmsis_nn_per_tensor_quant_params quant_params{
      static_cast<int32_t>(requantize_multipliers.at(0)),
      static_cast<int32_t>(requantize_shifts.at(0))};
  return quantized_linear_impl(
      "quantized_linear_out",
      context,
      input,
      weights,
      bias,
      kernel_sum,
      input_offset,
      filter_offset,
      output_offset,
      activation_max,
      activation_min,
      [&quant_params](auto ctx, auto fc_params, auto... args) {
        return arm_fully_connected_s8(ctx, fc_params, &quant_params, args...);
      },
      out);
}

// cppcheck-suppress unusedFunction
Tensor& quantized_linear_per_channel_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weights,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& kernel_sum,
    const int64_t input_offset,
    const int64_t filter_offset,
    const int64_t output_offset,
    const Tensor& requantize_multipliers,
    const Tensor& requantize_shifts,
    const int64_t activation_max,
    const int64_t activation_min,
    Tensor& out) {
  const int64_t out_features = out.size(out.dim() - 1);
  if (requantize_multipliers.scalar_type() != ScalarType::Int ||
      requantize_shifts.scalar_type() != ScalarType::Int ||
      requantize_multipliers.numel() != out_features ||
      requantize_shifts.numel() != out_features) {
    ET_LOG(
        Error,
        "quantized_linear_per_channel_out: requantization parameters must be int32 with one per output feature (%d)",
        static_cast<int>(out_features));
    context.fail(Error::InvalidArgument);
    return out;
  }
  const cmsis_nn_per_channel_quant_params quant_params{
      requantize_multipliers.mutable_data_ptr<int32_t>(),
      requantize_shifts.mutable_data_ptr<int32_t>()};
  return quantized_linear_impl(
      "quantized_linear_per_channel_out",
      context,
      input,
      weights,
      bias,
      kernel_sum,
      input_offset,
      filter_offset,
      output_offset,
      activation_max,
      activation_min,
      [&quant_params](auto ctx, auto fc_params, auto... args) {
        return arm_fully_connected_per_channel_s8(
            ctx, fc_params, &quant_params, args...);
      },
      out);
}

// cppcheck-suppress unusedFunction
Tensor& quantized_linear_s4_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weights,
    const std::optional<Tensor>& bias,
    const int64_t input_offset,
    const int64_t output_offset,
    const Tensor& requantize_multipliers,
    const Tensor& requantize_shifts,
    const int64_t activation_max,
    const int64_t activation_min,
    Tensor& out) {
  const int64_t in_features = input.size(input.dim() - 1);
  const int64_t out_features = out.size(out.dim() - 1);
  if (in_features <= 0 || input.scalar_type() != ScalarType::Char ||
      out.scalar_type() != ScalarType::Char ||
      out.numel() != input.numel() / in_features * out_features ||
      weights.scalar_type() != ScalarType::Char || weights.dim() != 1 ||
      weights.numel() != (in_features * out_features + 1) / 2) {
    ET_LOG(
        Error,
        "quantized_linear_s4_out: input and output must be int8 with matching rows, and weights must hold %d int4 values, two per byte",
        static_cast<int>(in_features * out_features));
    context.fail(Error::InvalidArgument);
    return out;
  }
  if (requantize_multipliers.scalar_type() != ScalarType::Int ||
      requantize_shifts.scalar_type() != ScalarType::Int ||
      requantize_multipliers.numel() != out_features ||
      requantize_shifts.numel() != out_features ||
      (bias.has_value() &&
       (bias.value().scalar_type() != ScalarType::Int ||
        bias.value().numel() != out_features))) {
    ET_LOG(
        Error,
        "quantized_linear_s4_out: requantization parameters and bias must be int32 with one per output feature (%d)",
        static_cast<int>(out_features));
    context.fail(Error::InvalidArgument);
    return out;
  }
#if defined(ARM_MATH_MVEI)
  // The MVE s4 matrix multiply de-interleaves rows of 32 or more values with
  // VLD2, which faults on rows that do not start on a 4-byte boundary.
  // Mirrors _int4_kernels_safe in passes/aten_to_cortex_m_pass.py.
  if (in_features >= 32 && in_features % 4 != 0) {
    ET_LOG(
        Error,
        "quantized_linear_s4_out: on MVE, 32 or more input features must be a multiple of 4 (%d)",
        static_cast<int>(in_features));
    context.fail(Error::NotSupported);
    return out;
  }
#endif
#ifdef CMSIS_NN_NO_INT4
  ET_LOG(
      Error,
      "quantized_linear_s4_out: CMSIS-NN was built without its int4 kernels");
  context.fail(Error::NotSupported);
  return out;
#else
  const int32_t rows = static_cast<int32_t>(input.numel() / in_features);
  const int32_t in_feat = static_cast<int32_t>(in_features);
  const int32_t out_feat = static_cast<int32_t>(out_features);

  // A linear layer is a 1x1 convolution over its rows, which runs the
  // per-channel int4 matrix multiply. CMSIS-NN's s4 fully connected kernel is
  // per-tensor only.
  cmsis_nn_conv_params conv_params;
  conv_params.input_offset = static_cast<int32_t>(input_offset);
  conv_params.output_offset = static_cast<int32_t>(output_offset);
  conv_params.stride = {1, 1};
  conv_params.padding = {0, 0};
  conv_params.dilation = {1, 1};
  conv_params.activation.min = static_cast<int32_t>(activation_min);
  conv_params.activation.max = static_cast<int32_t>(activation_max);
  const cmsis_nn_per_channel_quant_params quant_params{
      requantize_multipliers.mutable_data_ptr<int32_t>(),
      requantize_shifts.mutable_data_ptr<int32_t>()};
  const cmsis_nn_dims input_dims{rows, 1, 1, in_feat};
  const cmsis_nn_dims filter_dims{out_feat, 1, 1, in_feat};
  const cmsis_nn_dims bias_dims{1, 1, 1, out_feat};
  const cmsis_nn_dims output_dims{rows, 1, 1, out_feat};
  // The 1x1 int4 kernel needs no scratch.
  const cmsis_nn_context ctx{nullptr, 0};

  const arm_cmsis_nn_status status = arm_convolve_1x1_s4_fast(
      &ctx,
      &conv_params,
      &quant_params,
      &input_dims,
      input.const_data_ptr<int8_t>(),
      &filter_dims,
      weights.const_data_ptr<int8_t>(),
      &bias_dims,
      bias.has_value() ? bias.value().const_data_ptr<int32_t>() : nullptr,
      &output_dims,
      out.mutable_data_ptr<int8_t>());
  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(
        Error,
        "quantized_linear_s4_out: CMSIS-NN failed with status [%d]",
        status);
    context.fail(Error::Internal);
  }
  return out;
#endif
}

} // namespace native
} // namespace cortex_m
