/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
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

Tensor& quantized_linear_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& weights,
    const torch::executor::optional<Tensor>& bias,
    const torch::executor::optional<Tensor>& kernel_sum,
    const int64_t input_offset,
    const int64_t filter_offset,
    const int64_t output_offset,
    const Int64ArrayRef requantize_multipliers,
    const Int64ArrayRef requantize_shifts,
    const int64_t activation_max,
    const int64_t activation_min,
    Tensor& out) {
  ET_LOG(Info, "quantized_linear_out: called");

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

  cmsis_nn_per_tensor_quant_params per_tensor_quant_params;
  per_tensor_quant_params.multiplier =
      static_cast<int32_t>(requantize_multipliers.at(0));
  per_tensor_quant_params.shift = static_cast<int32_t>(requantize_shifts.at(0));

  auto in_feat = input.size(input.dim() - 1);
  auto out_feat = out.size(out.dim() - 1);
  auto batches = 1;
  for (size_t i = 0; i < input.dim() - 1; i++) {
    batches *= input.size(i);
  }
  ET_LOG(
      Info,
      "in features: %d, out_features: %d, batches: %d, kernel_sum_size: %d",
      in_feat,
      out_feat,
      batches,
      kernel_sum.has_value() ? kernel_sum.value().numel() : 0);
  ET_LOG(
      Info,
      "kernel_sum[0]: %d, kernel_sum[1]: %d",
      kernel_sum_data != nullptr ? kernel_sum_data[0] : -1,
      kernel_sum_data != nullptr ? kernel_sum_data[1] : -1);
  cmsis_nn_dims input_dims = {batches, 1, 1, in_feat};
  cmsis_nn_dims filter_dims = {in_feat, 1, 1, out_feat};
  cmsis_nn_dims bias_dims = {1, 1, 1, out_feat};
  cmsis_nn_dims output_dims = {batches, 1, 1, out_feat};

  arm_cmsis_nn_status status = arm_fully_connected_s8(
      &ctx,
      &fc_params,
      &per_tensor_quant_params,
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
        "quantized_linear_out: CMSIS-NN failed with status [%d]",
        status);
    context.fail(Error::Internal);
    return out;
  }

  return out;
}

} // namespace native
} // namespace cortex_m
