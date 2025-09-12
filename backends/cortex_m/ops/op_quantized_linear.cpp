/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cortex_m_ops_common.h"

// Include CMSIS-NN headers with C linkage
extern "C" {
#include "arm_nnfunctions.h"
}

namespace cortex_m {
namespace native {
using KernelRuntimeContext = torch::executor::KernelRuntimeContext;

struct kernel_sum_state {
  bool updated = false;
  char buffer[2048] = {0};
};

Tensor& quantized_linear_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Scalar& input_zero_point,
    const Scalar& input_multiplier,
    const Scalar& input_shift,
    const Tensor& weights,
    const Tensor& weight_zero_point,
    const Tensor& weight_multiplier,
    const Tensor& weight_shift,
    const torch::executor::optional<Tensor>& bias,
    const Tensor& bias_multiplier, // IGNORE - not used
    const Tensor& bias_shift, // IGNORE - not used
    const Tensor& scratch_buffer,
    const Scalar& output_zero_point,
    const Scalar& in_features,
    const Scalar& out_features,
    Tensor& out) {
  ET_LOG(Info, "quantized_linear_out: called");
  validate_cmsis_nn_tensor_requirements(input, weights, out);

  ET_CHECK_MSG(
      scratch_buffer.scalar_type() == ScalarType::Char,
      "Scratch buffer must be int8");

  // --- Parameter Extraction and Validation ---
  const int32_t batch_size = input.size(0);
  const int32_t in_feat = static_cast<int32_t>(in_features.to<int64_t>());
  const int32_t out_feat = static_cast<int32_t>(out_features.to<int64_t>());
  int32_t input_zp = static_cast<int32_t>(input_zero_point.to<int64_t>());
  int32_t output_zp = static_cast<int32_t>(output_zero_point.to<int64_t>());
  bool is_per_channel = (weight_zero_point.numel() > 1);
  const int8_t* input_data = input.const_data_ptr<int8_t>();
  const int8_t* weight_data = weights.const_data_ptr<int8_t>();
  const int32_t* bias_data =
      bias.has_value() ? bias.value().const_data_ptr<int32_t>() : nullptr;
  int8_t* output_data = out.mutable_data_ptr<int8_t>();
  int8_t* scratch_ptr = scratch_buffer.mutable_data_ptr<int8_t>();
  const int32_t* weight_zp_data = weight_zero_point.const_data_ptr<int32_t>();
  const int32_t* weight_mult_data = weight_multiplier.const_data_ptr<int32_t>();
  const int32_t* weight_shift_data = weight_shift.const_data_ptr<int32_t>();

  if (!bias.has_value()) {
    ET_LOG(Info, "No bias tensor provided (bias_data is nullptr)");
  }
  validate_per_channel_quant_params(
      weight_mult_data, weight_shift_data, out_feat);

  cmsis_nn_fc_params fc_params;
  fc_params.input_offset = -input_zp;
  fc_params.output_offset = output_zp;
  fc_params.activation.min = std::numeric_limits<int8_t>::min();
  fc_params.activation.max = std::numeric_limits<int8_t>::max();
  cmsis_nn_dims input_dims = {1, 1, 1, in_feat};
  cmsis_nn_dims filter_dims = {out_feat, 1, 1, in_feat};
  cmsis_nn_dims bias_dims = {1, 1, 1, out_feat};
  cmsis_nn_dims output_dims = {1, 1, 1, out_feat};
  arm_cmsis_nn_status status;

  // Pass allocates a flat scratch buffer:
  // [------------------- scratch_buffer -----------------------]
  // |<- CMSIS-NN workspace ->|<--- kernel_sum_state struct --->|
  //
  // Buffer pointers:
  // ^                        ^                                 ^
  // scratch_ptr(start)   scratch_ptr + cmsis_scratch   scratch_ptr + total_size
  //
  // - CMSIS-NN workspace: used by CMSIS-NN kernels for temporary data
  // - Always give CMSIS-NN the start of the buffer for alignment
  // - Place kernel_sum_state structs at the end to avoid breaking alignment
  cmsis_nn_context ctx;
  kernel_sum_state* state = reinterpret_cast<kernel_sum_state*>(
      scratch_ptr + scratch_buffer.size(0) - sizeof(kernel_sum_state));
  if (!state->updated) {
    int required_bytes = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    ET_CHECK_MSG(
        (scratch_buffer.size(0) - sizeof(kernel_sum_state) >= required_bytes),
        "Scratch buffer size %zu is not enough for kernel sum buffer size %d",
        sizeof(state->buffer),
        required_bytes);

    // Compute kernel sums once
    arm_vector_sum_s8(
        (int32_t*)scratch_ptr,
        in_feat,
        out_feat,
        weight_data,
        weight_zp_data[0],
        0, // rhs_offset (int32_t)
        nullptr // bias (const int32_t*)
    );
    state->updated = true;
    ET_LOG(
        Info,
        "Computed kernel sums, stored in state->buffer [required_bytes : %d]",
        required_bytes);
  }

  // start of cmsis buffer
  ctx.buf = scratch_ptr;
  ctx.size = scratch_buffer.size(0) - sizeof(kernel_sum_state);

  for (int32_t b = 0; b < batch_size; b++) {
    const int8_t* batch_input = input_data + b * in_feat;
    int8_t* batch_output = output_data + b * out_feat;
    if (is_per_channel) {
      // Per-channel quantization
      cmsis_nn_per_channel_quant_params per_channel_quant_params;
      per_channel_quant_params.multiplier =
          const_cast<int32_t*>(weight_mult_data);
      per_channel_quant_params.shift = const_cast<int32_t*>(weight_shift_data);

      status = arm_fully_connected_per_channel_s8(
          &ctx,
          &fc_params,
          &per_channel_quant_params,
          &input_dims,
          batch_input,
          &filter_dims,
          weight_data,
          &bias_dims,
          bias_data,
          &output_dims,
          batch_output);
    } else {
      // Per-tensor quantization
      fc_params.filter_offset = -weight_zp_data[0];
      cmsis_nn_per_tensor_quant_params per_tensor_quant_params;
      per_tensor_quant_params.multiplier = weight_mult_data[0];
      per_tensor_quant_params.shift = weight_shift_data[0];

      status = arm_fully_connected_s8(
          &ctx,
          &fc_params,
          &per_tensor_quant_params,
          &input_dims,
          batch_input,
          &filter_dims,
          weight_data,
          &bias_dims,
          bias_data,
          &output_dims,
          batch_output);
    }

    if (status != ARM_CMSIS_NN_SUCCESS) {
      ET_LOG(
          Error,
          "quantized_linear_out: CMSIS-NN failed with status [%d]",
          status);
      context.fail(Error::Internal);
      return out;
    }
  }
  return out;
}

// Functional variant (stub, not used at runtime)
Tensor quantized_linear(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Scalar& input_zero_point,
    const Scalar& input_multiplier,
    const Scalar& input_shift,
    const Tensor& weights,
    const Tensor& weight_zero_point,
    const Tensor& weight_multiplier,
    const Tensor& weight_shift,
    const torch::executor::optional<Tensor>& bias,
    const Tensor& bias_multiplier,
    const Tensor& bias_shift,
    const Tensor& scratch_buffer,
    const Scalar& output_zero_point,
    const Scalar& in_features,
    const Scalar& out_features) {
  ET_LOG(Info, "quantized_linear: called");
  assert(false);
  return const_cast<Tensor&>(input);
}

} // namespace native
} // namespace cortex_m
