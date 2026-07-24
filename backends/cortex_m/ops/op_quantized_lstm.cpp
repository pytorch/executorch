/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cortex_m_ops_common.h"

namespace cortex_m {
namespace native {
using KernelRuntimeContext = torch::executor::KernelRuntimeContext;

namespace {

// PyTorch stacks gate weights in IFGO order (input, forget, cell, output);
// CMSIS-NN addresses gates by name. Only the cell gate uses tanh.
constexpr int kNumGates = 4;
constexpr int kCellGateIdx = 2;

void fail_arg(KernelRuntimeContext& context, const char* msg) {
  ET_LOG(Error, "quantized_lstm_out: %s", msg);
  context.fail(Error::InvalidArgument);
}

} // namespace

// cppcheck-suppress unusedFunction
Tensor& quantized_lstm_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Tensor& input_weights,
    const Tensor& hidden_weights,
    const Tensor& input_effective_bias,
    const Tensor& hidden_effective_bias,
    const Int64ArrayRef input_multipliers,
    const Int64ArrayRef input_shifts,
    const Int64ArrayRef hidden_multipliers,
    const Int64ArrayRef hidden_shifts,
    const int64_t input_offset,
    const int64_t output_offset,
    const int64_t forget_to_cell_multiplier,
    const int64_t forget_to_cell_shift,
    const int64_t input_to_cell_multiplier,
    const int64_t input_to_cell_shift,
    const int64_t output_multiplier,
    const int64_t output_shift,
    const int64_t cell_scale_power,
    const int64_t cell_clip,
    const bool time_major,
    const Tensor& temp1,
    const Tensor& temp2,
    const Tensor& cell_state,
    Tensor& out) {
  if (input.dim() != 3 || out.dim() != 3 || input_weights.dim() != 2 ||
      hidden_weights.dim() != 2) {
    fail_arg(context, "input/out must be 3-D, weights 2-D");
    return out;
  }
  if (input.scalar_type() != ScalarType::Char ||
      out.scalar_type() != ScalarType::Char ||
      input_weights.scalar_type() != ScalarType::Char ||
      hidden_weights.scalar_type() != ScalarType::Char) {
    fail_arg(context, "input/output/weights must be int8");
    return out;
  }
  if (input_effective_bias.scalar_type() != ScalarType::Int ||
      hidden_effective_bias.scalar_type() != ScalarType::Int) {
    fail_arg(context, "effective biases must be int32");
    return out;
  }
  if (input_weights.size(0) % kNumGates != 0) {
    fail_arg(context, "input_weights rows must be 4*hidden");
    return out;
  }

  const int32_t hidden_size =
      static_cast<int32_t>(input_weights.size(0)) / kNumGates;
  const int32_t input_size = static_cast<int32_t>(input_weights.size(1));
  const int32_t time_steps =
      static_cast<int32_t>(time_major ? input.size(0) : input.size(1));
  const int32_t batch_size =
      static_cast<int32_t>(time_major ? input.size(1) : input.size(0));

  // Shape self-consistency: CMSIS strides these tensors using the dims above.
  const int64_t gated_hidden = static_cast<int64_t>(kNumGates) * hidden_size;
  if (input.size(2) != input_size || hidden_weights.size(0) != gated_hidden ||
      hidden_weights.size(1) != hidden_size ||
      input_effective_bias.numel() != gated_hidden ||
      hidden_effective_bias.numel() != gated_hidden ||
      out.numel() !=
          static_cast<int64_t>(time_steps) * batch_size * hidden_size) {
    fail_arg(context, "tensor shapes are inconsistent");
    return out;
  }
  if (input_multipliers.size() != kNumGates ||
      input_shifts.size() != kNumGates ||
      hidden_multipliers.size() != kNumGates ||
      hidden_shifts.size() != kNumGates) {
    fail_arg(context, "per-gate multiplier/shift lists must be length 4");
    return out;
  }

  // Each working buffer holds batch*hidden int16 values.
  const size_t buffer_bytes =
      static_cast<size_t>(batch_size) * hidden_size * sizeof(int16_t);
  if (temp1.nbytes() < buffer_bytes || temp2.nbytes() < buffer_bytes ||
      cell_state.nbytes() < buffer_bytes) {
    fail_arg(context, "scratch buffers too small");
    return out;
  }

  const int8_t* input_data = input.const_data_ptr<int8_t>();
  const int8_t* iw = input_weights.const_data_ptr<int8_t>();
  const int8_t* hw = hidden_weights.const_data_ptr<int8_t>();
  const int32_t* ieb = input_effective_bias.const_data_ptr<int32_t>();
  const int32_t* heb = hidden_effective_bias.const_data_ptr<int32_t>();
  int8_t* output_data = out.mutable_data_ptr<int8_t>();

  cmsis_nn_lstm_gate gates[kNumGates];
  for (int g = 0; g < kNumGates; ++g) {
    gates[g].input_multiplier = static_cast<int32_t>(input_multipliers[g]);
    gates[g].input_shift = static_cast<int32_t>(input_shifts[g]);
    gates[g].input_weights = iw + g * hidden_size * input_size;
    gates[g].input_effective_bias = ieb + g * hidden_size;
    gates[g].hidden_multiplier = static_cast<int32_t>(hidden_multipliers[g]);
    gates[g].hidden_shift = static_cast<int32_t>(hidden_shifts[g]);
    gates[g].hidden_weights = hw + g * hidden_size * hidden_size;
    gates[g].hidden_effective_bias = heb + g * hidden_size;
    gates[g].bias = nullptr; // unused by the s8 gate kernel
    gates[g].activation_type = (g == kCellGateIdx) ? ARM_TANH : ARM_SIGMOID;
  }

  cmsis_nn_lstm_params params;
  params.time_major = time_major ? 1 : 0;
  params.batch_size = batch_size;
  params.time_steps = time_steps;
  params.input_size = input_size;
  params.hidden_size = hidden_size;
  params.input_offset = static_cast<int32_t>(input_offset);
  params.forget_to_cell_multiplier =
      static_cast<int32_t>(forget_to_cell_multiplier);
  params.forget_to_cell_shift = static_cast<int32_t>(forget_to_cell_shift);
  params.input_to_cell_multiplier =
      static_cast<int32_t>(input_to_cell_multiplier);
  params.input_to_cell_shift = static_cast<int32_t>(input_to_cell_shift);
  params.cell_clip = static_cast<int32_t>(cell_clip);
  params.cell_scale_power = static_cast<int32_t>(cell_scale_power);
  params.output_multiplier = static_cast<int32_t>(output_multiplier);
  params.output_shift = static_cast<int32_t>(output_shift);
  params.output_offset = static_cast<int32_t>(output_offset);
  params.input_gate = gates[0];
  params.forget_gate = gates[1];
  params.cell_gate = gates[kCellGateIdx];
  params.output_gate = gates[3];

  cmsis_nn_lstm_context buffers;
  buffers.temp1 = reinterpret_cast<int16_t*>(temp1.mutable_data_ptr<int8_t>());
  buffers.temp2 = reinterpret_cast<int16_t*>(temp2.mutable_data_ptr<int8_t>());
  buffers.cell_state =
      reinterpret_cast<int16_t*>(cell_state.mutable_data_ptr<int8_t>());

  const arm_cmsis_nn_status status =
      arm_lstm_unidirectional_s8(input_data, output_data, &params, &buffers);
  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(
        Error, "quantized_lstm_out: CMSIS-NN failed with status [%d]", status);
    context.fail(Error::Internal);
    return out;
  }
  return out;
}

} // namespace native
} // namespace cortex_m
