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

Tensor& quantized_linear_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    const Scalar& input_zero_point,
    const Scalar& input_multiplier,
    const Scalar& input_shift,
    const Tensor& weights,
    const Tensor& kernel_sums,
    const Scalar& weight_zero_point,
    const Scalar& weight_multiplier,
    const Scalar& weight_shift,
    const torch::executor::optional<Tensor>& bias,
    const Scalar& bias_multiplier,
    const Scalar& bias_shift,
    const Tensor& scratch_buffer,
    const Scalar& output_zero_point,
    const Scalar& in_features,
    const Scalar& out_features,
    Tensor& out) {
  ET_LOG(Info, "quantized_linear_out: called");

  // Validate tensor requirements (following your pattern)
  validate_cmsis_nn_tensor_requirements(input, weights, out);

  ET_CHECK_MSG(
      scratch_buffer.scalar_type() == ScalarType::Char,
      "Scratch buffer must be int8");

  // Extract precomputed dimensions (AOT calculated)
  const int32_t batch_size = input.size(0);
  const int32_t in_feat = static_cast<int32_t>(in_features.to<int64_t>());
  const int32_t out_feat = static_cast<int32_t>(out_features.to<int64_t>());

  // Extract quantization parameters (following your to<int64_t>() pattern)
  int32_t input_zp = static_cast<int32_t>(input_zero_point.to<int64_t>());
  int32_t input_mult = static_cast<int32_t>(input_multiplier.to<int64_t>());
  int input_shift_val = static_cast<int>(input_shift.to<int64_t>());

  int32_t weight_zp = static_cast<int32_t>(weight_zero_point.to<int64_t>());
  int32_t weight_mult = static_cast<int32_t>(weight_multiplier.to<int64_t>());
  int weight_shift_val = static_cast<int>(weight_shift.to<int64_t>());

  int32_t output_zp = static_cast<int32_t>(output_zero_point.to<int64_t>());

  // Get data pointers (all precomputed at AOT!)
  const int8_t* input_data = input.const_data_ptr<int8_t>();
  const int8_t* weight_data =
      weights.const_data_ptr<int8_t>(); // Already transposed!
  const int32_t* kernel_sum_data =
      kernel_sums.const_data_ptr<int32_t>(); // Already computed!
  const int32_t* bias_data =
      bias.has_value() ? bias.value().const_data_ptr<int32_t>() : nullptr;
  int8_t* output_data = out.mutable_data_ptr<int8_t>();

  // Get persistent scratch buffer pointer
  int8_t* scratch_ptr = scratch_buffer.mutable_data_ptr<int8_t>();
  const size_t scratch_size = scratch_buffer.size(0);

  ET_LOG(Info, "quantized_linear_out: Using AOT-precomputed parameters");
  ET_LOG(
      Info,
      "   Dimensions: [%d, %d] -> [%d, %d]",
      batch_size,
      in_feat,
      batch_size,
      out_feat);
  ET_LOG(
      Info, "   Scratch buffer: ptr=%p, size=%zu", scratch_ptr, scratch_size);

  // SETUP CMSIS-NN CONTEXT WITH PERSISTENT SCRATCH BUFFER (TBD testing)
  cmsis_nn_context ctx;
  ctx.buf = scratch_ptr;
  ctx.size = scratch_size;

  cmsis_nn_fc_params fc_params;
  fc_params.input_offset = -input_zp; // Note: negative for CMSIS-NN
  fc_params.output_offset = output_zp;
  fc_params.filter_offset = -weight_zp; // Note: negative for CMSIS-NN
  fc_params.activation.min = std::numeric_limits<int8_t>::min();
  fc_params.activation.max = std::numeric_limits<int8_t>::max();

  cmsis_nn_per_tensor_quant_params quant_params;
  quant_params.multiplier = weight_mult; // AOT precomputed!
  quant_params.shift = weight_shift_val; // AOT precomputed!

  // Setup dimension structures (CMSIS-NN format)
  cmsis_nn_dims input_dims = {1, 1, 1, in_feat};
  cmsis_nn_dims filter_dims = {out_feat, 1, 1, in_feat};
  cmsis_nn_dims bias_dims = {1, 1, 1, out_feat};
  cmsis_nn_dims output_dims = {1, 1, 1, out_feat};

  for (int32_t b = 0; b < batch_size; b++) {
    const int8_t* batch_input = input_data + b * in_feat;
    int8_t* batch_output = output_data + b * out_feat;

    // *** OPTIMAL CMSIS-NN API CALL ***
    arm_cmsis_nn_status status = arm_fully_connected_s8(
        &ctx, // Persistent scratch buffer context
        &fc_params, // FC parameters
        &quant_params, // AOT precomputed quantization params
        &input_dims, // Input dimensions
        batch_input, // Input data
        &filter_dims, // Filter dimensions
        weight_data, // AOT precomputed transposed weights
        &bias_dims, // Bias dimensions
        bias_data, // Bias data (optional)
        &output_dims, // Output dimensions
        batch_output // Output data
    );

    if (status != ARM_CMSIS_NN_SUCCESS) {
      ET_LOG(
          Error,
          "quantized_linear_out: arm_fully_connected_s8 failed with status [%d]",
          status);
      context.fail(Error::Internal);
      return out;
    }
  }

  ET_LOG(
      Info,
      "quantized_linear_out: Successfully completed with AOT-precomputed parameters!");
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
    const Tensor& kernel_sums,
    const Scalar& weight_zero_point,
    const Scalar& weight_multiplier,
    const Scalar& weight_shift,
    const torch::executor::optional<Tensor>& bias,
    const Scalar& bias_multiplier,
    const Scalar& bias_shift,
    const Tensor& scratch_buffer,
    const Scalar& output_zero_point,
    const Scalar& in_features,
    const Scalar& out_features) {
  ET_LOG(Info, "quantized_linear: called");
  assert(false); // Should be converted to .out variant
  return const_cast<Tensor&>(input);
}

} // namespace native
} // namespace cortex_m
