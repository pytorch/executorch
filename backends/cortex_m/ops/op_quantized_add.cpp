/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2025 Arm Limited and/or its affiliates.
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

Tensor& quantized_add_out(
    KernelRuntimeContext& context,
    const Tensor& input1_int8,
    const Scalar& input1_zero_point,
    const Scalar& input1_multiplier,
    const Scalar& input1_shift,
    const Tensor& input2_int8,
    const Scalar& input2_zero_point,
    const Scalar& input2_multiplier,
    const Scalar& input2_shift,
    const Scalar& output_zero_point,
    const Scalar& output_multiplier,
    const Scalar& output_shift,
    Tensor& out) {
  // Validate tensor types and dim order
  validate_cmsis_nn_tensor_requirements(input1_int8, input2_int8, out);

  // Validate quantization parameters
  validate_quantization_params(
      input1_zero_point,
      input1_multiplier,
      input1_shift,
      input2_zero_point,
      input2_multiplier,
      input2_shift,
      output_zero_point,
      output_multiplier,
      output_shift,
      out);

  ET_LOG(
      Info,
      "quantized_add_out: input1_int8.sizes() = %zu",
      input1_int8.sizes().size());

  int32_t zp1 = extractScalarToInt32(input1_zero_point);
  int32_t input1_mult = extractScalarToInt32(input1_multiplier);
  int input1_shift_val = extractScalarToInt(input1_shift);
  int32_t zp2 = extractScalarToInt32(input2_zero_point);
  int32_t input2_mult = extractScalarToInt32(input2_multiplier);
  int input2_shift_val = extractScalarToInt(input2_shift);
  int32_t out_zp = extractScalarToInt32(output_zero_point);
  int32_t output_mult = extractScalarToInt32(output_multiplier);
  int output_shift_val = extractScalarToInt(output_shift);

  // Left shift to maximize precision
  const int32_t left_shift = 20;
  const int32_t activation_min = std::numeric_limits<int8_t>::min();
  const int32_t activation_max = std::numeric_limits<int8_t>::max();

  ET_LOG(
      Info,
      "Using AoT-computed parameters: input1[mult=%d, shift=%d], input2[mult=%d, shift=%d], output[mult=%d, shift=%d]",
      input1_mult,
      input1_shift_val,
      input2_mult,
      input2_shift_val,
      output_mult,
      output_shift_val);

  // Call CMSIS-NN kernel with precomputed parameters
  arm_cmsis_nn_status status = arm_elementwise_add_s8(
      input1_int8.const_data_ptr<int8_t>(),
      input2_int8.const_data_ptr<int8_t>(),
      -static_cast<int32_t>(zp1),
      input1_mult,
      input1_shift_val,
      -static_cast<int32_t>(zp2),
      input2_mult,
      input2_shift_val,
      left_shift,
      out.mutable_data_ptr<int8_t>(),
      static_cast<int32_t>(out_zp),
      output_mult,
      output_shift_val,
      activation_min,
      activation_max,
      static_cast<int32_t>(out.numel()));

  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(
        Error,
        "quantized_add_out: arm_elementwise_add_s8 failed with status [%d]",
        status);

    context.fail(Error::Internal); // Fail the execution context
    return out;
  }
  ET_LOG(
      Info,
      "quantized_add_out: Successfully completed with AoT-computed parameters!");

  return out;
}

} // namespace native
} // namespace cortex_m
