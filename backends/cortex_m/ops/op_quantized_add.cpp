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
  bool channel_broadcast = is_channel_broadcast(input1_int8, input2_int8);
  validate_cmsis_nn_tensor_requirements(
      input1_int8,
      input2_int8,
      out,
      ScalarType::Char,
      /*require_channels_last=*/channel_broadcast,
      /*require_same_sizes=*/!channel_broadcast);

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
  int8_t* input1_ptr = input1_int8.data_ptr<int8_t>();
  int8_t* input2_ptr = input2_int8.data_ptr<int8_t>();

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

  // Note 1: The CMSIS-NN kernel implementation uses offsets which are always
  // added to the data, whereas zero_points are subtracted when dequantizing
  // (for the inputs) and added when quantizing (for the  output). Hence the
  // negative signs required here.

  // Note 2: It is not possible to perform the same rewrite as for mul for
  // addition. To preserve precision when rescaling the inputs, they are first
  // upscaled as much as possible, Hence the left_shift parameter required here.

  int32_t adds_per_loop = 0;
  if (channel_broadcast) {
    if (input1_int8.numel() < input2_int8.numel()) {
      std::swap<int32_t>(zp1, zp2);
      std::swap<int32_t>(input1_mult, input2_mult);
      std::swap<int>(input1_shift_val, input2_shift_val);
      std::swap<int8_t*>(input1_ptr, input2_ptr);
    }
    adds_per_loop = input1_int8.size(1);
  } else {
    adds_per_loop = out.numel();
  }

  for (int32_t broadcast_offset = 0; broadcast_offset < out.numel();
       broadcast_offset += adds_per_loop) {
    // Call CMSIS-NN kernel with precomputed parameters
    arm_cmsis_nn_status status = arm_elementwise_add_s8(
        input1_ptr + broadcast_offset,
        input2_ptr,
        -static_cast<int32_t>(zp1),
        input1_mult,
        input1_shift_val,
        -static_cast<int32_t>(zp2),
        input2_mult,
        input2_shift_val,
        left_shift,
        out.mutable_data_ptr<int8_t>() + broadcast_offset,
        static_cast<int32_t>(out_zp),
        output_mult,
        output_shift_val,
        activation_min,
        activation_max,
        adds_per_loop);

    if (status != ARM_CMSIS_NN_SUCCESS) {
      ET_LOG(
          Error,
          "quantized_add_out: arm_elementwise_add_s8 failed with status [%d]",
          status);

      context.fail(Error::Internal); // Fail the execution context
      return out;
    }
  }
  ET_LOG(
      Info,
      "quantized_add_out: Successfully completed with AoT-computed parameters!");

  return out;
}

} // namespace native
} // namespace cortex_m
