/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/kernels/portable/cpu/util/elementwise_util.h>
#include <executorch/kernels/portable/cpu/util/kernel_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

// Include CMSIS-NN headers with C linkage
extern "C" {
#include "arm_nnfunctions.h"
}

using Tensor = torch::executor::Tensor;
using ScalarType = executorch::aten::ScalarType;
using Scalar = torch::executor::Scalar;
using Error = executorch::runtime::Error;

// Basic tensor type / layout validation and dimension order checking
inline void validate_quantized_tensor_types_and_dim_order(
    const Tensor& input1,
    const Tensor& input2,
    Tensor& output) {
  ET_CHECK_MSG(input1.scalar_type() == ScalarType::Char, "Input1 must be int8");
  ET_CHECK_MSG(input2.scalar_type() == ScalarType::Char, "Input2 must be int8");
  ET_CHECK_MSG(output.scalar_type() == ScalarType::Char, "Output must be int8");
  ET_CHECK_MSG(
      executorch::runtime::tensors_have_same_dim_order(input1, input2, output),
      "Tensors must have same dimension order");
}

/**
 * Validate quantization parameters for inputs and output.
 *
 * Checks that zero points fit in int8 range, multipliers fit in int32 range,
 * and shifts are within a valid bit-shift range (0-31).
 *
 * Ensures parameters comply with Ahead-Of-Time (AOT) quantization requirements
 * and CMSIS-NN kernel expectations.
 *
 * Raises errors via ET_KERNEL_CHECK if any check fails.
 */
inline void validate_quantization_params(
    const Scalar& zero_point1,
    const Scalar& multiplier1,
    const Scalar& shift1,
    const Scalar& zero_point2,
    const Scalar& multiplier2,
    const Scalar& shift2,
    const Scalar& output_zero_point,
    const Scalar& output_multiplier,
    const Scalar& output_shift,
    Tensor& output) {
  // Extract int64_t values from Scalars
  int64_t zp1_val = zero_point1.to<int64_t>();
  int64_t mult1_val = multiplier1.to<int64_t>();
  int64_t shift1_val = shift1.to<int64_t>();

  int64_t zp2_val = zero_point2.to<int64_t>();
  int64_t mult2_val = multiplier2.to<int64_t>();
  int64_t shift2_val = shift2.to<int64_t>();

  int64_t out_zp_val = output_zero_point.to<int64_t>();
  int64_t out_mult_val = output_multiplier.to<int64_t>();
  int64_t out_shift_val = output_shift.to<int64_t>();

  ET_CHECK_MSG(
      zp1_val >= std::numeric_limits<int8_t>::min() &&
          zp1_val <= std::numeric_limits<int8_t>::max(),
      "Zero point 1 must be in int8 range [Value: %d]",
      zp1_val);

  ET_CHECK_MSG(
      zp2_val >= std::numeric_limits<int8_t>::min() &&
          zp2_val <= std::numeric_limits<int8_t>::max(),
      "Zero point 2 must be in int8 range [Value: %d]",
      zp2_val);

  ET_CHECK_MSG(
      out_zp_val >= std::numeric_limits<int8_t>::min() &&
          out_zp_val <= std::numeric_limits<int8_t>::max(),
      "Output zero point must be in int8 range [Value: %d]",
      out_zp_val);

  // Check multipliers fit in int32 range (AOT quantize_multiplier_aot clamps to
  // int32)
  ET_CHECK_MSG(
      mult1_val >= std::numeric_limits<int32_t>::min() &&
          mult1_val <= std::numeric_limits<int32_t>::max(),
      "Multiplier 1 must be in int32 range [Value: %d]",
      mult1_val);

  ET_CHECK_MSG(
      mult2_val >= std::numeric_limits<int32_t>::min() &&
          mult2_val <= std::numeric_limits<int32_t>::max(),
      "Multiplier 2 must be in int32 range [Value: %d]",
      mult2_val);

  ET_CHECK_MSG(
      out_mult_val >= std::numeric_limits<int32_t>::min() &&
          out_mult_val <= std::numeric_limits<int32_t>::max(),
      "Output multiplier must be in int32 range [Value: %d]",
      out_mult_val);

  ET_CHECK_MSG(
      shift1_val >= -31 && shift1_val <= 31,
      "Shift 1 must be in range [-31, 31] [Value: %d]",
      shift1_val);

  ET_CHECK_MSG(
      shift2_val >= -31 && shift2_val <= 31,
      "Shift 2 must be in range [-31, 31] [Value: %d]",
      shift2_val);

  ET_CHECK_MSG(
      out_shift_val >= -31 && out_shift_val <= 31,
      "Output shift must be in range [-31, 31] [Value: %d]",
      out_shift_val);
}

inline Error resize_to_broadcast_target_size_quantized(
    const Tensor& input1,
    const Tensor& input2,
    Tensor& output) {
  static constexpr int kTensorDimensionLimit = 5;

  int inp1_shape[kTensorDimensionLimit];
  int inp2_shape[kTensorDimensionLimit];
  int out_shape[kTensorDimensionLimit];

  int max_dim = std::max({input1.dim(), input2.dim(), output.dim()});
  max_dim = std::min(max_dim, kTensorDimensionLimit);

  // Initialize shapes with 1s for padding
  for (int i = 0; i < max_dim; i++) {
    inp1_shape[i] = 1;
    inp2_shape[i] = 1;
    out_shape[i] = 1;
  }

  int offset_inp1 = max_dim - input1.dim();
  int offset_inp2 = max_dim - input2.dim();
  int offset_out = max_dim - output.dim();

  for (int i = 0; i < input1.dim(); i++) {
    inp1_shape[i + offset_inp1] = input1.size(i);
  }
  for (int i = 0; i < input2.dim(); i++) {
    inp2_shape[i + offset_inp2] = input2.size(i);
  }
  for (int i = 0; i < output.dim(); i++) {
    out_shape[i + offset_out] = output.size(i);
  }

  // Compute broadcasted shape (use existing get_broadcast_target_size or
  // equivalent)
  Tensor::SizesType expected_output_size[kTensorDimensionLimit];
  size_t expected_output_dim = 0;

  auto err = torch::executor::get_broadcast_target_size(
      input1,
      input2,
      expected_output_size,
      kTensorDimensionLimit,
      &expected_output_dim);
  if (err != Error::Ok) {
    return err;
  }

  // Resize output tensor to broadcasted shape
  return executorch::runtime::resize_tensor(
      output, {expected_output_size, expected_output_dim});
}
