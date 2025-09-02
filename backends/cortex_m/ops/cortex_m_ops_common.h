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

using Tensor = torch::executor::Tensor;
using ScalarType = executorch::aten::ScalarType;
using Scalar = torch::executor::Scalar;
using Error = executorch::runtime::Error;

// Basic tensor type / layout validation and dimension order checking
inline void validate_cmsis_nn_tensor_requirements(
    const Tensor& input1,
    const Tensor& input2,
    Tensor& output,
    ScalarType expected_dtype = ScalarType::Char,
    bool require_channels_last = false) {
  // Basic dtype validation
  ET_CHECK_MSG(
      input1.scalar_type() == expected_dtype,
      "Input1 dtype must be %hhd",
      expected_dtype);
  ET_CHECK_MSG(
      input2.scalar_type() == expected_dtype,
      "Input2 dtype must be %hhd",
      expected_dtype);
  ET_CHECK_MSG(
      output.scalar_type() == expected_dtype,
      "Output dtype must be %hhd",
      expected_dtype);

  // Dim order consistency
  ET_CHECK_MSG(
      executorch::runtime::tensors_have_same_dim_order(input1, input2, output),
      "Tensors must have same dimension order");

  // TBD: Validate memory alignment (CMSIS-NN requirement)
}

inline void validate_single_quant_params(
    const Scalar& zero_point,
    const Scalar& multiplier,
    const Scalar& shift,
    const char* param_name) {
  int64_t zp_val = zero_point.to<int64_t>();
  int64_t mult_val = multiplier.to<int64_t>();
  int64_t shift_val = shift.to<int64_t>();

  ET_CHECK_MSG(
      zp_val >= std::numeric_limits<int8_t>::min() &&
          zp_val <= std::numeric_limits<int8_t>::max(),
      "%s zero point must be in int8 range [Value: %d]",
      param_name,
      zp_val);

  ET_CHECK_MSG(
      mult_val >= std::numeric_limits<int32_t>::min() &&
          mult_val <= std::numeric_limits<int32_t>::max(),
      "%s multiplier must be in int32 range [Value: %d]",
      param_name,
      mult_val);

  ET_CHECK_MSG(
      shift_val >= -31 && shift_val <= 31,
      "%s shift must be in range [-31, 31] [Value: %d]",
      param_name,
      shift_val);
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
  validate_single_quant_params(
      zero_point1, multiplier1, shift1, "Single quant Input1");
  validate_single_quant_params(
      zero_point2, multiplier2, shift2, "Single quant Input2");
  validate_single_quant_params(
      output_zero_point,
      output_multiplier,
      output_shift,
      "Single quant Output");
}

inline Error resize_to_broadcast_target_size(
    const Tensor& input1,
    const Tensor& input2,
    Tensor& output) {
  static constexpr int kTensorDimensionLimit = 5;
  Tensor::SizesType expected_output_size[kTensorDimensionLimit];
  size_t expected_output_dim = 0;
  auto err = torch::executor::get_broadcast_target_size(
      input1,
      input2,
      expected_output_size,
      kTensorDimensionLimit,
      &expected_output_dim);

  if (err != Error::Ok)
    return err;

  return executorch::runtime::resize_tensor(
      output, {expected_output_size, expected_output_dim});
}

/**
 * Convert Scalar to CMSIS-NN int32 format
 * For multipliers, zero_points, etc. from quantize_multiplier_aot
 */
inline int32_t extractScalarToInt32(const Scalar& scalar_value) {
  return static_cast<int32_t>(scalar_value.to<int64_t>());
}

/**
 * Convert Scalar to CMSIS-NN int format
 * For shift values from quantize_multiplier_aot
 */
inline int extractScalarToInt(const Scalar& scalar_value) {
  return static_cast<int>(scalar_value.to<int64_t>());
}
