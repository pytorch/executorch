/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * Copyright 2025-2026 Arm Limited and/or its affiliates.
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

#include <limits>
#include <optional>

extern "C" {
#include "arm_nn_types.h"
}

using Tensor = torch::executor::Tensor;
using ScalarType = executorch::aten::ScalarType;
using Scalar = torch::executor::Scalar;
using Error = executorch::runtime::Error;
using Int64ArrayRef = executorch::aten::ArrayRef<int64_t>;
using KernelRuntimeContext = torch::executor::KernelRuntimeContext;

// From arm_nn_math_types.h
#define ARM_NN_Q31_MAX ((int32_t)(0x7FFFFFFFL))
#define ARM_NN_Q31_MIN ((int32_t)(0x80000000L))

// Basic tensor type / layout validation and dimension order checking
inline void validate_cmsis_nn_tensor_requirements(
    const Tensor& input1,
    const Tensor& input2,
    Tensor& output,
    ScalarType expected_dtype = ScalarType::Char,
    bool require_channels_last = false,
    bool require_same_sizes = true) {
  // Basic dtype validation
  ET_CHECK_MSG(
      input1.scalar_type() == expected_dtype,
      "Input1 dtype must be %hhd, got %hhd",
      expected_dtype,
      input1.scalar_type());
  ET_CHECK_MSG(
      input2.scalar_type() == expected_dtype,
      "Input2 dtype must be %hhd, got %hhd",
      expected_dtype,
      input2.scalar_type());
  ET_CHECK_MSG(
      output.scalar_type() == expected_dtype,
      "Output dtype must be %hhd, got %hhd",
      expected_dtype,
      output.scalar_type());
  if (require_same_sizes) {
    ET_CHECK_MSG(
        input1.sizes() == input2.sizes(),
        "Input1 and Input2 must have the same sizes");
    ET_CHECK_MSG(
        output.sizes() == input1.sizes(),
        "Output must have the same sizes as inputs");
  }

  // TBD (#16032): Validate dim_order
  // TBD: Validate memory alignment (CMSIS-NN requirement)
}

inline void validate_single_quant_params(
    const int64_t zero_point,
    const int64_t multiplier,
    const int64_t shift,
    const char* param_name) {
  ET_CHECK_MSG(
      multiplier >= std::numeric_limits<int32_t>::min() &&
          multiplier <= std::numeric_limits<int32_t>::max(),
      "%s multiplier must be in int32 range [Value: %d]",
      param_name,
      multiplier);

  ET_CHECK_MSG(
      shift >= -31 && shift <= 31,
      "%s shift must be in range [-31, 31] [Value: %d]",
      param_name,
      shift);
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
    const int64_t zero_point1,
    const int64_t multiplier1,
    const int64_t shift1,
    const int64_t zero_point2,
    const int64_t multiplier2,
    const int64_t shift2,
    const int64_t output_zero_point,
    const int64_t output_multiplier,
    const int64_t output_shift,
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

inline bool is_channels_last_tensor(const Tensor& tensor) {
  if (tensor.dim() != 4) {
    return false;
  }

  // When channels or spatial dims are 1 the layout information is ambiguous.
  if (tensor.size(1) == 1 || (tensor.size(2) == 1 && tensor.size(3) == 1)) {
    return true;
  }

  constexpr executorch::aten::DimOrderType kChannelsLastDimOrder[] = {
      0, 2, 3, 1};
  executorch::aten::ArrayRef<executorch::aten::DimOrderType>
      channels_last_order(kChannelsLastDimOrder, 4);

  return tensor.dim_order() == channels_last_order;
}

inline bool is_channel_broadcast(const Tensor& tensor1, const Tensor& tensor2) {
  if (tensor1.dim() != tensor2.dim()) {
    return false;
  }

  if (tensor1.dim() != 4) {
    return false;
  }

  if (tensor1.size(1) != tensor2.size(1)) {
    return false;
  }

  const bool tensor1_channels_only = tensor1.numel() == tensor1.size(1);
  const bool tensor2_channels_only = tensor2.numel() == tensor2.size(1);

  return tensor1_channels_only || tensor2_channels_only;
}

inline bool check_int32_within_range(
    KernelRuntimeContext& context,
    const char* op_name,
    int64_t value,
    const char* value_name,
    int32_t& out_value) {
  if (value < std::numeric_limits<int32_t>::min() ||
      value > std::numeric_limits<int32_t>::max()) {
    ET_LOG(
        Error,
        "%s: %s value (%ld) exceeds int32_t range",
        op_name,
        value_name,
        value);
    context.fail(Error::InvalidArgument);
    return false;
  }
  out_value = static_cast<int32_t>(value);
  return true;
}

struct CmsisPool2DConfig {
  cmsis_nn_pool_params pool_params;
  cmsis_nn_dims input_dims;
  cmsis_nn_dims filter_dims;
  cmsis_nn_dims output_dims;
};

inline bool prepare_cmsis_pool2d_config(
    KernelRuntimeContext& context,
    const char* op_name,
    const Tensor& input,
    Tensor& output,
    const Int64ArrayRef& kernel_size,
    const Int64ArrayRef& stride,
    const Int64ArrayRef& padding,
    const Int64ArrayRef& dilation,
    bool ceil_mode,
    int64_t activation_min,
    int64_t activation_max,
    CmsisPool2DConfig& config,
    bool require_channels_last = true,
    bool allow_ceil_mode = false) {
  if (input.dim() != 4 || output.dim() != 4) {
    ET_LOG(Error, "%s: tensors must be 4-D", op_name);
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (input.scalar_type() != ScalarType::Char ||
      output.scalar_type() != ScalarType::Char) {
    ET_LOG(Error, "%s: tensors must be int8", op_name);
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (input.size(0) != output.size(0) || input.size(1) != output.size(1)) {
    ET_LOG(
        Error,
        "%s: batch and channel dimensions must match between input and output",
        op_name);
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (require_channels_last) {
    if (!is_channels_last_tensor(input) || !is_channels_last_tensor(output)) {
      ET_LOG(
          Error, "%s: tensors must use channels_last dimension order", op_name);
      context.fail(Error::InvalidArgument);
      return false;
    }
  }

  auto check_tuple_len = [&](const Int64ArrayRef& arr,
                             const char* name) -> bool {
    if (arr.size() != 2) {
      ET_LOG(Error, "%s: %s must have length 2", op_name, name);
      context.fail(Error::InvalidArgument);
      return false;
    }
    return true;
  };

  if (!check_tuple_len(kernel_size, "kernel_size") ||
      !check_tuple_len(stride, "stride") ||
      !check_tuple_len(padding, "padding") ||
      !check_tuple_len(dilation, "dilation")) {
    return false;
  }

  if (!allow_ceil_mode && ceil_mode) {
    ET_LOG(Error, "%s: ceil_mode=True is not supported", op_name);
    context.fail(Error::InvalidArgument);
    return false;
  }

  if (activation_min > activation_max) {
    ET_LOG(
        Error,
        "%s: activation_min (%lld) must be <= activation_max (%lld)",
        op_name,
        static_cast<long long>(activation_min),
        static_cast<long long>(activation_max));
    context.fail(Error::InvalidArgument);
    return false;
  }

  int32_t activation_min_i32, activation_max_i32;
  if (!check_int32_within_range(
          context,
          op_name,
          activation_min,
          "activation_min",
          activation_min_i32) ||
      !check_int32_within_range(
          context,
          op_name,
          activation_max,
          "activation_max",
          activation_max_i32)) {
    return false;
  }

  int32_t kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w;
  if (!check_int32_within_range(
          context, op_name, kernel_size[0], "kernel_size[0]", kernel_h) ||
      !check_int32_within_range(
          context, op_name, kernel_size[1], "kernel_size[1]", kernel_w) ||
      !check_int32_within_range(
          context, op_name, stride[0], "stride[0]", stride_h) ||
      !check_int32_within_range(
          context, op_name, stride[1], "stride[1]", stride_w) ||
      !check_int32_within_range(
          context, op_name, padding[0], "padding[0]", pad_h) ||
      !check_int32_within_range(
          context, op_name, padding[1], "padding[1]", pad_w) ||
      !check_int32_within_range(
          context, op_name, dilation[0], "dilation[0]", dil_h) ||
      !check_int32_within_range(
          context, op_name, dilation[1], "dilation[1]", dil_w)) {
    return false;
  }

  if (dil_h != 1 || dil_w != 1) {
    ET_LOG(Error, "%s: dilation other than 1 is unsupported", op_name);
    context.fail(Error::InvalidArgument);
    return false;
  }

  int32_t batch, channels, input_h, input_w, output_h, output_w;
  if (!check_int32_within_range(
          context, op_name, input.size(0), "input batch", batch) ||
      !check_int32_within_range(
          context, op_name, input.size(1), "input channels", channels) ||
      !check_int32_within_range(
          context, op_name, input.size(2), "input height", input_h) ||
      !check_int32_within_range(
          context, op_name, input.size(3), "input width", input_w) ||
      !check_int32_within_range(
          context, op_name, output.size(2), "output height", output_h) ||
      !check_int32_within_range(
          context, op_name, output.size(3), "output width", output_w)) {
    return false;
  }

  config.input_dims = cmsis_nn_dims{batch, input_h, input_w, channels};
  config.filter_dims = cmsis_nn_dims{1, kernel_h, kernel_w, 1};
  config.output_dims = cmsis_nn_dims{batch, output_h, output_w, channels};
  config.pool_params.padding.h = pad_h;
  config.pool_params.padding.w = pad_w;
  config.pool_params.stride.h = stride_h;
  config.pool_params.stride.w = stride_w;
  config.pool_params.activation.min = activation_min_i32;
  config.pool_params.activation.max = activation_max_i32;

  return true;
}

// Refer to CMSIS-NN 'arm_nn_requantize' implementation for details:
// https://github.com/ARM-software/CMSIS-NN/blob/main/Include/arm_nnsupportfunctions.h#L1625
// multiplier: Range {ARM_NN_Q31_MIN + 1, Q32_MAX}
// shift     : Range {-31, 30}
inline bool validate_per_channel_quant_params(
    const Int64ArrayRef multipliers,
    const Int64ArrayRef shifts,
    int num_channels) {
  for (int i = 0; i < num_channels; ++i) {
    // Multiplier: {ARM_NN_Q31_MIN + 1, ARM_NN_Q31_MAX}
    if (multipliers[i] <= ARM_NN_Q31_MIN || multipliers[i] > ARM_NN_Q31_MAX) {
      ET_LOG(
          Error,
          "weight_multiplier[%d] out of CMSIS-NN range: %d",
          i,
          multipliers[i]);
      return false;
    }
    // Shift: {-31, 30} for arm_nn_requantize
    if (shifts[i] < -31 || shifts[i] > 30) {
      ET_LOG(Error, "weight_shift[%d] out of range: %d", i, shifts[i]);
      return false;
    }
  }
  return true;
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
