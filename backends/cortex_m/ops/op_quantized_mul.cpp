/*
 * Copyright 2025-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "cortex_m_ops_common.h"

namespace cortex_m {
namespace native {
namespace {

constexpr int32_t kInt8ActivationMin = std::numeric_limits<int8_t>::min();
constexpr int32_t kInt8ActivationMax = std::numeric_limits<int8_t>::max();

} // namespace

using KernelRuntimeContext = torch::executor::KernelRuntimeContext;

static Tensor& quantized_mul_out_impl(
    KernelRuntimeContext& context,
    const Tensor& input1_int8,
    const int64_t input1_zero_point,
    const Tensor& input2_int8,
    const int64_t input2_zero_point,
    const int64_t output_zero_point,
    const int64_t output_multiplier,
    const int64_t output_shift,
    ActivationLayout layout,
    const char* op_name,
    Tensor& out) {
  const int64_t channel_dim = layout == ActivationLayout::NHWCLogical ? 3 : 1;
  bool channel_broadcast =
      is_channel_broadcast(input1_int8, input2_int8, channel_dim);
  validate_cmsis_nn_tensor_requirements(
      input1_int8,
      input2_int8,
      out,
      ScalarType::Char,
      /*require_channels_last=*/
      channel_broadcast && layout == ActivationLayout::NCHWLogical,
      /*require_same_sizes=*/!channel_broadcast);
  if (layout == ActivationLayout::NHWCLogical) {
    ET_CHECK_MSG(
        input1_int8.dim() == 4 && input2_int8.dim() == 4 && out.dim() == 4,
        "%s: tensors must be 4-D",
        op_name);
    ET_CHECK_MSG(
        executorch::runtime::is_contiguous_dim_order(
            input1_int8.dim_order().data(), input1_int8.dim_order().size()) &&
            executorch::runtime::is_contiguous_dim_order(
                input2_int8.dim_order().data(),
                input2_int8.dim_order().size()) &&
            executorch::runtime::is_contiguous_dim_order(
                out.dim_order().data(), out.dim_order().size()),
        "%s: tensors must use contiguous dimension order",
        op_name);
  } else if (channel_broadcast) {
    ET_CHECK_MSG(
        is_channels_last_tensor(input1_int8) &&
            is_channels_last_tensor(input2_int8) &&
            is_channels_last_tensor(out),
        "%s: channel-broadcast tensors must use channels-last dimension order",
        op_name);
  }
  if (channel_broadcast) {
    const Tensor& full_input =
        input1_int8.numel() > input2_int8.numel() ? input1_int8 : input2_int8;
    ET_CHECK_MSG(
        out.sizes() == full_input.sizes(),
        "%s: output must have the broadcast result shape",
        op_name);
  }

  const int32_t kIdentityMultiplier(/*value=*/1);
  const int32_t kZeroShift(/*value=*/0);
  validate_quantization_params(
      input1_zero_point,
      kIdentityMultiplier,
      kZeroShift,
      input2_zero_point,
      kIdentityMultiplier,
      kZeroShift,
      output_zero_point,
      output_multiplier,
      output_shift);

  // Extract quantization parameters
  int8_t* input1_ptr = input1_int8.data_ptr<int8_t>();
  int8_t* input2_ptr = input2_int8.data_ptr<int8_t>();
  int32_t zp1 = static_cast<int32_t>(input1_zero_point);
  int32_t zp2 = static_cast<int32_t>(input2_zero_point);
  const int32_t out_zp = static_cast<int32_t>(output_zero_point);
  const int32_t output_mult = static_cast<int32_t>(output_multiplier);
  const int32_t output_shift_val = static_cast<int32_t>(output_shift);

  int32_t muls_per_loop = 0;

  if (channel_broadcast) {
    if (input1_int8.numel() < input2_int8.numel()) {
      std::swap<int32_t>(zp1, zp2);
      std::swap<int8_t*>(input1_ptr, input2_ptr);
    }

    muls_per_loop = input1_int8.size(channel_dim);
  } else {
    muls_per_loop = out.numel();
  }
  // Note 1: The CMSIS-NN kernel implementation uses offsets which are always
  // added to the data, whereas zero_points are subtracted when dequantizing
  // (for the inputs) and added when quantizing (for the  output). Hence the
  // negative signs required here.

  // Note 2: The following rewrite is used
  //    yq = y / scale_out + zp_out
  //    y = x_1*x_2
  //    x_i = scale_in_i * (xq_i - xq_i),  i = 1, 2
  //    ==>
  //    yq = (xq_1 - zp_in1) * (xq_2 - zp_in_2) * effective_scale + zp_out
  //    where
  //    effective_scale = (scale_in1 * scale_in2 / scale_out)
  // Hence no input quantization params required here.

  for (int32_t broadcast_offset = 0; broadcast_offset < out.numel();
       broadcast_offset += muls_per_loop) {
    // Call CMSIS-NN elementwise multiply kernel
    arm_cmsis_nn_status status = arm_elementwise_mul_s8(
        input1_ptr + broadcast_offset,
        input2_ptr,
        -static_cast<int32_t>(zp1),
        -static_cast<int32_t>(zp2),
        out.mutable_data_ptr<int8_t>() + broadcast_offset,
        static_cast<int32_t>(out_zp),
        output_mult,
        output_shift_val,
        kInt8ActivationMin,
        kInt8ActivationMax,
        muls_per_loop);

    if (status != ARM_CMSIS_NN_SUCCESS) {
      ET_LOG(
          Error,
          "%s: arm_elementwise_mul_s8 failed with status [%d]",
          op_name,
          status);
      context.fail(Error::Internal);
      return out;
    }
  }
  return out;
}

// cppcheck-suppress unusedFunction
Tensor& quantized_mul_out(
    KernelRuntimeContext& context,
    const Tensor& input1_int8,
    const int64_t input1_zero_point,
    const Tensor& input2_int8,
    const int64_t input2_zero_point,
    const int64_t output_zero_point,
    const int64_t output_multiplier,
    const int64_t output_shift,
    Tensor& out) {
  return quantized_mul_out_impl(
      context,
      input1_int8,
      input1_zero_point,
      input2_int8,
      input2_zero_point,
      output_zero_point,
      output_multiplier,
      output_shift,
      ActivationLayout::NCHWLogical,
      "quantized_mul_out",
      out);
}

// cppcheck-suppress unusedFunction
Tensor& quantized_mul_nhwc_out(
    KernelRuntimeContext& context,
    const Tensor& input1_int8,
    const int64_t input1_zero_point,
    const Tensor& input2_int8,
    const int64_t input2_zero_point,
    const int64_t output_zero_point,
    const int64_t output_multiplier,
    const int64_t output_shift,
    Tensor& out) {
  return quantized_mul_out_impl(
      context,
      input1_int8,
      input1_zero_point,
      input2_int8,
      input2_zero_point,
      output_zero_point,
      output_multiplier,
      output_shift,
      ActivationLayout::NHWCLogical,
      "quantized_mul_nhwc_out",
      out);
}

} // namespace native
} // namespace cortex_m
