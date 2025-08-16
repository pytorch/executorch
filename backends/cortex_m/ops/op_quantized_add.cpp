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
  validate_quantized_inputs(context, input1_int8, input2_int8, out);

  ET_LOG(
      Info,
      "quantized_add_out: input1_int8.sizes() = %zu",
      input1_int8.sizes().size());

  // FIX: Use template types that ExecutorTorch definitely provides
  // Use to<int64_t>() and to<double>() which are commonly instantiated
  int32_t zp1 = static_cast<int32_t>(input1_zero_point.to<int64_t>());
  int32_t input1_mult = static_cast<int32_t>(input1_multiplier.to<int64_t>());
  int input1_shift_val = static_cast<int>(input1_shift.to<int64_t>());

  int32_t zp2 = static_cast<int32_t>(input2_zero_point.to<int64_t>());
  int32_t input2_mult = static_cast<int32_t>(input2_multiplier.to<int64_t>());
  int input2_shift_val = static_cast<int>(input2_shift.to<int64_t>());

  int32_t out_zp = static_cast<int32_t>(output_zero_point.to<int64_t>());
  int32_t output_mult = static_cast<int32_t>(output_multiplier.to<int64_t>());
  int output_shift_val = static_cast<int>(output_shift.to<int64_t>());

  // Left shift to maximize precision (tune as needed)
  const int32_t left_shift = 20;
  const int32_t activation_min = std::numeric_limits<int8_t>::min();
  const int32_t activation_max = std::numeric_limits<int8_t>::max();

  // Resize output tensor to match input shape
  auto err = torch::executor::resize_tensor(out, input1_int8.sizes());
  if (err != executorch::runtime::Error::Ok) {
    ET_LOG(
        Error,
        "quantized_add_out: resize_tensor failed with error code [%d]",
        static_cast<int>(err));
    std::memset(out.mutable_data_ptr<int8_t>(), 0, out.nbytes());
    return out;
  }

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
      static_cast<int32_t>(zp1),
      input1_mult,
      input1_shift_val,
      static_cast<int32_t>(zp2),
      input2_mult,
      input2_shift_val,
      left_shift,
      out.mutable_data_ptr<int8_t>(),
      static_cast<int32_t>(out_zp),
      output_mult,
      output_shift_val,
      static_cast<int32_t>(out.numel()),
      activation_min,
      activation_max);

  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(
        Error,
        "quantized_add_out: arm_elementwise_add_s8 failed with status [%d]",
        status);
    std::memset(out.mutable_data_ptr<int8_t>(), 0, out.nbytes());
  } else {
    ET_LOG(
        Info,
        "quantized_add_out: Successfully completed with AoT-computed parameters!");
  }

  return out;
}

// Stub Implementation: Non-out variant for compatibility (functional variant)
// EXIR/ExecuTorch runs an out-variant pass that converts
// .default operations to .out variants before memory planning.
// In the pass we are calling quantized_add's default variant
// but ExecuTorch's kernel dispatch mechanism will end up calling the out
// variant. This stub is to make sure that compiler doesn't complain.
Tensor quantized_add(
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
    const Scalar& output_shift) {
  ET_LOG(Info, "quantized_add: input1_int8.sizes() = %zu", input1_int8.sizes());

  // Crash on Debug builds if invoked
  assert(False);
  // This is to make sure compiler doesn't complain.
  return const_cast<Tensor&>(input1_int8);
}

} // namespace native
} // namespace cortex_m
