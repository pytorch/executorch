/*
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
namespace {

constexpr int32_t kInt8ActivationMin = std::numeric_limits<int8_t>::min();
constexpr int32_t kInt8ActivationMax = std::numeric_limits<int8_t>::max();

} // namespace

using KernelRuntimeContext = torch::executor::KernelRuntimeContext;

Tensor& quantized_mul_out(
    KernelRuntimeContext& context,
    const Tensor& input1_int8,
    const Scalar& input1_zero_point,
    const Tensor& input2_int8,
    const Scalar& input2_zero_point,
    const Scalar& output_zero_point,
    const Scalar& output_multiplier,
    const Scalar& output_shift,
    Tensor& out) {
  // Validate tensor types and quantization parameters
  validate_cmsis_nn_tensor_requirements(input1_int8, input2_int8, out);

  const Scalar kIdentityMultiplier(/*value=*/1);
  const Scalar kZeroShift(/*value=*/0);
  validate_quantization_params(
      input1_zero_point,
      kIdentityMultiplier,
      kZeroShift,
      input2_zero_point,
      kIdentityMultiplier,
      kZeroShift,
      output_zero_point,
      output_multiplier,
      output_shift,
      out);

  // Extract quantization parameters
  const int32_t zp1 = extractScalarToInt32(input1_zero_point);
  const int32_t zp2 = extractScalarToInt32(input2_zero_point);
  const int32_t out_zp = extractScalarToInt32(output_zero_point);
  const int32_t output_mult = extractScalarToInt32(output_multiplier);
  const int32_t output_shift_val = extractScalarToInt(output_shift);

  // Call CMSIS-NN elementwise multiply kernel
  arm_cmsis_nn_status status = arm_elementwise_mul_s8(
      input1_int8.const_data_ptr<int8_t>(),
      input2_int8.const_data_ptr<int8_t>(),
      -static_cast<int32_t>(zp1),
      -static_cast<int32_t>(zp2),
      out.mutable_data_ptr<int8_t>(),
      static_cast<int32_t>(out_zp),
      output_mult,
      output_shift_val,
      kInt8ActivationMin,
      kInt8ActivationMax,
      static_cast<int32_t>(out.numel()));

  if (status != ARM_CMSIS_NN_SUCCESS) {
    ET_LOG(
        Error,
        "quantized_mul_out: arm_elementwise_mul_s8 failed with status [%d]",
        status);
    context.fail(Error::Internal);
    return out;
  }

  return out;
}

} // namespace native
} // namespace cortex_m
