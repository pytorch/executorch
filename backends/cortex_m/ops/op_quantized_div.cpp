/*
 * Copyright 2025-2026 Arm Limited and/or its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cmath>

#include "cortex_m_ops_common.h"

namespace cortex_m {
namespace native {
namespace {

template <typename T>
void quantized_div_typed(
    const Tensor& input1,
    const int32_t zp1,
    const Tensor& input2,
    const int32_t zp2,
    const int32_t out_zp,
    const float effective_scale,
    Tensor& out) {
  const T* input1_ptr = input1.data_ptr<T>();
  const T* input2_ptr = input2.data_ptr<T>();
  T* out_ptr = out.mutable_data_ptr<T>();

  constexpr int32_t kActivationMin = std::numeric_limits<T>::min();
  constexpr int32_t kActivationMax = std::numeric_limits<T>::max();

  const int64_t num_elements = out.numel();
  for (int64_t i = 0; i < num_elements; ++i) {
    const int32_t numerator = static_cast<int32_t>(input1_ptr[i]) - zp1;
    const int32_t denominator = static_cast<int32_t>(input2_ptr[i]) - zp2;

    const float quotient = (denominator != 0)
        ? static_cast<float>(numerator) / static_cast<float>(denominator)
        : 0.0f;

    int32_t result =
        static_cast<int32_t>(std::round(quotient * effective_scale)) + out_zp;
    result = std::max(kActivationMin, std::min(kActivationMax, result));
    out_ptr[i] = static_cast<T>(result);
  }
}

} // namespace

using KernelRuntimeContext = torch::executor::KernelRuntimeContext;

// CMSIS-NN has no integer elementwise-division primitive, so the quotient is
// evaluated in float. The effective scale (scale_in1 / (scale_in2 * scale_out))
// is carried in the AoT-computed output_multiplier/output_shift and
// reconstructed here, mirroring the softmax kernel's fixed-point-to-float
// reconstruction. Both int8 and int16 activations are supported.
// cppcheck-suppress unusedFunction
Tensor& quantized_div_out(
    KernelRuntimeContext& context,
    const Tensor& input1,
    const int64_t input1_zero_point,
    const Tensor& input2,
    const int64_t input2_zero_point,
    const int64_t output_zero_point,
    const int64_t output_multiplier,
    const int64_t output_shift,
    Tensor& out) {
  const ScalarType dtype = out.scalar_type();
  if (dtype != ScalarType::Char && dtype != ScalarType::Short) {
    ET_LOG(
        Error,
        "quantized_div: only int8 and int16 are supported, got %d",
        static_cast<int>(dtype));
    context.fail(Error::InvalidArgument);
    return out;
  }

  // Division is not commutative, so channel broadcasting (which relies on
  // operand swapping in quantized_mul) is unsupported: require equal shapes.
  validate_cmsis_nn_tensor_requirements(
      input1,
      input2,
      out,
      dtype,
      /*require_channels_last=*/false,
      /*require_same_sizes=*/true);

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

  const int32_t zp1 = static_cast<int32_t>(input1_zero_point);
  const int32_t zp2 = static_cast<int32_t>(input2_zero_point);
  const int32_t out_zp = static_cast<int32_t>(output_zero_point);

  const float effective_scale = std::ldexp(
      static_cast<float>(output_multiplier) / static_cast<float>(1LL << 31),
      static_cast<int>(output_shift));

  if (dtype == ScalarType::Char) {
    quantized_div_typed<int8_t>(
        input1, zp1, input2, zp2, out_zp, effective_scale, out);
  } else {
    quantized_div_typed<int16_t>(
        input1, zp1, input2, zp2, out_zp, effective_scale, out);
  }

  return out;
}

} // namespace native
} // namespace cortex_m
