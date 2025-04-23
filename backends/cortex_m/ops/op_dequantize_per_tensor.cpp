/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/runtime/kernel/kernel_includes.h>
#include <cinttypes>

// Check for Helium/MVE support
#if defined(__ARM_FEATURE_MVE) && (__ARM_FEATURE_MVE & 1)
#include <arm_mve.h>
#define HAS_HELIUM_SIMD 1
#endif

namespace cortex_m {
namespace native {

using Tensor = executorch::aten::Tensor;
using ScalarType = executorch::aten::ScalarType;
using KernelRuntimeContext = torch::executor::KernelRuntimeContext;

namespace {

/**
 * Asserts that the parameters are valid for float to int8 quantization.
 */
void check_dequantize_args(
    const Tensor& input,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  // Ensure input is char type
  ET_CHECK_MSG(
      input.scalar_type() == ScalarType::Char,
      "input.scalar_type() %" PRId8 " is not char type",
      static_cast<int8_t>(input.scalar_type()));

  // Check output dtype is float
  ET_CHECK_MSG(
      out.scalar_type() == ScalarType::Float,
      "out.scalar_type() %" PRId8 " is not float",
      static_cast<int8_t>(out.scalar_type()));

  // Check dtype is int8 (Char)
  ET_CHECK_MSG(
      dtype == ScalarType::Char,
      "dtype %" PRId8 " is not int8 (Char)",
      static_cast<int8_t>(dtype));

  // Validate quant_min and quant_max for int8
  int32_t quant_min_lower_bound = std::numeric_limits<int8_t>::min();
  int32_t quant_max_upper_bound = std::numeric_limits<int8_t>::max();

  ET_CHECK_MSG(
      quant_min >= quant_min_lower_bound,
      "quant_min out of bound for int8, expected quant_min_lower_bound: %" PRId32
      " actual quant_min: %" PRId64,
      quant_min_lower_bound,
      quant_min);

  ET_CHECK_MSG(
      quant_max <= quant_max_upper_bound,
      "quant_max out of bound for int8, expected quant_max_upper_bound: %" PRId32
      " actual quant_max: %" PRId64,
      quant_max_upper_bound,
      quant_max);
}

/**
 * Scalar implementation of quantization for a single value.
 */
template <typename K, typename T>
T dequantize_val(
    float scale,
    int32_t zero_point,
    K value,
    int64_t quant_min,
    int64_t quant_max) {
  (void)quant_min;
  (void)quant_max;
  return static_cast<T>((static_cast<int32_t>(value) - zero_point) * scale);
}

} // namespace

Tensor& dequantize_per_tensor_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  // Ignore context for now
  (void)context;

  // Resize output tensor to match input dimensions
  torch::executor::Error err = resize_tensor(out, input.sizes());
  ET_CHECK_MSG(
      err == torch::executor::Error::Ok,
      "Failed to resize out Tensor in dequantize_per_tensor_out");

  // Validate input parameters
  check_dequantize_args(input, quant_min, quant_max, dtype, out);

  // Pre-compute inverse scale for better performance
  int32_t zp = static_cast<int32_t>(zero_point);
  int32_t qmin = static_cast<int32_t>(quant_min);
  int32_t qmax = static_cast<int32_t>(quant_max);

  // Get pointers to input and output data
  const int8_t* input_data = input.const_data_ptr<int8_t>();
  float* out_data = out.mutable_data_ptr<float>();
  const size_t numel = input.numel();

#if defined(HAS_HELIUM_SIMD)
// Helium MVE implementation for float32 to int8 quantization
#Error "Implement MVE version!"
#else
  // Scalar implementation for float32 to int8 quantization
  for (size_t i = 0; i < numel; i++) {
    out_data[i] =
        dequantize_val<int8_t, float>(scale, zp, input_data[i], qmin, qmax);
  }
#endif

  return out;
}

Tensor& dequantize_per_tensor_out(
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  KernelRuntimeContext context;
  return dequantize_per_tensor_out(
      context, input, scale, zero_point, quant_min, quant_max, dtype, out);
}

} // namespace native
} // namespace cortex_m
