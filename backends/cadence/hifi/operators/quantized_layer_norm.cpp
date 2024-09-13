/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/reference/kernels/kernels.h>
#include <executorch/runtime/kernel/kernel_includes.h>

#include <algorithm>
#include <cmath>
#include <tuple>

using Tensor = exec_aten::Tensor;
using executorch::runtime::KernelRuntimeContext;

namespace impl {
namespace HiFi {
namespace native {

// Compute quantized layer_norm. The current implementation assumes that the
// input is per-tensor quantized.
template <typename T>
void quantized_layer_norm_(
    const Tensor& input,
    float input_scale,
    int64_t input_zero_point,
    const Tensor& weight,
    const Tensor& bias,
    double eps,
    double output_scale,
    int64_t output_zero_point,
    Tensor& out) {
  // Get the raw pointers to input, output, weight, and bias
  const T* __restrict__ in_data = input.const_data_ptr<T>();
  T* __restrict__ out_data = out.mutable_data_ptr<T>();
  const float* __restrict__ weight_data = weight.const_data_ptr<float>();
  const float* __restrict__ bias_data = bias.const_data_ptr<float>();

  float output_inv_scale = XT_RECIP_S(output_scale);

  size_t last_dim = input.size(input.dim() - 1);
  size_t leading_dims = getLeadingDims(input, input.dim() - 1);

  // Visualize the input tensor as a set of 1d vectors, and compute the
  // layer_norm for each vector.
  for (size_t i = 0; i < leading_dims; ++i) {
    const T* __restrict__ x = in_data + i * last_dim;
    T* __restrict__ y = out_data + i * last_dim;

    // compute sum and squared sum. The fp32 sum can be approximated as:
    // (X_1 - in_zero_point) * in_scale + (X_2 - in_zero_point) * in_scale + ...
    // (X_N - in_zero_point) * in_scale.
    int32_t sum = 0;
    int32_t sq_sum = last_dim * input_zero_point * input_zero_point;
#pragma simd
    for (size_t j = 0; j < last_dim; ++j) {
      int32_t val = x[j];
      sum += val;
      sq_sum += val * val;
    }
    sq_sum -= (2 * sum * input_zero_point);
    sum -= (last_dim * input_zero_point);

    float mean = XT_DIV_S(XT_MUL_S(input_scale, sum), last_dim);
    float variance =
        XT_DIV_S(
            XT_MUL_S(sq_sum, XT_MUL_S(input_scale, input_scale)), last_dim) -
        XT_MUL_S(mean, mean);
    float inv_std = XT_RECIP_S(XT_SQRT_S(XT_ADD_S(variance, (float)eps)));

    // y = (x - mean) / std * kGamma + kBeta
#pragma simd
    for (size_t j = 0; j < last_dim; ++j) {
      // Since X is quantized, we dequantize it, compute fp32 result, and
      // quantize the result to an int8/uint8 value.
      float val = kernels::dequantize<T>(x[j], input_scale, input_zero_point);
      val = (val - mean) * inv_std * weight_data[j] + bias_data[j];
      y[j] = kernels::quantize<T>(val, output_inv_scale, output_zero_point);
    }
  }
}

// Compute quantized layer_norm. The current implementation assumes that the
// input is per-tensor quantized.
template <typename T>
void quantized_layer_norm_(
    const Tensor& input,
    const Tensor& in_scale,
    const Tensor& in_zero_point,
    const Tensor& weight,
    const Tensor& bias,
    double eps,
    double output_scale,
    int64_t output_zero_point,
    Tensor& out) {
  // Extract the zero point and scale for input tensor.
  float input_scale = in_scale.const_data_ptr<float>()[0];
  int64_t input_zero_point = in_zero_point.const_data_ptr<int64_t>()[0];

  // Call other overload
  quantized_layer_norm_<T>(
      input,
      input_scale,
      input_zero_point,
      weight,
      bias,
      eps,
      output_scale,
      output_zero_point,
      out);
}

void quantized_layer_norm_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& in_scale,
    const Tensor& in_zero_point,
    const exec_aten::IntArrayRef normalized_shape,
    const Tensor& weight,
    const Tensor& bias,
    double eps,
    double output_scale,
    int64_t output_zero_point,
    Tensor& out) {
  if (input.scalar_type() == exec_aten::ScalarType::Byte) {
    quantized_layer_norm_<uint8_t>(
        input,
        in_scale,
        in_zero_point,
        weight,
        bias,
        eps,
        output_scale,
        output_zero_point,
        out);
  } else if (input.scalar_type() == exec_aten::ScalarType::Char) {
    quantized_layer_norm_<int8_t>(
        input,
        in_scale,
        in_zero_point,
        weight,
        bias,
        eps,
        output_scale,
        output_zero_point,
        out);
  } else {
    ET_CHECK_MSG(false, "Unhandled input dtype %hhd", input.scalar_type());
  }
}

}; // namespace native
}; // namespace HiFi
}; // namespace impl
