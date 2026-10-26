/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_quantized_layer_norm.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#include <executorch/backends/cadence/generic/kernels/kernels.h>
#include <executorch/backends/cadence/generic/operators/cadence_type_util.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>

namespace impl {
namespace generic {
namespace native {

using ::executorch::aten::IntArrayRef;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;
using ::impl::generic::kernels::dequantize;
using ::impl::generic::kernels::quantize;

// Compute quantized layer_norm. The current implementation assumes that the
// input is per-tensor quantized.
template <typename T>
void quantized_layer_norm_per_tensor_(
    const Tensor& input,
    double input_scale,
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

  float output_inv_scale = 1.0f / output_scale;

  size_t last_dim = input.size(input.dim() - 1);
  size_t leading_dims =
      ::executorch::runtime::getLeadingDims(input, input.dim() - 1);

  // Visualize the input tensor as a set of 1d vectors, and compute the
  // layer_norm for each vector.
  for (size_t i = 0; i < leading_dims; ++i) {
    const T* x = in_data + i * last_dim;
    T* y = out_data + i * last_dim;

    // 16-bit squares overflow an int32 sum after two elements.
    using acc_t = std::conditional_t<sizeof(T) == 1, int32_t, int64_t>;
    acc_t sum = 0;
    acc_t sq_sum = 0;
    for (size_t j = 0; j < last_dim; ++j) {
      const int32_t val = x[j];
      sum += val;
      sq_sum += static_cast<acc_t>(val) * val;
    }

    // Shift raw sums to zero-point-centered sums, then form the variance
    // numerator N * sum(v^2) - sum(v)^2 exactly in integers: doing
    // E[x^2] - E[x]^2 in fp32 cancels catastrophically for 16-bit inputs.
    const int64_t n = static_cast<int64_t>(last_dim);
    const int64_t zp = input_zero_point;
    const int64_t centered_sum = static_cast<int64_t>(sum) - n * zp;
    const int64_t centered_sq_sum =
        static_cast<int64_t>(sq_sum) - 2 * zp * sum + n * zp * zp;
    const int64_t var_numerator =
        n * centered_sq_sum - centered_sum * centered_sum;

    const double n_d = static_cast<double>(n);
    float mean = input_scale * centered_sum / n_d;
    float variance = var_numerator / (n_d * n_d) * (input_scale * input_scale);
    float inv_std = 1.0f / std::sqrt(variance + eps);

    // y = (x - mean) / std * kGamma + kBeta
    for (size_t j = 0; j < last_dim; ++j) {
      // y[j] = (x[j] - mean) / std * kGamma + kBeta;
      // Since X is quantized, we dequantize it, compute fp32 result, and
      // quantize the result back to T.
      float val = dequantize<T>(x[j], input_scale, input_zero_point);

      val = (val - mean) * inv_std * weight_data[j] + bias_data[j];
      y[j] = quantize<T>(val, output_inv_scale, output_zero_point);
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
  quantized_layer_norm_per_tensor_<T>(
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

Tensor& quantized_layer_norm_out(
    ET_UNUSED ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& in_scale,
    const Tensor& in_zero_point,
    ET_UNUSED const IntArrayRef normalized_shape,
    const Tensor& weight,
    const Tensor& bias,
    double eps,
    double output_scale,
    int64_t output_zero_point,
    Tensor& out) {
#define typed_quantized_layer_norm(ctype, dtype) \
  case ScalarType::dtype: {                      \
    quantized_layer_norm_<ctype>(                \
        input,                                   \
        in_scale,                                \
        in_zero_point,                           \
        weight,                                  \
        bias,                                    \
        eps,                                     \
        output_scale,                            \
        output_zero_point,                       \
        out);                                    \
    break;                                       \
  }

  ScalarType dtype = input.scalar_type();
  switch (dtype) {
    ET_FORALL_CADENCE_QUANTIZED_TYPES_WITH_INT16(typed_quantized_layer_norm)
    default:
      ET_DCHECK_MSG(
          false, "Unhandled dtype %s", torch::executor::toString(dtype));
  }

#undef typed_quantized_layer_norm
  return out;
}

Tensor& quantized_layer_norm_per_tensor_out(
    ET_UNUSED ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    double in_scale,
    int64_t in_zero_point,
    ET_UNUSED const IntArrayRef normalized_shape,
    const Tensor& weight,
    const Tensor& bias,
    double eps,
    double output_scale,
    int64_t output_zero_point,
    Tensor& out) {
#define typed_quantized_layer_norm(ctype, dtype) \
  case ScalarType::dtype: {                      \
    quantized_layer_norm_per_tensor_<ctype>(     \
        input,                                   \
        in_scale,                                \
        in_zero_point,                           \
        weight,                                  \
        bias,                                    \
        eps,                                     \
        output_scale,                            \
        output_zero_point,                       \
        out);                                    \
    break;                                       \
  }

  ScalarType dtype = input.scalar_type();
  switch (dtype) {
    ET_FORALL_CADENCE_QUANTIZED_TYPES_WITH_INT16(typed_quantized_layer_norm)
    default:
      ET_DCHECK_MSG(
          false, "Unhandled dtype %s", torch::executor::toString(dtype));
  }

#undef typed_quantized_layer_norm
  return out;
}

} // namespace native
} // namespace generic
} // namespace impl
