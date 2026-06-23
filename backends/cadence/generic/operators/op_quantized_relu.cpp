/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/operators/op_quantized_relu.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>

#include <executorch/backends/cadence/generic/kernels/kernels.h>
#include <executorch/backends/cadence/generic/operators/cadence_type_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

namespace impl {
namespace generic {
namespace native {

using ::executorch::aten::IntArrayRef;
using ::executorch::aten::optional;
using ::executorch::aten::Scalar;
using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;
using ::impl::generic::kernels::quantize;

template <typename T>
void quantized_relu_per_tensor_out_(
    ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    const int64_t in_zero_point,
    const int64_t out_zero_point,
    const int64_t out_multiplier,
    const int64_t out_shift,
    Tensor& output) {
  const T* __restrict__ in = input.const_data_ptr<T>();
  T* __restrict__ out = output.mutable_data_ptr<T>();

  // Compute the out_scale from out_multiplier and out_shift
  const float out_scale = -out_multiplier * 1.0 / (1 << 31) * pow(2, out_shift);

  for (size_t i = 0, e = input.numel(); i < e; ++i) {
    const float temp = in[i] > in_zero_point ? (in[i] - in_zero_point) : 0;
    out[i] = quantize<T>(temp, out_scale, out_zero_point);
  }
}

Tensor& quantized_relu_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const int64_t in_zero_point,
    const int64_t out_zero_point,
    const int64_t out_multiplier,
    const int64_t out_shift,
    Tensor& output) {
#define typed_quantized_relu(ctype, dtype) \
  case ScalarType::dtype: {                \
    quantized_relu_per_tensor_out_<ctype>( \
        ctx,                               \
        input,                             \
        in_zero_point,                     \
        out_zero_point,                    \
        out_multiplier,                    \
        out_shift,                         \
        output);                           \
    break;                                 \
  }

  ScalarType dtype = input.scalar_type();
  switch (dtype) {
    ET_FORALL_CADENCE_QUANTIZED_TYPES(typed_quantized_relu)
    default:
      ET_DCHECK_MSG(
          false, "Unhandled dtype %s", torch::executor::toString(dtype));
  }

#undef typed_quantized_relu
  return output;
}

template <typename T>
void quantized_relu_(
    const Tensor& input,
    const Tensor& in_zero_point,
    const int64_t out_zero_point,
    const Tensor& out_multiplier,
    const Tensor& out_shift,
    Tensor& output) {
  T q_zero_point = in_zero_point.const_data_ptr<T>()[0];
  const T* __restrict__ in = input.const_data_ptr<T>();
  T* __restrict__ out = output.mutable_data_ptr<T>();

  const int32_t* __restrict__ out_multiplier_data =
      out_multiplier.const_data_ptr<int32_t>();
  const int32_t* __restrict__ out_shift_data =
      out_shift.const_data_ptr<int32_t>();

  // Compute the out_scale from out_multiplier and out_shift
  const float out_scale =
      -out_multiplier_data[0] * 1.0 / (1 << 31) * pow(2, out_shift_data[0]);

  for (size_t i = 0, e = input.numel(); i < e; ++i) {
    const T temp = in[i] > q_zero_point ? (in[i] - q_zero_point) : 0;
    out[i] = quantize<T>(temp, out_scale, out_zero_point);
  }
}

Tensor& quantized_relu_out(
    ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& in_zero_point,
    const int64_t out_zero_point,
    const Tensor& out_multiplier,
    const Tensor& out_shift,
    Tensor& output) {
#define typed_quantized_relu(ctype, dtype) \
  case ScalarType::dtype: {                \
    quantized_relu_<ctype>(                \
        input,                             \
        in_zero_point,                     \
        out_zero_point,                    \
        out_multiplier,                    \
        out_shift,                         \
        output);                           \
    break;                                 \
  }

  ScalarType dtype = input.scalar_type();
  switch (dtype) {
    ET_FORALL_CADENCE_QUANTIZED_TYPES(typed_quantized_relu)
    default:
      ET_DCHECK_MSG(
          false, "Unhandled dtype %s", torch::executor::toString(dtype));
  }

#undef typed_quantized_relu
  return output;
}

Tensor& quantized_relu_asym8s_asym8s_per_tensor_out(
    ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    const int64_t in_zero_point,
    const int64_t out_zero_point,
    const int64_t out_multiplier,
    const int64_t out_shift,
    Tensor& output) {
  quantized_relu_per_tensor_out_<int8_t>(
      ctx,
      input,
      in_zero_point,
      out_zero_point,
      out_multiplier,
      out_shift,
      output);
  return output;
}

Tensor& quantized_relu_asym8u_asym8u_per_tensor_out(
    ET_UNUSED KernelRuntimeContext& ctx,
    const Tensor& input,
    const int64_t in_zero_point,
    const int64_t out_zero_point,
    const int64_t out_multiplier,
    const int64_t out_shift,
    Tensor& output) {
  quantized_relu_per_tensor_out_<uint8_t>(
      ctx,
      input,
      in_zero_point,
      out_zero_point,
      out_multiplier,
      out_shift,
      output);
  return output;
}

} // namespace native
} // namespace generic
} // namespace impl
