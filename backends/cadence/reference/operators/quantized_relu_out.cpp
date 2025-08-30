/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/reference/kernels/kernels.h>
#include <executorch/backends/cadence/reference/operators/operators.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace impl {
namespace reference {
namespace native {

using executorch::aten::Tensor;
using executorch::runtime::KernelRuntimeContext;

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
    out[i] = kernels::quantize<T>(temp, out_scale, out_zero_point);
  }
}

void quantized_relu_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& in_zero_point,
    const int64_t out_zero_point,
    const Tensor& out_multiplier,
    const Tensor& out_shift,
    Tensor& output) {
  if (input.scalar_type() == executorch::aten::ScalarType::Byte) {
    quantized_relu_<uint8_t>(
        input,
        in_zero_point,
        out_zero_point,
        out_multiplier,
        out_shift,
        output);
  } else if (input.scalar_type() == executorch::aten::ScalarType::Char) {
    quantized_relu_<int8_t>(
        input,
        in_zero_point,
        out_zero_point,
        out_multiplier,
        out_shift,
        output);
  } else {
    ET_CHECK_MSG(
        false,
        "Unhandled input dtype %hhd",
        static_cast<int8_t>(input.scalar_type()));
  }
}

template <typename T>
void quantized_relu_per_tensor_out_(
    __ET_UNUSED KernelRuntimeContext& ctx,
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
    out[i] = kernels::quantize<T>(temp, out_scale, out_zero_point);
  }
}

void quantized_relu_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const int64_t in_zero_point,
    const int64_t out_zero_point,
    const int64_t out_multiplier,
    const int64_t out_shift,
    Tensor& output) {
#define typed_quantized_relu(ctype, dtype)    \
  case executorch::aten::ScalarType::dtype: { \
    quantized_relu_per_tensor_out_<ctype>(    \
        ctx,                                  \
        input,                                \
        in_zero_point,                        \
        out_zero_point,                       \
        out_multiplier,                       \
        out_shift,                            \
        output);                              \
    break;                                    \
  }

  executorch::aten::ScalarType dtype = input.scalar_type();
  switch (dtype) {
    ET_FORALL_CADENCE_QUANTIZED_TYPES(typed_quantized_relu)
    default:
      ET_DCHECK_MSG(
          false, "Unhandled dtype %s", torch::executor::toString(dtype));
  }

#undef typed_quantized_relu
}

void quantized_relu_asym8s_asym8s_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const int64_t in_zero_point,
    const int64_t out_zero_point,
    const int64_t out_multiplier,
    const int64_t out_shift,
    Tensor& output) {
#define typed_quantized_relu(ctype, dtype)    \
  case executorch::aten::ScalarType::dtype: { \
    quantized_relu_per_tensor_out_<ctype>(    \
        ctx,                                  \
        input,                                \
        in_zero_point,                        \
        out_zero_point,                       \
        out_multiplier,                       \
        out_shift,                            \
        output);                              \
    break;                                    \
  }

  executorch::aten::ScalarType dtype = input.scalar_type();
  switch (dtype) {
    ET_FORALL_CADENCE_QUANTIZED_TYPES(typed_quantized_relu)
    default:
      ET_DCHECK_MSG(
          false, "Unhandled dtype %s", torch::executor::toString(dtype));
  }

#undef typed_quantized_relu
}

void quantized_relu_asym8u_asym8u_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const int64_t in_zero_point,
    const int64_t out_zero_point,
    const int64_t out_multiplier,
    const int64_t out_shift,
    Tensor& output) {
#define typed_quantized_relu(ctype, dtype)    \
  case executorch::aten::ScalarType::dtype: { \
    quantized_relu_per_tensor_out_<ctype>(    \
        ctx,                                  \
        input,                                \
        in_zero_point,                        \
        out_zero_point,                       \
        out_multiplier,                       \
        out_shift,                            \
        output);                              \
    break;                                    \
  }

  executorch::aten::ScalarType dtype = input.scalar_type();
  switch (dtype) {
    ET_FORALL_CADENCE_QUANTIZED_TYPES(typed_quantized_relu)
    default:
      ET_DCHECK_MSG(
          false, "Unhandled dtype %s", torch::executor::toString(dtype));
  }

#undef typed_quantized_relu
}

}; // namespace native
}; // namespace reference
}; // namespace impl
