/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/reference/kernels/kernels.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace impl {
namespace reference {
namespace native {

using Tensor = exec_aten::Tensor;
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
  if (input.scalar_type() == exec_aten::ScalarType::Byte) {
    quantized_relu_<uint8_t>(
        input,
        in_zero_point,
        out_zero_point,
        out_multiplier,
        out_shift,
        output);
  } else if (input.scalar_type() == exec_aten::ScalarType::Char) {
    quantized_relu_<int8_t>(
        input,
        in_zero_point,
        out_zero_point,
        out_multiplier,
        out_shift,
        output);
  } else {
    ET_CHECK_MSG(false, "Unhandled input dtype %hhd", input.scalar_type());
  }
}

}; // namespace native
}; // namespace reference
}; // namespace impl
