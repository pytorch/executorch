/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using torch::executor::KernelRuntimeContext;

namespace cadence {
namespace impl {
namespace HiFi {
namespace native {

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
    float temp = in[i] > q_zero_point ? (in[i] - q_zero_point) : 0;
    out[i] = kernels::quantize<T>(temp, out_scale, (int32_t)out_zero_point);
  }
}

void quantized_relu_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& in_zero_point,
    const int64_t out_zero_point,
    const Tensor& out_multiplier,
    const Tensor& out_shift,
    Tensor& output) {
  if (input.scalar_type() == executorch::aten::ScalarType::Byte) {
    const uint8_t* p_in = input.const_data_ptr<uint8_t>();
    uint8_t* p_out = output.mutable_data_ptr<uint8_t>();
    uint8_t q_zero_point = in_zero_point.const_data_ptr<uint8_t>()[0];

    WORD32 ret_val = xa_nn_vec_relu_asym8u_asym8u(
        p_out,
        p_in,
        (int)q_zero_point,
        out_multiplier.const_data_ptr<int32_t>()[0],
        out_shift.const_data_ptr<int32_t>()[0],
        (int)out_zero_point,
        (int)out_zero_point,
        255,
        input.numel());

    ET_CHECK_MSG(ret_val == 0, "An internal error occured");

  } else if (input.scalar_type() == executorch::aten::ScalarType::Char) {
    const int8_t* p_in = input.const_data_ptr<int8_t>();
    int8_t* p_out = output.mutable_data_ptr<int8_t>();
    int8_t q_zero_point = in_zero_point.const_data_ptr<int8_t>()[0];

    WORD32 ret_val = xa_nn_vec_relu_asym8s_asym8s(
        p_out,
        p_in,
        (int)q_zero_point,
        out_multiplier.const_data_ptr<int32_t>()[0],
        out_shift.const_data_ptr<int32_t>()[0],
        (int)out_zero_point,
        (int)out_zero_point,
        127,
        input.numel());

    ET_CHECK_MSG(ret_val == 0, "An internal error occured");

  } else {
    ET_CHECK_MSG(
        false,
        "Unhandled input dtype %hhd",
        static_cast<int8_t>(input.scalar_type()));
  }
}

} // namespace native
} // namespace HiFi
} // namespace impl
} // namespace cadence
