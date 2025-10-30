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

namespace impl {
namespace HiFi {
namespace native {

void quantized_relu_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const int64_t in_zero_point,
    const int64_t out_zero_point,
    const int64_t out_multiplier,
    const int64_t out_shift,
    Tensor& output) {
  const int32_t _out_multiplier = static_cast<int32_t>(out_multiplier);
  const int32_t _out_shift = static_cast<int32_t>(out_shift);

  if (input.scalar_type() == executorch::aten::ScalarType::Byte) {
    const uint8_t _in_zero_point = static_cast<uint8_t>(in_zero_point);
    const uint8_t _out_zero_point = static_cast<uint8_t>(out_zero_point);
    const uint8_t* p_in = input.const_data_ptr<uint8_t>();
    uint8_t* p_out = output.mutable_data_ptr<uint8_t>();

    WORD32 ret_val = xa_nn_vec_relu_asym8u_asym8u(
        p_out,
        p_in,
        _in_zero_point,
        _out_multiplier,
        _out_shift,
        _out_zero_point,
        0,
        255,
        input.numel());

    ET_CHECK_MSG(ret_val == 0, "An internal error occured");

  } else if (input.scalar_type() == executorch::aten::ScalarType::Char) {
    const int8_t _in_zero_point = static_cast<int8_t>(in_zero_point);
    const int8_t _out_zero_point = static_cast<int8_t>(out_zero_point);
    const int8_t* p_in = input.const_data_ptr<int8_t>();
    int8_t* p_out = output.mutable_data_ptr<int8_t>();

    WORD32 ret_val = xa_nn_vec_relu_asym8s_asym8s(
        p_out,
        p_in,
        _in_zero_point,
        _out_multiplier,
        _out_shift,
        _out_zero_point,
        -128,
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

void quantized_relu_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& in_zero_point,
    const int64_t out_zero_point,
    const Tensor& out_multiplier,
    const Tensor& out_shift,
    Tensor& output) {
  int8_t _in_zero_point = in_zero_point.const_data_ptr<int8_t>()[0];
  int32_t _out_multiplier = out_multiplier.const_data_ptr<int32_t>()[0];
  int32_t _out_shift = out_shift.const_data_ptr<int32_t>()[0];

  quantized_relu_per_tensor_out(
      ctx,
      input,
      _in_zero_point,
      out_zero_point,
      _out_multiplier,
      _out_shift,
      output);
}

void quantized_relu_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& in_zero_point,
    const int64_t out_zero_point,
    const Tensor& out_multiplier,
    const Tensor& out_shift,
    Tensor& output) {
  quantized_relu_per_tensor_out(
      ctx,
      input,
      in_zero_point,
      out_zero_point,
      out_multiplier,
      out_shift,
      output);
}

} // namespace native
} // namespace HiFi
} // namespace impl
