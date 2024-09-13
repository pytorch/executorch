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
namespace HiFi {
namespace native {

using Tensor = exec_aten::Tensor;
using executorch::runtime::KernelRuntimeContext;
using ScalarType = exec_aten::ScalarType;

void dequantize_per_tensor_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  float* out_data = out.mutable_data_ptr<float>();
  size_t numel = out.numel();

  if (input.scalar_type() == ScalarType::Byte) {
    const uint8_t* input_data = input.const_data_ptr<uint8_t>();
    impl::HiFi::kernels::dequantize<uint8_t>(
        out_data, input_data, scale, zero_point, numel);
  } else if (input.scalar_type() == ScalarType::Char) {
    const int8_t* input_data = input.const_data_ptr<int8_t>();
    xa_nn_elm_dequantize_asym8s_f32(
        out_data, input_data, zero_point, scale, numel);
  } else if (input.scalar_type() == ScalarType::Int) {
    const int32_t* input_data = input.const_data_ptr<int32_t>();
    impl::HiFi::kernels::dequantize<int32_t>(
        out_data, input_data, scale, zero_point, numel);
  } else {
    ET_CHECK_MSG(false, "Unhandled input dtype %hhd", input.scalar_type());
  }
}

}; // namespace native
}; // namespace HiFi
}; // namespace impl
