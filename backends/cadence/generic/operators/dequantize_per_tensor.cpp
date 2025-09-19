/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/generic/kernels/kernels.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace impl {
namespace generic {
namespace native {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;
using ::impl::generic::kernels::dequantize;

Tensor& dequantize_per_tensor_out(
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
    dequantize<uint8_t>(out_data, input_data, scale, zero_point, numel);
  } else if (input.scalar_type() == ScalarType::Char) {
    const int8_t* input_data = input.const_data_ptr<int8_t>();
    dequantize<int8_t>(out_data, input_data, scale, zero_point, numel);
  } else if (
      input.scalar_type() == ScalarType::Bits16 ||
      input.scalar_type() == ScalarType::UInt16) {
    const uint16_t* input_data = input.const_data_ptr<uint16_t>();
    dequantize<uint16_t>(out_data, input_data, scale, zero_point, numel);
  } else if (input.scalar_type() == ScalarType::Short) {
    const int16_t* input_data = input.const_data_ptr<int16_t>();
    dequantize<int16_t>(out_data, input_data, scale, zero_point, numel);
  } else {
    ET_CHECK_MSG(
        false,
        "Unhandled input dtype %hhd",
        static_cast<int8_t>(input.scalar_type()));
  }
  return out;
}

Tensor& dequantize_per_tensor_asym8s_out(
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
  const int8_t* input_data = input.const_data_ptr<int8_t>();
  dequantize<int8_t>(out_data, input_data, scale, zero_point, numel);
  return out;
}

Tensor& dequantize_per_tensor_asym8u_out(
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
  const uint8_t* input_data = input.const_data_ptr<uint8_t>();
  dequantize<uint8_t>(out_data, input_data, scale, zero_point, numel);
  return out;
}

Tensor& dequantize_per_tensor_asym16s_out(
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
  const int16_t* input_data = input.const_data_ptr<int16_t>();
  dequantize<int16_t>(out_data, input_data, scale, zero_point, numel);
  return out;
}

Tensor& dequantize_per_tensor_asym16u_out(
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
  const uint16_t* input_data = input.const_data_ptr<uint16_t>();
  dequantize<uint16_t>(out_data, input_data, scale, zero_point, numel);
  return out;
}

} // namespace native
} // namespace generic
} // namespace impl
