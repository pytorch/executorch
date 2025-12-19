/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <lib.h>
#include <executorch/backends/cadence/generic/kernels/kernels.h>
#include <executorch/runtime/kernel/kernel_includes.h>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::KernelRuntimeContext;
using ::impl::generic::kernels::quantize;

namespace impl {
namespace vision {
namespace native {

// Quantize the input tensor (PT2 version). Note that quant_<min,max> are not
// used in any computation.
Tensor& quantize_per_tensor_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  const float* input_data = input.const_data_ptr<float>();
  size_t numel = out.numel();
 
  if (out.scalar_type() == ScalarType::Byte) {
    uint8_t* out_data = out.mutable_data_ptr<uint8_t>();
    quantize<uint8_t>(out_data, input_data, 1. / scale, zero_point, numel);
  } else if (out.scalar_type() == ScalarType::Char) {
    int8_t* out_data = out.mutable_data_ptr<int8_t>();
    quantize<int8_t>(out_data, input_data, 1. / scale, zero_point, numel);
  } else if (
      out.scalar_type() == ScalarType::Bits16 ||
      out.scalar_type() == ScalarType::UInt16) {
    uint16_t* out_data = out.mutable_data_ptr<uint16_t>();
    quantize<uint16_t>(out_data, input_data, 1. / scale, zero_point, numel);
  } else if (out.scalar_type() == ScalarType::Short) {
    int16_t* out_data = out.mutable_data_ptr<int16_t>();
    quantize<int16_t>(out_data, input_data, 1. / scale, zero_point, numel);
  } else if (out.scalar_type() == ScalarType::Int) {
    int32_t* out_data = out.mutable_data_ptr<int32_t>();
    quantize<int32_t>(out_data, input_data, 1. / scale, zero_point, numel);
  } else {
    ET_CHECK_MSG(
        false,
        "Unhandled input dtype %hhd",
        static_cast<int8_t>(out.scalar_type()));
  }
  return out;
}

Tensor& quantize_per_tensor_asym8s_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  const float* input_data = input.const_data_ptr<float>();
  size_t numel = out.numel();
  int8_t* out_data = out.mutable_data_ptr<int8_t>();
  quantize<int8_t>(out_data, input_data, 1. / scale, zero_point, numel);
  return out;
}

Tensor& quantize_per_tensor_asym8u_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  const float* input_data = input.const_data_ptr<float>();
  size_t numel = out.numel();
  uint8_t* out_data = out.mutable_data_ptr<uint8_t>();
  quantize<uint8_t>(out_data, input_data, 1. / scale, zero_point, numel);
  return out;
}

Tensor& quantize_per_tensor_asym16s_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  const float* input_data = input.const_data_ptr<float>();
  size_t numel = out.numel();
  int16_t* out_data = out.mutable_data_ptr<int16_t>();
  quantize<int16_t>(out_data, input_data, 1. / scale, zero_point, numel);
  return out;
}

Tensor& quantize_per_tensor_asym16u_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  const float* input_data = input.const_data_ptr<float>();
  size_t numel = out.numel();
  uint16_t* out_data = out.mutable_data_ptr<uint16_t>();
  quantize<uint16_t>(out_data, input_data, 1. / scale, zero_point, numel);
  return out;
}

Tensor& quantize_per_tensor_asym32s_out(
    KernelRuntimeContext& context,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    int64_t quant_min,
    int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  const float* input_data = input.const_data_ptr<float>();
  size_t numel = out.numel();
  int32_t* out_data = out.mutable_data_ptr<int32_t>();
  quantize<int32_t>(out_data, input_data, 1. / scale, zero_point, numel);
  return out;
}

}; // namespace native
}; // namespace vision
}; // namespace impl
