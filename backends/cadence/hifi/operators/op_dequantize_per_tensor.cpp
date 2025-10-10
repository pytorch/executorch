/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/hifi/kernels/kernels.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <xa_nnlib_kernels_api.h>

namespace impl {
namespace HiFi {
namespace native {

using ::executorch::aten::ScalarType;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;
using ::impl::HiFi::kernels::dequantize;

void dequantize_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    double scale,
    int64_t zero_point,
    __ET_UNUSED int64_t quant_min,
    __ET_UNUSED int64_t quant_max,
    ScalarType dtype,
    Tensor& out) {
  float* out_data = out.mutable_data_ptr<float>();
  const size_t numel = out.numel();
  if (input.scalar_type() == ScalarType::Byte) {
    const uint8_t* input_data = input.const_data_ptr<uint8_t>();
    dequantize<uint8_t>(out_data, input_data, scale, zero_point, numel);
  } else if (input.scalar_type() == ScalarType::Char) {
    const int8_t* input_data = input.const_data_ptr<int8_t>();
    xa_nn_elm_dequantize_asym8s_f32(
        out_data, input_data, zero_point, scale, numel);
  } else if (input.scalar_type() == ScalarType::Short) {
    const int16_t* input_data = input.const_data_ptr<int16_t>();
    dequantize<int16_t>(out_data, input_data, scale, zero_point, numel);
  } else if (
      input.scalar_type() == ScalarType::Bits16 ||
      input.scalar_type() == ScalarType::UInt16) {
    const uint16_t* input_data = input.const_data_ptr<uint16_t>();
    dequantize<uint16_t>(out_data, input_data, scale, zero_point, numel);
  } else if (input.scalar_type() == ScalarType::Int) {
    const int32_t* input_data = input.const_data_ptr<int32_t>();
    dequantize<int32_t>(out_data, input_data, scale, zero_point, numel);
  } else {
    ET_CHECK_MSG(
        false,
        "Unhandled input dtype %hhd",
        static_cast<int8_t>(input.scalar_type()));
  }
}

void dequantize_per_tensor_asym8u_out(
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
}

void dequantize_per_tensor_asym16s_out(
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
}

void dequantize_per_tensor_asym16u_out(
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
}

void dequantize_per_tensor_asym32s_out(
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
  const int32_t* input_data = input.const_data_ptr<int32_t>();
  dequantize<int32_t>(out_data, input_data, scale, zero_point, numel);
}

} // namespace native
} // namespace HiFi
} // namespace impl
