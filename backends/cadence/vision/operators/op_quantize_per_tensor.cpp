/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/cadence/vision/kernels/kernels.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace impl {
namespace vision {
namespace native {

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::KernelRuntimeContext;

// Quantize the input tensor (PT2 version). Note that quant_<min,max> are not
// used in any computation.
void quantize_per_tensor_out(
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
    kernels::quantize<uint8_t>(
        out_data, input_data, 1. / scale, zero_point, numel);
  } else if (out.scalar_type() == ScalarType::Char) {
    int8_t* out_data = out.mutable_data_ptr<int8_t>();
    kernels::quantize<int8_t>(
        out_data, input_data, 1. / scale, zero_point, numel);
  } else if (
      out.scalar_type() == ScalarType::Bits16 ||
      out.scalar_type() == ScalarType::UInt16) {
    uint16_t* out_data = out.mutable_data_ptr<uint16_t>();
    kernels::quantize<uint16_t>(
        out_data, input_data, 1. / scale, zero_point, numel);
  } else if (out.scalar_type() == ScalarType::Short) {
    int16_t* out_data = out.mutable_data_ptr<int16_t>();
    kernels::quantize<int16_t>(
        out_data, input_data, 1. / scale, zero_point, numel);
  } else if (out.scalar_type() == ScalarType::Int) {
    int32_t* out_data = out.mutable_data_ptr<int32_t>();
    kernels::quantize<int32_t>(
        out_data, input_data, 1. / scale, zero_point, numel);
  } else {
    ET_CHECK_MSG(
        false,
        "Unhandled input dtype %hhd",
        static_cast<int8_t>(out.scalar_type()));
  }
}

}; // namespace native
}; // namespace vision
}; // namespace impl
