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

using Tensor = exec_aten::Tensor;
using executorch::runtime::KernelRuntimeContext;
using ScalarType = exec_aten::ScalarType;

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
    impl::HiFi::kernels::quantize<uint8_t>(
        out_data, input_data, 1. / scale, zero_point, numel);
  } else if (out.scalar_type() == ScalarType::Char) {
    int8_t* out_data = out.mutable_data_ptr<int8_t>();
    xa_nn_elm_quantize_f32_asym8s(
        out_data, input_data, scale, zero_point, numel);
  } else if (out.scalar_type() == ScalarType::Int) {
    int32_t* out_data = out.mutable_data_ptr<int32_t>();
    impl::HiFi::kernels::quantize<int32_t>(
        out_data, input_data, 1. / scale, zero_point, numel);
  } else {
    ET_CHECK_MSG(false, "Unhandled input dtype %hhd", out.scalar_type());
  }
}

}; // namespace native
}; // namespace HiFi
}; // namespace impl
