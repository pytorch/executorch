/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "quantized_conv1d_impl.h"

using Tensor = executorch::aten::Tensor;
using KernelRuntimeContext = torch::executor::KernelRuntimeContext;
using ScalarType = executorch::aten::ScalarType;
using ::executorch::aten::IntArrayRef;

namespace impl {
namespace HiFi {
namespace native {

void quantized_conv1d_ncl_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    __ET_UNUSED IntArrayRef dilation,
    __ET_UNUSED int64_t groups,
    int64_t in_zero_point,
    int64_t weight_zero_point,
    double bias_scale,
    double output_scale,
    int64_t output_zero_point,
    __ET_UNUSED int64_t out_multiplier,
    __ET_UNUSED int64_t out_shift,
    Tensor& out) {
  if (input.scalar_type() == ScalarType::Byte) {
    quantized_conv1d::ncl_asym8uxsym8u_asym8u_per_tensor_impl(
        ctx,
        input,
        weight,
        bias,
        stride,
        padding,
        in_zero_point,
        weight_zero_point,
        bias_scale,
        output_scale,
        output_zero_point,
        out);
  } else if (input.scalar_type() == ScalarType::Char) {
    quantized_conv1d::ncl_asym8sxsym8s_asym8s_per_tensor_impl(
        ctx,
        input,
        weight,
        bias,
        stride,
        padding,
        in_zero_point,
        weight_zero_point,
        bias_scale,
        output_scale,
        output_zero_point,
        out);
  } else {
    ET_CHECK_MSG(false, "Unhandled input type %hhd", input.scalar_type());
  }
}

} // namespace native
} // namespace HiFi
} // namespace impl
