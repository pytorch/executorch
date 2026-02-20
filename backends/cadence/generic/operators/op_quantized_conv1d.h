/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/kernel/kernel_runtime_context.h>

namespace impl {
namespace generic {
namespace native {

executorch::aten::Tensor&
quantized_conv1d_ncl_asym8sxsym8s_asym8s_per_tensor_out(
    executorch::runtime::KernelRuntimeContext& ctx,
    const executorch::aten::Tensor& input,
    const executorch::aten::Tensor& weight,
    const executorch::aten::Tensor& bias,
    executorch::aten::IntArrayRef stride,
    executorch::aten::IntArrayRef padding,
    executorch::aten::IntArrayRef dilation,
    int64_t groups,
    int64_t in_zero_point,
    int64_t weight_zero_point,
    double bias_scale,
    double output_scale,
    int64_t output_zero_point,
    int64_t out_multiplier,
    int64_t out_shift,
    executorch::aten::Tensor& out);

executorch::aten::Tensor&
quantized_conv1d_ncl_asym8uxsym8u_asym8u_per_tensor_out(
    executorch::runtime::KernelRuntimeContext& ctx,
    const executorch::aten::Tensor& input,
    const executorch::aten::Tensor& weight,
    const executorch::aten::Tensor& bias,
    executorch::aten::IntArrayRef stride,
    executorch::aten::IntArrayRef padding,
    executorch::aten::IntArrayRef dilation,
    int64_t groups,
    int64_t in_zero_point,
    int64_t weight_zero_point,
    double bias_scale,
    double output_scale,
    int64_t output_zero_point,
    int64_t out_multiplier,
    int64_t out_shift,
    executorch::aten::Tensor& out);

executorch::aten::Tensor&
quantized_conv1d_nlc_asym8sxsym8s_asym8s_per_tensor_out(
    executorch::runtime::KernelRuntimeContext& ctx,
    const executorch::aten::Tensor& input,
    const executorch::aten::Tensor& weight,
    const executorch::aten::Tensor& bias,
    executorch::aten::IntArrayRef stride,
    executorch::aten::IntArrayRef padding,
    executorch::aten::IntArrayRef dilation,
    int64_t groups,
    int64_t in_zero_point,
    int64_t weight_zero_point,
    double bias_scale,
    double output_scale,
    int64_t output_zero_point,
    int64_t out_multiplier,
    int64_t out_shift,
    executorch::aten::Tensor& out);

executorch::aten::Tensor&
quantized_conv1d_nlc_asym8uxsym8u_asym8u_per_tensor_out(
    executorch::runtime::KernelRuntimeContext& ctx,
    const executorch::aten::Tensor& input,
    const executorch::aten::Tensor& weight,
    const executorch::aten::Tensor& bias,
    executorch::aten::IntArrayRef stride,
    executorch::aten::IntArrayRef padding,
    executorch::aten::IntArrayRef dilation,
    int64_t groups,
    int64_t in_zero_point,
    int64_t weight_zero_point,
    double bias_scale,
    double output_scale,
    int64_t output_zero_point,
    int64_t out_multiplier,
    int64_t out_shift,
    executorch::aten::Tensor& out);

} // namespace native
} // namespace generic
} // namespace impl
