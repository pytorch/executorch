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

using ::executorch::aten::IntArrayRef;
using ::executorch::aten::Tensor;
using ::executorch::runtime::KernelRuntimeContext;

// NLC format (N=batch, L=length, C=channels)
::executorch::aten::Tensor& quantized_conv1d_nlc_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    int64_t input_zero_point,
    const Tensor& weight_zero_point,
    const Tensor& bias_scale,
    double output_scale,
    int64_t output_zero_point,
    const Tensor& out_multiplier,
    const Tensor& out_shift,
    Tensor& out);

::executorch::aten::Tensor& quantized_conv1d_nlc_per_tensor_out(
    KernelRuntimeContext& ctx,
    const Tensor& input,
    const Tensor& weight,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    int64_t input_zero_point,
    int64_t weight_zero_point,
    double bias_scale,
    double output_scale,
    int64_t output_zero_point,
    int64_t out_multiplier,
    int64_t out_shift,
    const ::executorch::aten::optional<Tensor>& offset,
    Tensor& out);

} // namespace native
} // namespace generic
} // namespace impl
