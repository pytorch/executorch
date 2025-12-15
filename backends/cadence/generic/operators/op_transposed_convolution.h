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

::executorch::aten::Tensor& transposed_convolution_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& input,
    const ::executorch::aten::Tensor& weight,
    const ::executorch::aten::Tensor& bias,
    ::executorch::aten::IntArrayRef stride,
    ::executorch::aten::IntArrayRef padding,
    ::executorch::aten::IntArrayRef dilation,
    ::executorch::aten::IntArrayRef output_padding,
    int64_t groups,
    bool channel_last,
    ::executorch::aten::Tensor& output);

::executorch::aten::Tensor& quantized_transposed_conv_out(
    ::executorch::runtime::KernelRuntimeContext& ctx,
    const ::executorch::aten::Tensor& input,
    const ::executorch::aten::Tensor& weight,
    const ::executorch::aten::Tensor& bias,
    ::executorch::aten::IntArrayRef stride,
    ::executorch::aten::IntArrayRef padding,
    ::executorch::aten::IntArrayRef dilation,
    ::executorch::aten::IntArrayRef output_padding,
    int64_t groups,
    int64_t in_zero_point,
    const ::executorch::aten::Tensor& weight_zero_point,
    const ::executorch::aten::Tensor& bias_scale,
    double output_scale,
    int64_t output_zero_point,
    const ::executorch::aten::Tensor& out_multiplier,
    const ::executorch::aten::Tensor& out_shift,
    bool channel_last,
    ::executorch::aten::Tensor& out);

} // namespace native
} // namespace generic
} // namespace impl
