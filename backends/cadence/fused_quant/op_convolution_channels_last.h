/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace cadence {
namespace fused_quant {
namespace native {

executorch::aten::Tensor& convolution_channels_last_out(
    executorch::runtime::KernelRuntimeContext& ctx,
    const executorch::aten::Tensor& inp,
    const executorch::aten::Tensor& weight,
    const executorch::aten::optional<executorch::aten::Tensor>& bias,
    // inp qparams (6)
    const executorch::aten::optional<executorch::aten::Tensor>& inp_scale,
    const executorch::aten::optional<executorch::aten::Tensor>& inp_zero_point,
    executorch::aten::ScalarType inp_dtype,
    int64_t inp_quant_min,
    int64_t inp_quant_max,
    executorch::aten::optional<int64_t> inp_axis,
    // weight qparams (6)
    const executorch::aten::optional<executorch::aten::Tensor>& weight_scale,
    const executorch::aten::optional<executorch::aten::Tensor>&
        weight_zero_point,
    executorch::aten::ScalarType weight_dtype,
    int64_t weight_quant_min,
    int64_t weight_quant_max,
    executorch::aten::optional<int64_t> weight_axis,
    // bias qparams (6)
    const executorch::aten::optional<executorch::aten::Tensor>& bias_scale,
    const executorch::aten::optional<executorch::aten::Tensor>& bias_zero_point,
    executorch::aten::ScalarType bias_dtype,
    int64_t bias_quant_min,
    int64_t bias_quant_max,
    executorch::aten::optional<int64_t> bias_axis,
    // out qparams (6)
    const executorch::aten::optional<executorch::aten::Tensor>& out_scale,
    const executorch::aten::optional<executorch::aten::Tensor>& out_zero_point,
    executorch::aten::ScalarType out_dtype,
    int64_t out_quant_min,
    int64_t out_quant_max,
    executorch::aten::optional<int64_t> out_axis,
    // conv params
    executorch::aten::IntArrayRef stride,
    executorch::aten::IntArrayRef padding,
    executorch::aten::IntArrayRef dilation,
    int64_t groups,
    executorch::aten::Tensor& out);

} // namespace native
} // namespace fused_quant
} // namespace cadence
