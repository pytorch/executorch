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

executorch::aten::Tensor& hardswish_out(
    executorch::runtime::KernelRuntimeContext& ctx,
    const executorch::aten::Tensor& inp,
    const executorch::aten::optional<executorch::aten::Tensor>& inp_scale,
    const executorch::aten::optional<executorch::aten::Tensor>& inp_zero_point,
    executorch::aten::ScalarType inp_dtype,
    int64_t inp_quant_min,
    int64_t inp_quant_max,
    executorch::aten::optional<int64_t> inp_axis,
    const executorch::aten::optional<executorch::aten::Tensor>& out_scale,
    const executorch::aten::optional<executorch::aten::Tensor>& out_zero_point,
    executorch::aten::ScalarType out_dtype,
    int64_t out_quant_min,
    int64_t out_quant_max,
    executorch::aten::optional<int64_t> out_axis,
    executorch::aten::Tensor& out);

} // namespace native
} // namespace fused_quant
} // namespace cadence
