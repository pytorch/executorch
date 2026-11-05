/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>

namespace vkcompute {

// Dispatches the direct conv2d shader, or the transposed conv2d shader when
// `transposed` is true; the two share weight prepacking and dispatch logic.
void conv2d_direct_impl(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef transposed,
    const ValueRef output_padding,
    const ValueRef groups,
    const ValueRef out,
    const bool clamp_out,
    const float out_min_val,
    const float out_max_val,
    const bool force_masked = false);

// Eligibility for the tiled shaders' workgroup tuning and the im2col bypass.
bool can_use_fast_tiled_conv2d(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef out,
    const Kernel2dParams& kernel_params,
    const bool transposed,
    const int64_t groups);

bool can_use_direct_conv2d(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef out,
    const Kernel2dParams& kernel_params,
    const int64_t groups);

} // namespace vkcompute
