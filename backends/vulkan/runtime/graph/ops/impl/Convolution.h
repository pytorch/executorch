/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

namespace vkcompute {

ValueRef prepack_biases(
    ComputeGraph& graph,
    const ValueRef vref,
    const ValueRef weight,
    const bool transposed,
    const utils::StorageType storage_type,
    const utils::GPUMemoryLayout memory_layout);

void check_conv_args(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef out);

void conv2d_pw_impl(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef out,
    const bool transposed_val,
    const bool clamp_out,
    const float out_min_val,
    const float out_max_val);

void conv2d_dw_impl(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef out,
    const bool clamp_out,
    const float out_min_val,
    const float out_max_val);

void resize_conv2d_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args);

// `force_direct` overrides the im2col-vs-direct routing heuristic: when true,
// a SlidingWindow conv2d always takes the direct sliding-window path,
// bypassing should_use_conv2d_im2col(). The default (false) preserves the
// production routing exactly. Pointwise / Depthwise / Transposed methods are
// unaffected by this flag.
void add_conv2d_node(
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
    const ValueRef out_min,
    const ValueRef out_max,
    const ValueRef out,
    const bool clamp_out,
    const bool force_direct = false);

} // namespace vkcompute
