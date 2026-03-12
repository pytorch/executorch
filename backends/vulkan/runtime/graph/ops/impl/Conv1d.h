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

// Resize function shared by all buffer conv1d dispatch nodes.
void resize_conv1d_buf_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args);

// Global workgroup size function shared by all buffer conv1d dispatch nodes.
// Returns (L_out, C_out, N) from the output tensor dimensions.
utils::uvec3 conv1d_buf_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

// Prepack a 1D bias tensor into a width-packed buffer. Uses weight_data's dtype
// and staging dtype so that bias=None (where vref has no dtype) is handled.
ValueRef prepack_conv1d_bias(
    ComputeGraph& graph,
    const ValueRef vref,
    const ValueRef weight_data,
    const int64_t out_channels);

// Dispatch a depthwise 1D convolution node using width-packed buffer tensors.
// arg_weight and arg_bias must already be prepacked.
void add_conv1d_dw_buf_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef arg_weight,
    const ValueRef arg_bias,
    const ValueRef out,
    const Kernel1dParams& kernel_params,
    const float out_min_val,
    const float out_max_val,
    const bool clamp_out,
    const ValueRef weight_data,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation);

// Dispatch a depthwise 1D convolution node using width-packed TEXTURE_3D
// tensors. arg_weight (buffer) and arg_bias (buffer) must already be prepacked.
void add_conv1d_dw_texture_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef arg_weight,
    const ValueRef arg_bias,
    const ValueRef out,
    const Kernel1dParams& kernel_params,
    const float out_min_val,
    const float out_max_val,
    const bool clamp_out,
    const ValueRef weight_data,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation);

// Dispatch a pointwise (kernel_size=1) 1D convolution node using width-packed
// buffer tensors. arg_weight and arg_bias must already be prepacked.
void add_conv1d_pw_buf_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef arg_weight,
    const ValueRef arg_bias,
    const ValueRef out,
    const Kernel1dParams& kernel_params,
    const float out_min_val,
    const float out_max_val,
    const bool clamp_out,
    const ValueRef weight_data,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation);

// Top-level entry point. Determines whether the convolution is depthwise,
// pointwise, or general, prepacks weight/bias, and dispatches accordingly.
// Requires that `in` is a width-packed buffer tensor.
void add_conv1d_buf_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef groups,
    const float out_min_val,
    const float out_max_val,
    const ValueRef out,
    const bool clamp_out);

// Entry point for depthwise 1D convolution using width-packed texture3d
// input/output with buffer weight/bias. Handles prepacking internally.
void add_conv1d_dw_texture_entry(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef groups,
    const float out_min_val,
    const float out_max_val,
    const ValueRef out,
    const bool clamp_out);

// Entry point for pointwise (kernel_size=1) 1D convolution using width-packed
// texture3d input/output with buffer weight/bias. Handles prepacking
// internally.
void add_conv1d_pw_texture_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef weight_data,
    const ValueRef bias,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef groups,
    const float out_min_val,
    const float out_max_val,
    const ValueRef out,
    const bool clamp_out);

} // namespace vkcompute
