/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/runtime/graph/ops/ExecuteNode.h>

namespace vkcompute {

/**
 * Creates a global workgroup size based on the first output tensor in the args.
 * This is a utility function that extracts the output tensor from
 * args.at(0).refs.at(0) and calls graph->create_global_wg_size(out) on it.
 */
utils::uvec3 default_pick_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

/**
 * Creates a local workgroup size based on the first output tensor in the args.
 * This is a utility function that extracts the output tensor from
 * args.at(0).refs.at(0) and calls graph->create_local_wg_size(out) on it.
 */
utils::uvec3 default_pick_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

/**
 * Constructs a local work group size with the shape {W, H, 1}. The function
 * will try to set W == H == sqrt(num_invocations), where num_invocations is
 * typically 64. This configuration is good for ops like matrix multiplication
 * as it reduces the total volume of unique data that the entire work group
 * will need to read from input tensors in order to produce the output data.
 * To compute an output tile of {W, H, 1}, the work group will need to read
 * H unique rows = H * K unique elements from the input tensor and W unique cols
 * = W * K elements from the weight tensor, resulting in (W + H) * K unique
 * elements in total.
 */
utils::uvec3 pick_hw_square_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

} // namespace vkcompute
