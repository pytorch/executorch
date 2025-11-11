/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/api/api.h>
#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/runtime/graph/ops/ExecuteNode.h>

namespace vkcompute {

enum class TransferType { SELECT, SLICE };

/**
 * Adds a transfer copy operation node to the compute graph, which implements
 * operators for which each element of the output tensor maps to a unique
 * element of the input tensor.
 *
 * This function currently handles the following operations:
 * - select
 * - slice
 */
void add_transfer_copy_node(
    ComputeGraph& graph,
    TransferType transfer_type,
    const ValueRef in,
    const ValueRef dim_ref,
    const ValueRef index_or_start_ref,
    const ValueRef end_ref,
    const ValueRef step_ref,
    const ValueRef out,
    const std::vector<ValueRef>& resize_args,
    const ExecuteNode::ResizeFunction& resize_fn = nullptr);

} // namespace vkcompute
