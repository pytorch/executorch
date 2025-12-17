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

/*
 * Dispatches the view_copy compute shader. This can be used to implement ops
 * that preserve the "contiguous" indexes of elements between the input and
 * output such as view_copy, squeeze_copy, unsqueeze_copy, etc.
 */
void add_view_copy_buffer_node(
    ComputeGraph& graph,
    ValueRef in,
    ValueRef out,
    const std::vector<ValueRef>& resize_args,
    const ExecuteNode::ResizeFunction& resize_fn);

/*
 * Dispatches the view_convert_buffer compute shader. This can be used to
 * implement ops that preserve the "contiguous" indexes of elements between the
 * input and output while converting between different data types such as
 * view_copy with dtype conversion.
 */
void add_view_copy_convert_buffer_node(
    ComputeGraph& graph,
    ValueRef in,
    ValueRef out,
    const std::vector<ValueRef>& resize_args,
    const ExecuteNode::ResizeFunction& resize_fn);

void add_view_node(
    ComputeGraph& graph,
    ValueRef in,
    ValueRef sizes,
    ValueRef out);

} // namespace vkcompute
