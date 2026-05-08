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
 * output such as view_copy, squeeze_copy, unsqueeze_copy, etc. Handles both
 * buffer and texture storage internally.
 */
void add_view_copy_node(
    ComputeGraph& graph,
    ValueRef in,
    ValueRef out,
    const std::vector<ValueRef>& resize_args,
    const ExecuteNode::ResizeFunction& resize_fn);

/*
 * Dispatches the view_convert compute shader. This can be used to implement ops
 * that preserve the "contiguous" indexes of elements between the input and
 * output while converting between different data types. Handles both buffer and
 * texture storage internally.
 */
void add_view_copy_convert_node(
    ComputeGraph& graph,
    ValueRef in,
    ValueRef out,
    const std::vector<ValueRef>& resize_args,
    const ExecuteNode::ResizeFunction& resize_fn);

} // namespace vkcompute
