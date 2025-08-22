/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/runtime/graph/ops/ExecuteNode.h>

namespace vkcompute {
ExecuteNode::ExecuteNode(
    const ResizeFunction& resize_fn,
    const std::vector<ValueRef>& resize_args,
    const std::vector<ArgGroup>& args,
    const std::string& name)
    : resize_fn_(resize_fn),
      resize_args_(resize_args),
      args_(args),
      name_(name) {}

bool ExecuteNode::trigger_resize(ComputeGraph* graph) {
  const bool any_arg_updated = was_any_arg_updated(graph);
  if (resize_fn_ && any_arg_updated) {
    resize_fn_(graph, args_, resize_args_);
  }
  return any_arg_updated;
}

bool ExecuteNode::was_any_arg_updated(const ComputeGraph* const graph) const {
  // Check all ValueRefs in ArgGroups
  for (const auto& arg_group : args_) {
    for (const auto& value_ref : arg_group.refs) {
      if (graph->was_value_updated(value_ref)) {
        return true;
      }
    }
  }

  // Check all ValueRefs in resize_args
  for (const auto& value_ref : resize_args_) {
    if (graph->was_value_updated(value_ref)) {
      return true;
    }
  }

  return false;
}

} // namespace vkcompute
