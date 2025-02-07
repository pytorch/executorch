/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/BlitNode.h>
#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>
#include <set>

namespace vkcompute {

void resize_to_copy_op_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  vTensorPtr out = graph->get_tensor(args[0].refs[0]);
  vTensorPtr self = graph->get_tensor(args[1].refs[0]);

  out->virtual_resize(self->sizes());
}

void add_to_copy_node(ComputeGraph& graph, ValueRef in, ValueRef out) {
  static std::set<vkapi::ScalarType> supported_types = {
      vkapi::ScalarType::Float, vkapi::ScalarType::Half};

  VK_CHECK_COND(
      supported_types.find(graph.dtype_of(in)) != supported_types.end() &&
          supported_types.find(graph.dtype_of(out)) != supported_types.end(),
      "Unsupported dtype for to_copy, only Float and Half are currently supported, recieved ",
      vkapi::to_string(graph.dtype_of(in)),
      " <-> ",
      vkapi::to_string(graph.dtype_of(out)));

  graph.execute_nodes().emplace_back(new BlitNode(graph, in, out));
}

void to_copy(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_to_copy_node(graph, args[0], args[7]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten._to_copy.default, to_copy);
}
} // namespace vkcompute
