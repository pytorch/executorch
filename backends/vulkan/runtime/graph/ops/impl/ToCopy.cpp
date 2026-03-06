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
#include <executorch/backends/vulkan/runtime/graph/ops/impl/View.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void resize_to_copy_op_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef self = args.at(1).refs.at(0);

  graph->virtual_resize(out, graph->sizes_of(self));
}

bool is_float_type(vkapi::ScalarType dtype) {
  return dtype == vkapi::ScalarType::Float || dtype == vkapi::ScalarType::Half;
}

void add_to_copy_node(ComputeGraph& graph, ValueRef in, ValueRef out) {
  vkapi::ScalarType in_dtype = graph.dtype_of(in);
  vkapi::ScalarType out_dtype = graph.dtype_of(out);

  // Same-dtype or float<->half conversions can use BlitNode
  if (in_dtype == out_dtype ||
      (is_float_type(in_dtype) && is_float_type(out_dtype))) {
    graph.execute_nodes().emplace_back(new BlitNode(graph, in, out));
    return;
  }

  // Other conversions (e.g. int<->float) use view_convert shaders
  if (graph.is_buffer_storage(in)) {
    add_view_copy_convert_buffer_node(
        graph, in, out, {}, resize_to_copy_op_node);
  } else {
    add_view_copy_convert_texture_node(
        graph, in, out, {}, resize_to_copy_op_node);
  }
}

void to_copy(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_to_copy_node(graph, args[0], args[7]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten._to_copy.default, to_copy);
}
} // namespace vkcompute
