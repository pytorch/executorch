/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

using namespace utils;

void resize_tan_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef self = args.at(1).refs.at(0);

  const std::vector<int64_t> self_sizes = graph->sizes_of(self);
  graph->virtual_resize(out, self_sizes);
}

void add_tan_node(ComputeGraph& graph, const ValueRef in, const ValueRef out) {
  std::string kernel_name = "tan";
  add_dtype_suffix(kernel_name, graph.dtype_of(out));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));

  vkapi::ParamsBindList ubos({});
  ubos.append({graph.logical_limits_ubo(out)});

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {in, vkapi::kRead}},
      // Shader params buffers
      ubos,
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      resize_tan_node));
}

void tan(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_tan_node(graph, args[0], args[1]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.tan.default, tan);
}

} // namespace vkcompute
