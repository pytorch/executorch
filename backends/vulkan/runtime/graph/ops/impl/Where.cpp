// Where.cpp

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void resize_where_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef self = args.at(1).refs.at(1);

  const std::vector<int64_t> self_sizes = graph->sizes_of(self);
  graph->virtual_resize(out, self_sizes);
}

void add_where_node(
    ComputeGraph& graph,
    const ValueRef cond,
    const ValueRef self,
    const ValueRef other,
    const ValueRef out) {
  std::string kernel_name = "where";

  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  vkapi::ParamsBindList ubos = {
      graph.meta_ubo(out),
      graph.meta_ubo(cond),
      graph.meta_ubo(self),
      graph.meta_ubo(other)};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{cond, self, other}, vkapi::kRead}},
      // Parameter buffers
      ubos,
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Arguments
      {},
      // Resizing Logic
      resize_where_node));
}

void where(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int args_i = 0;
  const ValueRef cond = args[args_i++];
  const ValueRef self = args[args_i++];
  const ValueRef other = args[args_i++];
  const ValueRef out = args[args_i++];
  add_where_node(graph, cond, self, other, out);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.where.self, where);
}

} // namespace vkcompute
