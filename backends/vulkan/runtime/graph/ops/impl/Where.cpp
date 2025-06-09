// Where.cpp

/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void resize_where_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  vTensorPtr out = graph->get_tensor(args[0].refs[0]);
  vTensorPtr in = graph->get_tensor(args[1].refs[0]);

  std::vector<int64_t> in_sizes = in->sizes();
  out->virtual_resize(in_sizes);
}

void add_where_texture_node(
    ComputeGraph& graph,
    const ValueRef cond,
    const ValueRef self,
    const ValueRef other,
    const ValueRef out) {
  std::string kernel_name = "where";

  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  const utils::uvec3 global_wg_size = graph.create_global_wg_size(out);
  const utils::uvec3 local_wg_size = graph.create_local_wg_size(global_wg_size);

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      // Shader
      VK_KERNEL_FROM_STR(kernel_name),
      // Workgroup sizes
      global_wg_size,
      local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{cond, self, other}, vkapi::kRead}},
      // Parameter buffers
      {graph.logical_limits_ubo(self)},
      // Push Constants
      {},
      // Specialization Constants
      {graph.packed_dim_of(out)},
      // Resize Arguments
      {},
      // Resizing Logic
      resize_where_node));
}

void add_where_buffer_node(
    ComputeGraph& graph,
    const ValueRef cond,
    const ValueRef self,
    const ValueRef other,
    const ValueRef out) {
  std::string kernel_name = "where";

  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  const utils::uvec3 global_wg_size = graph.create_global_wg_size(out);
  const utils::uvec3 local_wg_size = graph.create_local_wg_size(global_wg_size);

  vkapi::ParamsBindList ubos = {
      graph.numel_ubo(out),
      graph.strides_ubo(out),
      graph.strides_ubo(cond),
      graph.strides_ubo(self),
      graph.strides_ubo(other)};

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      // Shader
      VK_KERNEL_FROM_STR(kernel_name),
      // Workgroup sizes
      global_wg_size,
      local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{cond, self, other}, vkapi::kRead}},
      // Parameter buffers
      ubos,
      // Push Constants
      {},
      // Specialization Constants
      {graph.packed_dim_of(out),
       graph.packed_dim_of(cond),
       graph.packed_dim_of(self),
       graph.packed_dim_of(other)},
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
  if (graph.is_buffer_storage(out)) {
    add_where_buffer_node(graph, cond, self, other, out);
  } else {
    add_where_texture_node(graph, cond, self, other, out);
  }
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.where.self, where);
}

} // namespace vkcompute
