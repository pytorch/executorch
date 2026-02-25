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

void resize_index_tensor_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef index = args.at(1).refs.at(1);

  std::vector<int64_t> out_sizes = graph->sizes_of(index);
  graph->virtual_resize(out, out_sizes);
}

void add_index_tensor_node(
    ComputeGraph& graph,
    const ValueRef self,
    const ValueRef index,
    const ValueRef out) {
  std::string kernel_name = "index_tensor";
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  vkapi::ParamsBindList param_ubos = {
      graph.meta_ubo(out), graph.meta_ubo(self), graph.meta_ubo(index)};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{self, index}, vkapi::kRead}},
      // Shader params buffers
      param_ubos,
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      resize_index_tensor_node));
}

void index_tensor(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  ValueRef self = args[0];
  ValueRef indices_list_ref = args[1];
  ValueRef out = args[2];

  ValueListPtr indices_list = graph.get_value_list(indices_list_ref);
  VK_CHECK_COND(
      indices_list->size() == 1,
      "index.Tensor: only one index tensor is supported");

  ValueRef index = indices_list->at(0);

  add_index_tensor_node(graph, self, index, out);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.index.Tensor, index_tensor);
}

} // namespace vkcompute
