/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void resize_meshgrid(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  vTensorPtr out_1 = graph->get_tensor(args[0].refs[0]);
  vTensorPtr out_2 = graph->get_tensor(args[1].refs[0]);
  vTensorPtr in_1 = graph->get_tensor(args[2].refs[0]);
  vTensorPtr in_2 = graph->get_tensor(args[3].refs[0]);

  std::vector<int64_t> out_sizes = out_1->sizes();
  out_sizes.at(2) = in_1->sizes().at(0);
  out_sizes.at(3) = in_2->sizes().at(0);

  out_1->virtual_resize(out_sizes);
  out_2->virtual_resize(out_sizes);
}

void add_meshgrid_node(
    ComputeGraph& graph,
    const ValueRef& in,
    const ValueRef& out) {
  ValueListPtr input_list = graph.get_value_list(in);
  ValueListPtr output_list = graph.get_value_list(out);
  VK_CHECK_COND(input_list->size() == 2, "meshgrid only support 2 inputs");
  VK_CHECK_COND(output_list->size() == 2, "meshgrid only support 2 outputs");

  std::string kernel_name = "";
  kernel_name = "meshgrid";
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, *graph.get_tensor(output_list->at(0)));

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      graph.create_global_wg_size(output_list->at(0)),
      graph.create_local_wg_size(output_list->at(0)),
      // Inputs and Outputs
      {{{output_list->at(0), output_list->at(1)},
        vkapi::MemoryAccessType::WRITE},
       {{input_list->at(0), input_list->at(1)}, vkapi::MemoryAccessType::READ}},
      // Shader params buffers
      {
          graph.get_tensor(output_list->at(0))->sizes_ubo(),
          graph.get_tensor(input_list->at(0))->sizes_ubo(),
          graph.get_tensor(input_list->at(1))->sizes_ubo(),
      },
      // Specialization Constants
      {},
      nullptr,
      {}));
}

void meshgrid(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_meshgrid_node(graph, args[0], args[1]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.meshgrid.default, meshgrid);
}

} // namespace vkcompute
