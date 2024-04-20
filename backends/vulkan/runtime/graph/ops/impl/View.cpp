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

void add_view_node(ComputeGraph& graph, ValueRef in, ValueRef out) {
  vTensorPtr t_in = graph.get_tensor(in);
  vTensorPtr t_out = graph.get_tensor(out);

  std::string kernel_name = "view";
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, *t_out);

  api::utils::uvec3 global_size = t_out->extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_size,
      local_size,
      // Inputs and Outputs
      {{out, api::MemoryAccessType::WRITE}, {in, api::MemoryAccessType::READ}},
      // Parameter Buffers
      {t_out->sizes_ubo(), t_in->sizes_ubo()},
      // Specialization Constants
      {SV(t_in->gpu_memory_layout_int())}));
}

void view(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  // Note: The second argument size_ref is not used here. Since the output
  // tensor's size have been determined during compilation.
  return add_view_node(graph, args[0], args[2]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.view_copy.default, view);
}

} // namespace vkcompute
