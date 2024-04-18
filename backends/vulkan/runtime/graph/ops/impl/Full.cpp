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

void resize_full_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  vTensorPtr out = graph->get_tensor(args[0].refs[0]);
  std::vector<int64_t> out_sizes = *graph->get_int_list(extra_args[0]);

  out->virtual_resize(out_sizes);
}

void add_full_node(
    ComputeGraph& graph,
    const ValueRef size,
    const ValueRef fill_value,
    const ValueRef out) {
  float fill_value_val = graph.extract_scalar<float>(fill_value);
  vTensorPtr t_out = graph.get_tensor(out);

  api::utils::uvec3 global_size = t_out->extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  std::string kernel_name("full");
  kernel_name.reserve(kShaderNameReserve);

  add_dtype_suffix(kernel_name, *t_out);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_size,
      local_size,
      // Inputs and Outputs
      {{out, api::MemoryAccessType::WRITE}},
      // Shader params buffers
      {t_out->gpu_sizes_ubo(),
       t_out->cpu_sizes_ubo(),
       graph.create_params_buffer(fill_value_val)},
      // Resizing
      resize_full_node,
      {size}));
}

void full(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_full_node(graph, args[0], args[1], args[6]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.full.default, full);
}

} // namespace vkcompute
