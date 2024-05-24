/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/ScalarUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void resize_upsample_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)graph;
  (void)args;
  (void)extra_args;
}

void add_upsample_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef out) {
  ValueRef arg = prepack_if_tensor_ref(graph, in);

  vTensorPtr t_out = graph.get_tensor(out);
  api::utils::uvec3 global_size = t_out->image_extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  std::string kernel_name("upsample");
  kernel_name.reserve(kShaderNameReserve);

  add_dtype_suffix(kernel_name, *t_out);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_size,
      local_size,
      // Inputs and Outputs
      {{out, api::MemoryAccessType::WRITE}, {arg, api::MemoryAccessType::READ}},
      // Shader params buffers
      {t_out->texture_limits_ubo(), graph.create_params_buffer(0.5)},
      // Specialization Constants
      {},
      // Resizing Logic
      resize_upsample_node));
}

void upsample(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_upsample_node(graph, args[0], args[3]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.upsample_nearest2d.vec, upsample);
}

} // namespace vkcompute
