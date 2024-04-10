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

void check_matmul_args(
    const vTensor& mat1,
    const vTensor& mat2,
    const vTensor& out) {
  VK_CHECK_COND(check_ndim_is(mat1, 2) || check_ndim_is(mat1, 3));
  VK_CHECK_COND(check_same_ndim(mat1, mat2));

  VK_CHECK_COND(
      check_memory_layout_is(mat1, api::kChannelsPacked) ||
      check_memory_layout_is(mat1, api::kWidthPacked));
  VK_CHECK_COND(check_same_memory_layout(mat1, out));

  VK_CHECK_COND(check_same_sizes_at(mat1, -1, mat2, -2));
}

void resize_matmul_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  vTensorPtr out = graph->get_tensor(args[0].refs[0]);
  vTensorPtr mat1 = graph->get_tensor(args[1].refs[0]);
  vTensorPtr mat2 = graph->get_tensor(args[1].refs[1]);

  std::vector<int64_t> new_out_sizes(3);
  if (mat1->sizes().size() == 2) {
    new_out_sizes.resize(2);
    new_out_sizes.at(0) = mat1->sizes().at(0);
    new_out_sizes.at(1) = mat2->sizes().at(1);
  } else {
    new_out_sizes.at(0) = mat1->sizes().at(0);
    new_out_sizes.at(1) = mat1->sizes().at(1);
    new_out_sizes.at(2) = mat2->sizes().at(2);
  }

  out->virtual_resize(new_out_sizes);
}

void add_matmul_node(
    ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef mat2,
    const ValueRef out) {
  ValueRef arg1 = prepack_if_tensor_ref(graph, mat1, api::kWidthPacked);

  api::GPUMemoryLayout mat2_layout =
      graph.memory_layout_of(arg1) == api::kChannelsPacked
      ? api::kChannelsPacked
      : api::kHeightPacked;

  ValueRef arg2 = prepack_if_tensor_ref(graph, mat2, mat2_layout);

  vTensorPtr t_mat1 = graph.get_tensor(arg1);
  vTensorPtr t_mat2 = graph.get_tensor(arg2);
  vTensorPtr t_out = graph.get_tensor(out);

  check_matmul_args(*t_mat1, *t_mat2, *t_out);

  api::utils::uvec3 global_size = t_out->extents();
  api::utils::uvec3 local_size = adaptive_work_group_size(global_size);

  std::string kernel_name("matmul");
  kernel_name.reserve(kShaderNameReserve);
  add_memory_layout_suffix(kernel_name, *t_mat1);
  add_memory_layout_suffix(kernel_name, *t_mat2);
  add_dtype_suffix(kernel_name, *t_out);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_size,
      local_size,
      // Inputs and Outputs
      {{out, api::MemoryAccessType::WRITE},
       {{arg1, arg2}, api::MemoryAccessType::READ}},
      // Shader params buffers
      {t_out->extents_ubo(), t_mat1->cpu_sizes_ubo()},
      // Resizing
      resize_matmul_node));
}

void matmul(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_matmul_node(graph, args[0], args[1], args[2]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.mm.default, matmul);
}

} // namespace vkcompute
