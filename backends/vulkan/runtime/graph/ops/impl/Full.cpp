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
  std::vector<int64_t> out_sizes;
  if (graph->val_is_tensor(extra_args[0])) {
    out_sizes = graph->get_tensor(extra_args[0])->sizes();
  } else {
    out_sizes = *graph->get_int_list(extra_args[0]);
  }

  out->virtual_resize(out_sizes);
}

// size_or_in is IntListPtr when op is full and vTensorPtr if op is full_like
void add_full_node(
    ComputeGraph& graph,
    const ValueRef size_or_in,
    const ValueRef fill_value,
    const ValueRef out) {
  float fill_value_val = graph.extract_scalar<float>(fill_value);
  vTensorPtr t_out = graph.get_tensor(out);

  std::string kernel_name("full");
  kernel_name.reserve(kShaderNameReserve);

  add_dtype_suffix(kernel_name, *t_out);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      graph.create_global_wg_size(out),
      graph.create_local_wg_size(out),
      // Inputs and Outputs
      {{out, vkapi::MemoryAccessType::WRITE}},
      // Shader params buffers
      {t_out->sizes_ubo(), graph.create_params_buffer(fill_value_val)},
      // Specialization Constants
      {SV(t_out->packed_dim_whcn_idx())},
      // Resizing Logic
      resize_full_node,
      {size_or_in}));
}

void full(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_full_node(graph, args[0], args[1], args[args.size() - 1]);
}

void zeros(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_full_node(
      graph, args[0], graph.add_scalar<int64_t>(0), args[args.size() - 1]);
}

void ones(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_full_node(
      graph, args[0], graph.add_scalar<int64_t>(1), args[args.size() - 1]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.full.default, full);
  VK_REGISTER_OP(aten.full_like.default, full);
  VK_REGISTER_OP(aten.zeros.default, zeros);
  VK_REGISTER_OP(aten.zeros_like.default, zeros);
  VK_REGISTER_OP(aten.ones.default, ones);
  VK_REGISTER_OP(aten.ones_like.default, ones);
}

} // namespace vkcompute
