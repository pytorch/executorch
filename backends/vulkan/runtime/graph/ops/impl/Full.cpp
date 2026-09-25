/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void resize_full_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  const ValueRef out = args.at(0).refs.at(0);
  std::vector<int64_t> out_sizes;
  if (graph->val_is_tensor(extra_args.at(0))) {
    out_sizes = graph->sizes_of(extra_args.at(0));
  } else {
    out_sizes = graph->extract_int_or_symint_list(extra_args.at(0));
  }

  graph->virtual_resize(out, out_sizes);
}

void add_full_node(
    ComputeGraph& graph,
    const ValueRef size_or_in,
    const ValueRef fill_value,
    const ValueRef out) {
  vkapi::BufferBindInfo fill_value_buffer;
  if (graph.dtype_of(out) == vkapi::kInt) {
    fill_value_buffer =
        graph.create_params_buffer(graph.extract_scalar<int32_t>(fill_value));
  } else if (
      graph.dtype_of(out) == vkapi::kBool ||
      graph.dtype_of(out) == vkapi::kByte) {
    const uint32_t value = graph.dtype_of(out) == vkapi::kBool
        ? graph.extract_scalar<bool>(fill_value)
        : graph.extract_scalar<uint32_t>(fill_value);
    fill_value_buffer = graph.create_params_buffer(value);
  } else {
    fill_value_buffer =
        graph.create_params_buffer(graph.extract_scalar<float>(fill_value));
  }

  std::string kernel_name("full");
  kernel_name.reserve(kShaderNameReserve);

  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_gwg,
      default_pick_lwg,
      // Inputs and Outputs
      {{out, vkapi::kWrite}},
      // Shader params buffers
      {graph.meta_ubo(out), fill_value_buffer},
      // Push Constants
      {},
      // Specialization Constants
      {graph.hashed_layout_of(out)},
      // Resize Args
      {size_or_in},
      // Resizing Logic
      resize_full_node));
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
