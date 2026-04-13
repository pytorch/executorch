/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/View.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

std::vector<int64_t> compute_out_sizes(
    std::vector<int64_t> orig_sizes,
    std::vector<int64_t>& view_sizes) {
  std::vector<int64_t> out_sizes(view_sizes.begin(), view_sizes.end());
  int64_t numel = 1;
  int64_t transferred_numel = 1;

  for (int i = 0; i < orig_sizes.size(); i++) {
    numel *= orig_sizes.at(i);
  }
  for (int i = 0; i < view_sizes.size(); i++) {
    if (view_sizes.at(i) > 0) {
      transferred_numel *= view_sizes.at(i);
    }
  }
  for (int i = 0; i < out_sizes.size(); i++) {
    if (out_sizes.at(i) == -1) {
      out_sizes.at(i) = numel / transferred_numel;
    }
  }
  return out_sizes;
}

void resize_view_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);
  if (extra_args.at(0) == kDummyValueRef ||
      graph->val_is_none(extra_args.at(0))) {
    const std::vector<int64_t> in_sizes = graph->sizes_of(in);
    graph->virtual_resize(out, in_sizes);
  } else {
    std::vector<int64_t> view_sizes =
        graph->extract_int_or_symint_list(extra_args.at(0));
    const std::vector<int64_t> in_sizes = graph->sizes_of(in);
    const std::vector<int64_t> out_sizes =
        compute_out_sizes(in_sizes, view_sizes);
    graph->virtual_resize(out, out_sizes);
  }
}

void resize_to_dim_order_copy_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in = args.at(1).refs.at(0);
  const std::vector<int64_t> in_sizes = graph->sizes_of(in);
  graph->virtual_resize(out, in_sizes);
}

void add_view_copy_node(
    ComputeGraph& graph,
    ValueRef in,
    ValueRef out,
    const std::vector<ValueRef>& resize_args,
    const ExecuteNode::ResizeFunction& resize_fn) {
  std::string kernel_name =
      graph.is_buffer_storage(out) ? "view_buffer" : "view_texture";
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  vkapi::ParamsBindList ubos = {graph.meta_ubo(out), graph.meta_ubo(in)};

  // Spec constant order differs between buffer and texture shaders.
  vkapi::SpecVarList spec_constants = graph.is_buffer_storage(out)
      ? vkapi::
            SpecVarList{graph.hashed_layout_of(out), graph.hashed_layout_of(in)}
      : vkapi::SpecVarList{
            graph.hashed_layout_of(in), graph.hashed_layout_of(out)};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {in, vkapi::kRead}},
      // Parameter Buffers
      ubos,
      // Push Constants
      {},
      // Specialization Constants
      spec_constants,
      // Resize Args
      resize_args,
      // Resizing Logic
      resize_fn));
}

void add_view_copy_convert_node(
    ComputeGraph& graph,
    ValueRef in,
    ValueRef out,
    const std::vector<ValueRef>& resize_args,
    const ExecuteNode::ResizeFunction& resize_fn) {
  std::string kernel_name;
  vkapi::ParamsBindList ubos;
  std::vector<PushConstantDataInfo> push_constants;
  vkapi::SpecVarList spec_constants;

  if (graph.is_buffer_storage(out)) {
    kernel_name = "view_convert_buffer";
    ubos = {graph.buffer_meta_ubo(out), graph.buffer_meta_ubo(in)};
    spec_constants = {graph.hashed_layout_of(out), graph.hashed_layout_of(in)};
  } else {
    kernel_name = "view_convert_texture";
    push_constants = {graph.sizes_pc_of(out)};
    spec_constants = {graph.packed_dim_of(out)};
  }
  add_dtype_suffix(kernel_name, graph.dtype_of(in));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {in, vkapi::kRead}},
      // Parameter Buffers
      ubos,
      // Push Constants
      {push_constants},
      // Specialization Constants
      spec_constants,
      // Resize Args
      resize_args,
      // Resizing Logic
      resize_fn));
}

void view(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int idx = 0;
  const ValueRef in = args.at(idx++);
  const ValueRef sizes = args.at(idx++);
  const ValueRef out = args.at(idx++);

  return add_view_copy_node(graph, in, out, {sizes}, resize_view_node);
}

void to_dim_order_copy(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int args_idx = 0;
  const ValueRef in = args.at(args_idx++);
  const ValueRef dtype = args.at(args_idx++);
  (void)dtype;
  const ValueRef layout = args.at(args_idx++);
  (void)layout;
  const ValueRef device = args.at(args_idx++);
  (void)device;
  const ValueRef pin_memory = args.at(args_idx++);
  (void)pin_memory;
  const ValueRef non_blocking = args.at(args_idx++);
  (void)non_blocking;
  const ValueRef dim_order = args.at(args_idx++);
  (void)dim_order;

  const ValueRef out = args.at(args_idx++);

  VK_CHECK_COND(graph.is_buffer_storage(in) && graph.is_buffer_storage(out));

  if (graph.dtype_of(in) == graph.dtype_of(out)) {
    return add_view_copy_node(
        graph, in, out, {}, resize_to_dim_order_copy_node);
  }

  return add_view_copy_convert_node(
      graph, in, out, {}, resize_to_dim_order_copy_node);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.view_copy.default, view);
  VK_REGISTER_OP(dim_order_ops._to_dim_order_copy.default, to_dim_order_copy);
}

} // namespace vkcompute
