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
  vTensorPtr out = graph->get_tensor(args[0].refs[0]);
  vTensorPtr in = graph->get_tensor(args[1].refs[0]);
  if (extra_args[0] == kDummyValueRef || graph->val_is_none(extra_args[0])) {
    out->virtual_resize(in->sizes());
  } else {
    IntListPtr view_sizes = graph->get_int_list(extra_args[0]);
    std::vector<int64_t> out_sizes =
        compute_out_sizes(in->sizes(), *view_sizes);
    out->virtual_resize(out_sizes);
  }
}

void add_view_node(
    ComputeGraph& graph,
    ValueRef in,
    ValueRef sizes,
    ValueRef out) {
  vTensorPtr t_in = graph.get_tensor(in);
  vTensorPtr t_out = graph.get_tensor(out);

  std::string kernel_name = "view";
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, *t_out);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      graph.create_global_wg_size(out),
      graph.create_local_wg_size(out),
      // Inputs and Outputs
      {{out, vkapi::MemoryAccessType::WRITE},
       {in, vkapi::MemoryAccessType::READ}},
      // Parameter Buffers
      {t_out->sizes_ubo(), t_in->sizes_ubo()},
      // Specialization Constants
      {SV(t_in->packed_dim_whcn_idx()), SV(t_out->packed_dim_whcn_idx())},
      // Resizing Logic
      resize_view_node,
      {sizes}));
}

void view(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  return add_view_node(graph, args[0], args[1], args[2]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.view_copy.default, view);
}

} // namespace vkcompute
