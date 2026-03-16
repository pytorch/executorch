/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/DimUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void resize_expand_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  const ValueRef in = args.at(1).refs.at(0);
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef size_ref = extra_args.at(0);

  const std::vector<int64_t> in_sizes = graph->sizes_of(in);
  const std::vector<int64_t> target_sizes =
      graph->extract_int_or_symint_list(size_ref);

  const size_t dim_offset = target_sizes.size() - in_sizes.size();
  std::vector<int64_t> out_sizes(target_sizes.size());
  for (size_t i = 0; i < target_sizes.size(); i++) {
    if (target_sizes[i] == -1 && i >= dim_offset) {
      out_sizes[i] = in_sizes[i - dim_offset];
    } else if (target_sizes[i] == -1) {
      out_sizes[i] = 1;
    } else {
      out_sizes[i] = target_sizes[i];
    }
  }
  graph->virtual_resize(out, out_sizes);
}

void add_expand_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef size,
    const ValueRef out) {
  std::string kernel_name = "expand";
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  vkapi::ParamsBindList param_buffers = {
      graph.meta_ubo(out),
      graph.meta_ubo(in),
  };

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      {{out, vkapi::kWrite}, {in, vkapi::kRead}},
      // Parameter buffers
      param_buffers,
      // Push Constants
      {},
      // Specialization Constants
      {graph.hashed_layout_of(out)},
      // Resize Args
      {size},
      // Resizing Logic
      resize_expand_node));
}

void expand(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int idx = 0;
  const ValueRef in = args.at(idx++);
  const ValueRef size = args.at(idx++);
  const ValueRef implicit = args.at(idx++);
  (void)implicit;
  const ValueRef out = args.at(idx++);

  add_expand_node(graph, in, size, out);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.expand_copy.default, expand);
}

} // namespace vkcompute
