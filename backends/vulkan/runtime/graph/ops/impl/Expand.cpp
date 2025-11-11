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

void add_expand_buffer_node(
    ComputeGraph& graph,
    const ValueRef in,
    const ValueRef size,
    const ValueRef out) {
  std::string kernel_name = "expand";
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  vkapi::ParamsBindList param_buffers = {
      graph.buffer_meta_ubo(out),
      graph.buffer_meta_ubo(in),
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
      {},
      // Resize Args
      {size},
      // Resizing Logic
      nullptr));
}

void expand(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int idx = 0;
  const ValueRef in = args.at(idx++);
  const ValueRef size = args.at(idx++);
  const ValueRef implicit = args.at(idx++);
  (void)implicit;
  const ValueRef out = args.at(idx++);

  if (graph.is_buffer_storage(out)) {
    return add_expand_buffer_node(graph, in, size, out);
  }

  VK_THROW("Expand operator only supports buffer storage");
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.expand_copy.default, expand);
}

} // namespace vkcompute
