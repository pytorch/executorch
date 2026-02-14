/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Print.h>

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q8taPrint.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void add_print_node(ComputeGraph& graph, const ValueRef input) {
  std::string kernel_name;
  vkapi::ParamsBindList param_buffers;
  vkapi::SpecVarList spec_vars;

  if (graph.is_buffer_storage(input)) {
    kernel_name = "print_buffer";
    param_buffers.append(graph.buffer_meta_ubo(input));
    spec_vars = {graph.hashed_layout_of(input)};
  } else {
    kernel_name = "print_texture3d";
    param_buffers.append(graph.texture_meta_ubo(input));
  }

  add_dtype_suffix(kernel_name, graph.dtype_of(input));

  int32_t value_ref_val = static_cast<int32_t>(input);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      default_pick_global_wg_size,
      default_pick_local_wg_size,
      {{input, vkapi::kRead}},
      param_buffers,
      {PushConstantDataInfo(&value_ref_val, sizeof(value_ref_val))},
      spec_vars,
      {}));
}

void print_op(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  const ValueRef input = args.at(0);
  if (graph.dtype_of(input) == vkapi::kInt8x4) {
    add_q8ta_print_node(graph, input);
  } else {
    add_print_node(graph, input);
  }
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.print.default, print_op);
}

} // namespace vkcompute
