/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

// Shader selection function for add operations
vkapi::ShaderInfo pick_add_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef in1 = args.at(1).refs.at(0);

  // Build shader name following the binary_op pattern
  std::string kernel_name = "add";
  add_storage_type_suffix(kernel_name, graph->storage_type_of(out));
  add_dtype_suffix(kernel_name, graph->dtype_of(in1));

  return VK_KERNEL_FROM_STR(kernel_name);
}

// Global workgroup size function for add operations
utils::uvec3 add_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  return default_pick_global_wg_size(graph, shader, args, resize_args);
}

// Local workgroup size function for add operations
utils::uvec3 add_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  return default_pick_local_wg_size(
      graph, shader, global_workgroup_size, args, resize_args);
}

void add_prototype(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int idx = 0;
  const ValueRef input_a = args.at(idx++);
  const ValueRef input_b = args.at(idx++);
  const ValueRef output = args.at(idx++);

  // Prepare parameter buffers (empty for add operation)
  vkapi::ParamsBindList param_buffers;

  // Prepare push constants based on storage type
  std::vector<PushConstantDataInfo> push_constants;
  push_constants.reserve(graph.is_buffer_storage(output) ? 1 : 1);

  if (graph.is_buffer_storage(output)) {
    // Buffer storage: pass numel as push constant
    push_constants.emplace_back(graph.numel_pc_of(output));
  } else {
    // Texture storage: pass sizes as push constant
    push_constants.emplace_back(graph.sizes_pc_of(output));
  }

  // Prepare specialization constants
  vkapi::SpecVarList spec_vars;
  if (graph.is_buffer_storage(output)) {
    spec_vars = {
        graph.hashed_layout_of(output),
        graph.hashed_layout_of(input_a),
        graph.hashed_layout_of(input_b)};
  } else {
    spec_vars = {graph.hashed_layout_of(output)};
  }

  // Add the compute node
  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_add_shader,
      add_global_wg_size,
      add_local_wg_size,
      // Inputs and Outputs
      {{output, vkapi::kWrite}, {{input_a, input_b}, vkapi::kRead}},
      // Shader params buffers
      param_buffers,
      // Push Constants
      push_constants,
      // Specialization Constants
      spec_vars,
      // Resize args
      {},
      // Resizing Logic
      nullptr));
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(etvk.add_prototype, add_prototype);
}

} // namespace vkcompute
