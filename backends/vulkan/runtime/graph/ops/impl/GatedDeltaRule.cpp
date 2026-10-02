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

void resize_gated_delta_rule_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  // Extract inputs
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef final_state = args.at(0).refs.at(1);
  const ValueRef q = args.at(1).refs.at(0);
  const ValueRef initial_state = args.at(1).refs.at(5);

  // The output tensor shapes depend directly on the inputs
  std::vector<int64_t> out_sizes = graph->sizes_of(q);
  std::vector<int64_t> final_state_sizes = graph->sizes_of(initial_state);

  graph->virtual_resize(out, out_sizes);
  graph->virtual_resize(final_state, final_state_sizes);
}

GlobalWorkGrid pick_gated_delta_rule_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  std::vector<int64_t> out_shape = graph->sizes_of(out);
  uint32_t batch = out_shape[0];
  uint32_t heads = out_shape[2];
  uint32_t v_dim = out_shape[3];
  return GlobalWorkGrid(
      {vkcompute::utils::div_up(v_dim, 4u), 1u, batch * heads},
      vkcompute::kExplicitWorkGrid);
}

void add_gated_delta_rule_node(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  // The Python signature has 6 inputs:
  const ValueRef q = args[0];
  const ValueRef k = args[1];
  const ValueRef v = args[2];
  const ValueRef decay = args[3];
  const ValueRef beta = args[4];
  const ValueRef initial_state = args[5];

  // The Partitioner appends 2 outputs:
  const ValueRef out = args[6];
  const ValueRef final_state = args[7];

  std::string kernel_name = "gated_delta_rule";
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(initial_state));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      pick_gated_delta_rule_gwg,
      default_pick_lwg,
      // Inputs and Outputs (Format: Write, Read)
      {{{out, final_state}, vkapi::kWrite},
       {{q, k, v, decay, beta, initial_state}, vkapi::kRead}},
      // Shader param buffers (sizes)
      {graph.sizes_ubo(out), graph.sizes_ubo(k)},
      // Push Constants
      {graph.logical_limits_pc_of(out)},
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      resize_gated_delta_rule_node));
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(llama.gated_delta_rule.default, add_gated_delta_rule_node);
}

} // namespace vkcompute
