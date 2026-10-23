/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <memory>

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void resize_gated_delta_rule_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  // Extract inputs
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef final_state = args.at(0).refs.at(1);
  const ValueRef q = args.at(1).refs.at(0);
  const ValueRef v = args.at(1).refs.at(2);
  const ValueRef initial_state = args.at(1).refs.at(5);

  // The output tensor shapes depend directly on the inputs
  std::vector<int64_t> q_sizes = graph->sizes_of(q);
  std::vector<int64_t> v_sizes = graph->sizes_of(v);
  std::vector<int64_t> out_sizes = {
      q_sizes[0], q_sizes[1], q_sizes[2], v_sizes[3]};
  std::vector<int64_t> init_state_sizes = graph->sizes_of(initial_state);
  std::vector<int64_t> final_state_sizes = {
      q_sizes[0],
      init_state_sizes[1],
      init_state_sizes[2],
      init_state_sizes[3]};

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
  // out_shape is [Batch, Heads, Seq_len, Head_V_Dim]
  uint32_t batch = out_shape[0];
  uint32_t heads = out_shape[1];
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

  ValueRef out;
  ValueRef final_state;
  if (graph.val_is_value_list(args[6])) {
    const ValueListPtr out_tuple = graph.get_value_list(args[6]);
    out = out_tuple->at(0);
    final_state = out_tuple->at(1);
  } else {
    out = args[6];
    final_state = args[7];
  }

  std::string kernel_name;
  if (graph.is_buffer_storage(out)) {
    kernel_name = "gated_delta_rule_buffer";
  } else {
    kernel_name = "gated_delta_rule_texture3d_texture3d";
  }
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
      {graph.sizes_ubo(out),
       graph.sizes_ubo(k),
       graph.sizes_ubo(initial_state)},
      // Push Constants
      {},
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
