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
    add_dtype_suffix(kernel_name, graph.dtype_of(out));
    add_storage_type_suffix(kernel_name, graph.storage_type_of(out));

    // We calculate a custom 3D grid:
    // X = D (div 4) for the Value dimension
    // Y = 1 (We must process the sequence loop INSIDE the shader sequentially due to recurrence)
    // Z = Batch * Heads
    
    std::vector<int64_t> out_shape = graph.sizes_of(out);
    int batch = out_shape[0];
    int heads = out_shape[1];
    int seq_len = out_shape[2];
    int v_dim = out_shape[3];
    
    utils::uvec3 gwg = {
        utils::div_up(v_dim, 4),
        1,
        batch * heads
    };

    graph.execute_nodes().emplace_back(new DynamicDispatchNode(
        graph,
        VK_KERNEL_FROM_STR(kernel_name),
        gwg,
        default_pick_lwg,
        // Inputs and Outputs (Format: Write, Read)
        {{out, final_state}, {q, k, v, decay, beta, initial_state}},
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
