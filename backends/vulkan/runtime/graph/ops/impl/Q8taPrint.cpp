/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q8taPrint.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>

namespace vkcompute {

void add_q8ta_print_node(
    ComputeGraph& graph,
    const ValueRef packed_int8_input) {
  VK_CHECK_COND(graph.dtype_of(packed_int8_input) == vkapi::kInt8x4);

  std::string kernel_name = "q8ta_print";

  vkapi::ParamsBindList param_buffers;
  param_buffers.append(graph.buffer_meta_ubo(packed_int8_input));

  const BlockConfig inp_block_config =
      create_block_config_for_tensor(graph, packed_int8_input);

  int32_t value_ref_val = static_cast<int32_t>(packed_int8_input);

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      {1u, 1u, 1u},
      {1u, 1u, 1u},
      // Inputs and Outputs
      {{packed_int8_input, vkapi::kRead}},
      // Shader params buffers
      param_buffers,
      // Push Constants
      {PushConstantDataInfo(&value_ref_val, sizeof(value_ref_val))},
      // Specialization Constants
      {graph.hashed_layout_of(packed_int8_input),
       inp_block_config.as_packed_int()}));
}

} // namespace vkcompute
