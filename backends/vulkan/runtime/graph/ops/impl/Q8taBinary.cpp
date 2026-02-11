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

//
// Dispatch nodes
//

void add_q8ta_binary_node(
    ComputeGraph& graph,
    const ValueRef packed_int8_input_a,
    const ValueRef packed_int8_input_b,
    const ValueRef input_a_scale,
    const ValueRef input_a_zp,
    const ValueRef input_b_scale,
    const ValueRef input_b_zp,
    const ValueRef output_scale,
    const ValueRef output_zp,
    const ValueRef alpha,
    const ValueRef packed_int8_output,
    const std::string& op_name) {
  float input_a_scale_val = graph.extract_scalar<float>(input_a_scale);
  int32_t input_a_zp_val = graph.extract_scalar<int32_t>(input_a_zp);
  float input_b_scale_val = graph.extract_scalar<float>(input_b_scale);
  int32_t input_b_zp_val = graph.extract_scalar<int32_t>(input_b_zp);

  float output_inv_scale_val = 1.0f / graph.extract_scalar<float>(output_scale);
  int32_t output_zp_val = graph.extract_scalar<int32_t>(output_zp);

  float alpha_val = 1.0f;
  // String is checked since some ops pass in an unused string argument in
  // place of alpha
  if (is_valid(alpha) && !graph.val_is_string(alpha)) {
    alpha_val = graph.extract_scalar<float>(alpha);
  }

  std::string kernel_name = "q8ta_" + op_name;
  add_storage_type_suffix(
      kernel_name, graph.storage_type_of(packed_int8_output));

  // Pass metadata for output and input tensors
  vkapi::ParamsBindList param_buffers;
  param_buffers.append(graph.buffer_meta_ubo(packed_int8_output));
  param_buffers.append(graph.buffer_meta_ubo(packed_int8_input_a));
  param_buffers.append(graph.buffer_meta_ubo(packed_int8_input_b));

  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&input_a_scale_val, sizeof(input_a_scale_val)),
      PushConstantDataInfo(&input_a_zp_val, sizeof(input_a_zp_val)),
      PushConstantDataInfo(&input_b_scale_val, sizeof(input_b_scale_val)),
      PushConstantDataInfo(&input_b_zp_val, sizeof(input_b_zp_val)),
      PushConstantDataInfo(&output_inv_scale_val, sizeof(output_inv_scale_val)),
      PushConstantDataInfo(&output_zp_val, sizeof(output_zp_val)),
      PushConstantDataInfo(&alpha_val, sizeof(alpha_val)),
  };

  // Create block config for output tensor: inner_dim = output's packed_dim
  const BlockConfig block_config =
      create_block_config_for_tensor(graph, packed_int8_output);

  // Cast block config to ValueRef for pick_linear_global_wg_with_block_config
  const ValueRef block_config_ref =
      static_cast<ValueRef>(block_config.as_packed_int());

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      pick_linear_global_wg_with_block_config,
      pick_square_local_wg_with_block_config,
      // Inputs and Outputs
      {{packed_int8_output, vkapi::kWrite},
       {{packed_int8_input_a, packed_int8_input_b}, vkapi::kRead}},
      // Shader params buffers
      param_buffers,
      // Push Constants
      push_constants,
      // Specialization Constants
      {graph.hashed_layout_of(packed_int8_output),
       graph.hashed_layout_of(packed_int8_input_a),
       block_config.as_packed_int()},
      // Resize args
      {block_config_ref},
      // Resizing Logic
      nullptr));
}

//
// High level operator impl
//

void q8ta_add(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef packed_int8_input_a = args.at(idx++);
  const ValueRef packed_int8_input_b = args.at(idx++);
  const ValueRef input_a_scale = args.at(idx++);
  const ValueRef input_a_zp = args.at(idx++);
  const ValueRef input_b_scale = args.at(idx++);
  const ValueRef input_b_zp = args.at(idx++);
  const ValueRef output_scale = args.at(idx++);
  const ValueRef output_zp = args.at(idx++);
  const ValueRef alpha = args.at(idx++);
  const ValueRef packed_int8_output = args.at(idx++);

  add_q8ta_binary_node(
      graph,
      packed_int8_input_a,
      packed_int8_input_b,
      input_a_scale,
      input_a_zp,
      input_b_scale,
      input_b_zp,
      output_scale,
      output_zp,
      alpha,
      packed_int8_output,
      "add");
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.q8ta_add.default, q8ta_add);
}

} // namespace vkcompute
