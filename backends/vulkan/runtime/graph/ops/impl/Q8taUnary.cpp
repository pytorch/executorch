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

void resize_q8ta_unary_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef self = args.at(1).refs.at(0);
  graph->virtual_resize(out, graph->sizes_of(self));
}

//
// Dispatch nodes
//

void add_q8ta_unary_node(
    ComputeGraph& graph,
    const ValueRef packed_int8_input,
    const ValueRef input_scale,
    const ValueRef input_zp,
    const ValueRef output_scale,
    const ValueRef output_zp,
    const ValueRef packed_int8_output,
    const std::string& op_name) {
  const api::PackedDimInfo& output_info =
      graph.packed_dim_info_of(packed_int8_output);
  const api::PackedDimInfo& input_info =
      graph.packed_dim_info_of(packed_int8_input);

  VK_CHECK_COND(input_info.packed_dim == output_info.packed_dim);
  VK_CHECK_COND(
      input_info.packed_dim_block_size == output_info.packed_dim_block_size);

  float input_scale_val = graph.extract_scalar<float>(input_scale);
  int32_t input_zp_val = graph.extract_scalar<int32_t>(input_zp);

  float output_inv_scale_val = 1.0f / graph.extract_scalar<float>(output_scale);
  int32_t output_zp_val = graph.extract_scalar<int32_t>(output_zp);

  std::string kernel_name = "q8ta_" + op_name;
  add_storage_type_suffix(
      kernel_name, graph.storage_type_of(packed_int8_output));

  vkapi::ParamsBindList param_buffers;
  param_buffers.append(graph.buffer_meta_ubo(packed_int8_output));
  param_buffers.append(graph.buffer_meta_ubo(packed_int8_input));

  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&input_scale_val, sizeof(input_scale_val)),
      PushConstantDataInfo(&input_zp_val, sizeof(input_zp_val)),
      PushConstantDataInfo(&output_inv_scale_val, sizeof(output_inv_scale_val)),
      PushConstantDataInfo(&output_zp_val, sizeof(output_zp_val)),
  };

  const BlockConfig block_config =
      create_block_config_for_tensor(graph, packed_int8_output);

  const ValueRef block_config_ref =
      static_cast<ValueRef>(block_config.as_packed_int());

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      pick_linear_global_wg_with_block_config,
      pick_square_local_wg_with_block_config,
      // Inputs and Outputs
      {{packed_int8_output, vkapi::kWrite}, {packed_int8_input, vkapi::kRead}},
      // Shader params buffers
      param_buffers,
      // Push Constants
      push_constants,
      // Specialization Constants
      {graph.hashed_layout_of(packed_int8_output),
       graph.hashed_layout_of(packed_int8_input),
       block_config.as_packed_int()},
      // Resize args
      {block_config_ref},
      // Resizing Logic
      resize_q8ta_unary_node));
}

//
// High level operator impl
//

void q8ta_relu(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef packed_int8_input = args.at(idx++);
  const ValueRef input_scale = args.at(idx++);
  const ValueRef input_zp = args.at(idx++);
  const ValueRef output_scale = args.at(idx++);
  const ValueRef output_zp = args.at(idx++);
  const ValueRef packed_int8_output = args.at(idx++);

  add_q8ta_unary_node(
      graph,
      packed_int8_input,
      input_scale,
      input_zp,
      output_scale,
      output_zp,
      packed_int8_output,
      "relu");
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.q8ta_relu.default, q8ta_relu);
}

} // namespace vkcompute
