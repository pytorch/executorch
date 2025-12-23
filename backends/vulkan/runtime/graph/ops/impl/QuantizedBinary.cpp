/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/QuantizeDequantize.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

//
// Shader dispatch utilities
//

utils::uvec3 pick_q8ta_q8ta_q8to_binary_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef packed_int8_output = args.at(0).refs.at(0);

  const uint32_t W = graph->size_at<uint32_t>(-1, packed_int8_output);
  const uint32_t H = graph->size_at<uint32_t>(-2, packed_int8_output);
  const uint32_t C = graph->size_at<uint32_t>(-3, packed_int8_output);

  const uint32_t W4 = utils::div_up_4(W);
  const uint32_t C4 = utils::div_up_4(C);

  return {W4 * H * C4, 1, 1};
}

//
// Dispatch nodes
//

void add_q8ta_q8ta_q8to_binary_node(
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

  std::string kernel_name = op_name + "_q8ta_q8ta_q8to";
  add_storage_type_suffix(
      kernel_name, graph.storage_type_of(packed_int8_output));

  vkapi::ParamsBindList param_buffers = {graph.sizes_ubo(packed_int8_output)};

  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&input_a_scale_val, sizeof(input_a_scale_val)),
      PushConstantDataInfo(&input_a_zp_val, sizeof(input_a_zp_val)),
      PushConstantDataInfo(&input_b_scale_val, sizeof(input_b_scale_val)),
      PushConstantDataInfo(&input_b_zp_val, sizeof(input_b_zp_val)),
      PushConstantDataInfo(&output_inv_scale_val, sizeof(output_inv_scale_val)),
      PushConstantDataInfo(&output_zp_val, sizeof(output_zp_val)),
      PushConstantDataInfo(&alpha_val, sizeof(alpha_val)),
  };

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      pick_q8ta_q8ta_q8to_binary_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{packed_int8_output, vkapi::kWrite},
       {{packed_int8_input_a, packed_int8_input_b}, vkapi::kRead}},
      // Shader params buffers
      param_buffers,
      // Push Constants
      push_constants,
      // Specialization Constants
      {},
      // Resize args
      {},
      // Resizing Logic
      nullptr));
}

//
// High level operator impl
//

void add_q8ta_q8ta_q8to(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
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

  add_q8ta_q8ta_q8to_binary_node(
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

//
// Test operators
//

void add_q8ta_q8ta_q8to_test(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef fp_input_a = args.at(idx++);
  const ValueRef fp_input_b = args.at(idx++);
  const ValueRef input_a_scale = args.at(idx++);
  const ValueRef input_a_zp = args.at(idx++);
  const ValueRef input_b_scale = args.at(idx++);
  const ValueRef input_b_zp = args.at(idx++);
  const ValueRef output_scale = args.at(idx++);
  const ValueRef output_zp = args.at(idx++);
  const ValueRef alpha = args.at(idx++);
  const ValueRef fp_output = args.at(idx++);

  TmpTensor packed_int8_input_a(
      &graph,
      graph.sizes_of(fp_input_a),
      vkapi::kInt8x4,
      utils::kBuffer,
      utils::kPackedInt8_4W4C);

  TmpTensor packed_int8_input_b(
      &graph,
      graph.sizes_of(fp_input_b),
      vkapi::kInt8x4,
      utils::kBuffer,
      utils::kPackedInt8_4W4C);

  TmpTensor packed_int8_output(
      &graph,
      graph.sizes_of(fp_output),
      vkapi::kInt8x4,
      utils::kBuffer,
      utils::kPackedInt8_4W4C);

  add_quantize_and_pack_4w4c_node(
      graph, fp_input_a, input_a_scale, input_a_zp, packed_int8_input_a);

  add_quantize_and_pack_4w4c_node(
      graph, fp_input_b, input_b_scale, input_b_zp, packed_int8_input_b);

  std::vector<ValueRef> add_args = {
      packed_int8_input_a,
      packed_int8_input_b,
      input_a_scale,
      input_a_zp,
      input_b_scale,
      input_b_zp,
      output_scale,
      output_zp,
      alpha,
      packed_int8_output};

  add_q8ta_q8ta_q8to(graph, add_args);

  add_unpack_4w4c_and_dequantize_node(
      graph, packed_int8_output, output_scale, output_zp, fp_output);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.add_q8ta_q8ta_q8to.default, add_q8ta_q8ta_q8to);
  VK_REGISTER_OP(et_vk.add_q8ta_q8ta_q8to.test, add_q8ta_q8ta_q8to_test);
}

} // namespace vkcompute
