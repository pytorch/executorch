/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q8taLinear.h>

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

bool q8ta_linear_check_packed_dim_info(const api::PackedDimInfo& info) {
  return info.packed_dim == WHCN::kWidthDim &&
      info.packed_dim_block_size == 4 &&
      info.outer_packed_dim == WHCN::kHeightDim &&
      info.outer_packed_dim_block_size == 4;
}

//
// Workgroup size selection
//

utils::uvec3 q8ta_linear_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;

  const ValueRef out = args.at(0).refs.at(0);

  std::vector<int64_t> out_sizes = graph->sizes_of(out);
  const uint32_t N = utils::val_at(-1, out_sizes);
  const uint32_t M = utils::val_at(-2, out_sizes);

  // Each output tile contains 8 columns (TILE_N4=2 -> 8 output channels)
  const uint32_t N_per_tile = 8;
  const uint32_t M_per_tile = 4;

  const uint32_t num_N_tiles = utils::div_up(N, N_per_tile);
  const uint32_t num_M_tiles = utils::div_up(M, M_per_tile);

  return {num_N_tiles, num_M_tiles, 1};
}

utils::uvec3 q8ta_linear_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  return pick_hw_square_wg_size(
      graph, shader, global_workgroup_size, args, resize_args);
}

//
// Dispatch node
//

void add_q8ta_linear_node(
    ComputeGraph& graph,
    const ValueRef packed_int8_input,
    const ValueRef input_scale,
    const ValueRef input_zp,
    const ValueRef packed_weight,
    const ValueRef packed_weight_sums,
    const ValueRef packed_weight_scales,
    const ValueRef output_scale,
    const ValueRef output_zp,
    const ValueRef bias_data,
    const ValueRef packed_bias,
    const ValueRef packed_int8_output) {
  // Validate packed dim info matches 4H4W layout
  VK_CHECK_COND(q8ta_linear_check_packed_dim_info(
      graph.packed_dim_info_of(packed_int8_input)));
  VK_CHECK_COND(q8ta_linear_check_packed_dim_info(
      graph.packed_dim_info_of(packed_int8_output)));

  float input_scale_val = graph.extract_scalar<float>(input_scale);
  int32_t input_zp_val = graph.extract_scalar<int32_t>(input_zp);

  float output_inv_scale_val = 1.0f / graph.extract_scalar<float>(output_scale);
  int32_t output_zp_val = graph.extract_scalar<int32_t>(output_zp);

  uint32_t apply_bias = 1;
  if (graph.val_is_none(bias_data)) {
    apply_bias = 0;
  }

  std::string kernel_name = "q8ta_linear";
  add_dtype_suffix(kernel_name, graph.dtype_of(packed_weight_scales));

  vkapi::ParamsBindList param_buffers = {
      graph.sizes_ubo(packed_int8_output), graph.sizes_ubo(packed_int8_input)};

  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&input_scale_val, sizeof(input_scale_val)),
      PushConstantDataInfo(&input_zp_val, sizeof(input_zp_val)),
      PushConstantDataInfo(&output_inv_scale_val, sizeof(output_inv_scale_val)),
      PushConstantDataInfo(&output_zp_val, sizeof(output_zp_val)),
  };

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      q8ta_linear_global_wg_size,
      q8ta_linear_local_wg_size,
      // Inputs and Outputs
      {{packed_int8_output, vkapi::kWrite},
       {{packed_int8_input,
         packed_weight,
         packed_weight_sums,
         packed_weight_scales,
         packed_bias},
        vkapi::kRead}},
      // Shader params buffers
      param_buffers,
      // Push Constants
      push_constants,
      // Specialization Constants
      {apply_bias},
      // Resize args
      {},
      // Resizing Logic
      nullptr));
}

//
// High level operator impl
//

void q8ta_linear(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef packed_int8_input = args.at(idx++);
  const ValueRef input_scale = args.at(idx++);
  const ValueRef input_zp = args.at(idx++);
  const ValueRef weight_data = args.at(idx++);
  const ValueRef weight_sums_data = args.at(idx++);
  const ValueRef weight_scales_data = args.at(idx++);
  const ValueRef output_scale = args.at(idx++);
  const ValueRef output_zp = args.at(idx++);
  const ValueRef bias_data = args.at(idx++);
  const ValueRef packed_int8_output = args.at(idx++);

  const int64_t K = graph.size_at<int64_t>(-1, packed_int8_input);
  VK_CHECK_COND(K % 4 == 0);

  QuantizationConfig weight_quant_config(8, kPerChannel, {K});

  // Prepack weight data
  const ValueRef packed_weight =
      prepack_quantized_linear_weight(graph, weight_quant_config, weight_data);
  const ValueRef packed_weight_scales = prepack_standard(
      graph, weight_scales_data, utils::kBuffer, utils::kWidthPacked);
  const ValueRef packed_weight_sums = prepack_standard(
      graph, weight_sums_data, utils::kBuffer, utils::kWidthPacked);

  // Prepack bias data
  TmpTensor dummy_bias(
      &graph,
      {},
      graph.dtype_of(packed_weight_scales),
      utils::kBuffer,
      utils::kWidthPacked);

  ValueRef packed_bias = dummy_bias.vref;
  if (graph.val_is_not_none(bias_data)) {
    packed_bias =
        prepack_standard(graph, bias_data, utils::kBuffer, utils::kWidthPacked);
  }

  add_q8ta_linear_node(
      graph,
      packed_int8_input,
      input_scale,
      input_zp,
      packed_weight,
      packed_weight_sums,
      packed_weight_scales,
      output_scale,
      output_zp,
      bias_data,
      packed_bias,
      packed_int8_output);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.q8ta_linear.default, q8ta_linear);
}

} // namespace vkcompute
