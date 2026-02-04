/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q8taConv2d.h>

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

//
// Shader dispatch utilities
//

utils::uvec3 pick_q8ta_conv2d_pw_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;

  const ValueRef output = args.at(0).refs.at(0);

  const uint32_t W = graph->size_at<uint32_t>(-1, output);
  const uint32_t H = graph->size_at<uint32_t>(-2, output);
  const uint32_t C = graph->size_at<uint32_t>(-3, output);

  // The 4W4C shader processes tiles of:
  // - TILE_N4=2 groups of 4 output channels (8 channels per thread)
  // - TILE_M4=1 groups of 4 widths (4 widths per thread)
  // - 1 height per thread
  constexpr uint32_t TILE_N4 = 2;
  constexpr uint32_t TILE_M4 = 1;

  const uint32_t C4 = utils::div_up_4(C);
  const uint32_t W4 = utils::div_up_4(W);

  // Global workgroup size:
  // x = output channels / (TILE_N4 * 4) = C4 / TILE_N4
  // y = width / (TILE_M4 * 4) = W4 / TILE_M4
  // z = height
  return {utils::div_up(C4, TILE_N4), utils::div_up(W4, TILE_M4), H};
}

utils::uvec3 pick_q8ta_conv2d_pw_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  return pick_hw_square_wg_size(
      graph, shader, global_workgroup_size, args, resize_args);
}

//
// 4W4C shader dispatch utilities
//

utils::uvec3 pick_q8ta_conv2d_pw_4w4c_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;

  const ValueRef output = args.at(0).refs.at(0);

  const uint32_t W = graph->size_at<uint32_t>(-1, output);
  const uint32_t H = graph->size_at<uint32_t>(-2, output);
  const uint32_t C = graph->size_at<uint32_t>(-3, output);

  // The 4W4C shader processes tiles of:
  // - TILE_N4=2 groups of 4 output channels (8 channels per thread)
  // - TILE_M4=1 groups of 4 widths (4 widths per thread)
  // - 1 height per thread
  constexpr uint32_t TILE_N4 = 2;
  constexpr uint32_t TILE_M4 = 1;

  const uint32_t C4 = utils::div_up_4(C);
  const uint32_t W4 = utils::div_up_4(W);

  // Global workgroup size:
  // x = output channels / (TILE_N4 * 4) = C4 / TILE_N4
  // y = width / (TILE_M4 * 4) = W4 / TILE_M4
  // z = height
  return {utils::div_up(C4, TILE_N4), utils::div_up(W4, TILE_M4), H};
}

utils::uvec3 pick_q8ta_conv2d_pw_4w4c_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  return pick_hw_square_wg_size(
      graph, shader, global_workgroup_size, args, resize_args);
}

//
// Prepack nodes
//

ValueRef prepack_quantized_conv2d_pw_weight(
    ComputeGraph& graph,
    const QuantizationConfig& weight_quant_config,
    const ValueRef weight_data,
    const ValueRef input,
    const ValueRef output) {
  VK_CHECK_COND(weight_quant_config.nbits == 8);
  VK_CHECK_COND(weight_quant_config.is_symmetric);

  const int64_t OC = graph.size_at<int64_t>(-3, output);
  const int64_t IC = graph.size_at<int64_t>(-3, input);

  // For pointwise convolution, kernel_size = 1x1
  const int64_t K_h = 1;
  const int64_t K_w = 1;

  const int64_t num_blocks_OC = utils::div_up_4(OC);
  const int64_t num_blocks_IC = utils::div_up_4(IC);

  const int64_t num_blocks_y = num_blocks_IC * K_h;
  const int64_t num_blocks_x = K_w * num_blocks_OC;

  // The packed tensor arranges blocks as [OC_blocks * K_total, IC_blocks]
  const int64_t output_height = num_blocks_y;
  const int64_t output_width = num_blocks_x * 4;

  // Store the original sizes of the weight data to pass to the shader
  utils::ivec4 orig_sizes = {
      utils::safe_downcast<int32_t>(OC),
      utils::safe_downcast<int32_t>(K_h),
      utils::safe_downcast<int32_t>(K_w),
      utils::safe_downcast<int32_t>(IC)};

  std::vector<int64_t> packed_weight_sizes{output_height, output_width};

  utils::StorageType storage_type = utils::kTexture2D;
  uint32_t max_extent = graph.context()->adapter_ptr()->max_texture2d_dim();
  if (output_width > max_extent * 4 || output_height > max_extent) {
    storage_type = utils::kBuffer;
  }

  ValueRef packed_weight = graph.add_tensor(
      packed_weight_sizes,
      vkcompute::vkapi::kInt,
      storage_type,
      utils::kWidthPacked);

  utils::uvec3 global_wg_size = {
      utils::safe_downcast<uint32_t>(num_blocks_x),
      utils::safe_downcast<uint32_t>(num_blocks_y),
      1u};

  std::string kernel_name = "pack_q8_conv2d_weights";
  add_storage_type_suffix(kernel_name, storage_type);

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_wg_size,
      graph.create_local_wg_size(global_wg_size),
      // Inputs and Outputs
      weight_data,
      packed_weight,
      // UBOs
      {},
      // Specialization Constants
      {},
      // Push Constants
      {graph.sizes_pc_of(packed_weight),
       PushConstantDataInfo(&orig_sizes, sizeof(utils::ivec4))}));

  return packed_weight;
}

//
// Dispatch nodes
//

void add_q8ta_conv2d_pw_node(
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
  float input_scale_val = graph.extract_scalar<float>(input_scale);
  int32_t input_zp_val = graph.extract_scalar<int32_t>(input_zp);

  float output_inv_scale_val = 1.0f / graph.extract_scalar<float>(output_scale);
  int32_t output_zp_val = graph.extract_scalar<int32_t>(output_zp);

  uint32_t apply_bias = 1;
  if (graph.val_is_none(bias_data)) {
    apply_bias = 0;
  }

  // Get input channel count for K4_per_group
  const uint32_t IC = graph.size_at<uint32_t>(-3, packed_int8_input);
  const uint32_t K4_per_group = utils::div_up_4(IC);

  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&input_scale_val, sizeof(input_scale_val)),
      PushConstantDataInfo(&input_zp_val, sizeof(input_zp_val)),
      PushConstantDataInfo(&output_inv_scale_val, sizeof(output_inv_scale_val)),
      PushConstantDataInfo(&output_zp_val, sizeof(output_zp_val)),
  };

  std::string kernel_name = "q8ta_conv2d_pw";
  add_dtype_suffix(kernel_name, graph.dtype_of(packed_weight_scales));

  // Pass metadata for both output and input tensors
  vkapi::ParamsBindList param_buffers = {
      graph.buffer_meta_ubo(packed_int8_output),
      graph.buffer_meta_ubo(packed_int8_input)};

  // Build spec constants: apply_bias + layout constants
  vkapi::SpecVarList spec_constants = {
      apply_bias,
      K4_per_group,
      // Layout specialization constants
      graph.hashed_layout_of(packed_int8_output),
      graph.hashed_layout_of(packed_int8_input),
  };

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      pick_q8ta_conv2d_pw_global_wg_size,
      pick_q8ta_conv2d_pw_local_wg_size,
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
      spec_constants,
      // Resize args
      {}));
}

void add_q8ta_conv2d_pw_4w4c_node(
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
  float input_scale_val = graph.extract_scalar<float>(input_scale);
  int32_t input_zp_val = graph.extract_scalar<int32_t>(input_zp);

  float output_inv_scale_val = 1.0f / graph.extract_scalar<float>(output_scale);
  int32_t output_zp_val = graph.extract_scalar<int32_t>(output_zp);

  uint32_t apply_bias = 1;
  if (graph.val_is_none(bias_data)) {
    apply_bias = 0;
  }

  // Get input channel count for K4_per_group
  const uint32_t IC = graph.size_at<uint32_t>(-3, packed_int8_input);
  const uint32_t K4_per_group = utils::div_up_4(IC);

  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&input_scale_val, sizeof(input_scale_val)),
      PushConstantDataInfo(&input_zp_val, sizeof(input_zp_val)),
      PushConstantDataInfo(&output_inv_scale_val, sizeof(output_inv_scale_val)),
      PushConstantDataInfo(&output_zp_val, sizeof(output_zp_val)),
  };

  std::string kernel_name = "q8ta_conv2d_pw_4w4c_ref";
  add_dtype_suffix(kernel_name, graph.dtype_of(packed_weight_scales));

  // Build spec constants: apply_bias + K4_per_group + layout constants
  vkapi::SpecVarList spec_constants = {
      apply_bias,
      K4_per_group,
      // Layout specialization constants
      graph.hashed_layout_of(packed_int8_output),
      graph.hashed_layout_of(packed_int8_input),
  };

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      pick_q8ta_conv2d_pw_4w4c_global_wg_size,
      pick_q8ta_conv2d_pw_4w4c_local_wg_size,
      // Inputs and Outputs
      {{packed_int8_output, vkapi::kWrite},
       {{packed_int8_input,
         packed_weight,
         packed_weight_sums,
         packed_weight_scales,
         packed_bias},
        vkapi::kRead}},
      // Shader params buffers
      {graph.meta_ubo(packed_int8_output), graph.meta_ubo(packed_int8_input)},
      // Push Constants
      push_constants,
      // Specialization Constants
      spec_constants,
      // Resize args
      {}));
}

//
// High level operator impl
//

void q8ta_conv2d_pw(ComputeGraph& graph, const std::vector<ValueRef>& args) {
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

  QuantizationConfig weight_quant_config(8, kPerChannel, {});

  // Prepack weight using pointwise-specific packing
  ValueRef packed_weight = prepack_quantized_conv2d_pw_weight(
      graph,
      weight_quant_config,
      weight_data,
      packed_int8_input,
      packed_int8_output);

  ValueRef packed_weight_sums = prepack_standard(
      graph, weight_sums_data, utils::kBuffer, utils::kWidthPacked);

  ValueRef packed_weight_scales = prepack_standard(
      graph, weight_scales_data, utils::kBuffer, utils::kWidthPacked);

  // Create a dummy tensor to fill the binding slot of the bias tensor if it is
  // not provided. This helps simplify dispatch logic and makes it so that
  // fewer shader variants need to be generated.
  TmpTensor dummy_bias(
      &graph,
      {},
      graph.dtype_of(weight_scales_data),
      utils::kBuffer,
      utils::kWidthPacked);

  ValueRef packed_bias = dummy_bias.vref;
  if (graph.val_is_not_none(bias_data)) {
    packed_bias =
        prepack_standard(graph, bias_data, utils::kBuffer, utils::kWidthPacked);
  }

  // Check if input and output have 4W4C layout for optimized path
  const utils::GPUMemoryLayout inp_layout =
      graph.estimate_memory_layout_of(packed_int8_input);
  const utils::GPUMemoryLayout outp_layout =
      graph.estimate_memory_layout_of(packed_int8_output);

  const bool use_4w4c_path =
      (inp_layout == utils::kPackedInt8_4W4C &&
       outp_layout == utils::kPackedInt8_4W4C);
  (void)use_4w4c_path;

  if (false) {
    add_q8ta_conv2d_pw_4w4c_node(
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
  } else {
    add_q8ta_conv2d_pw_node(
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
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(etvk.q8ta_conv2d_pw.default, q8ta_conv2d_pw);
}

} // namespace vkcompute
