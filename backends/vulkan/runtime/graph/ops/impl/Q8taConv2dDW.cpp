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
#include <executorch/backends/vulkan/runtime/graph/ops/impl/ConvolutionUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

//
// Shader dispatch utilities
//

utils::uvec3 pick_q8ta_conv2d_dw_global_wg_size(
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

  // Each thread processes 4 adjacent width positions and 4 channels (4Wx4C
  // tile)
  const uint32_t W4 = utils::div_up_4(W);
  const uint32_t C4 = utils::div_up_4(C);

  return {W4, H, C4};
}

utils::uvec3 pick_q8ta_conv2d_dw_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)graph;
  (void)shader;
  (void)args;
  (void)resize_args;

  // Some inactive invocations are okay; set 6 as the threshold to use the
  // a square wg size.
  if (global_workgroup_size[0u] >= 6 && global_workgroup_size[2u] >= 6) {
    return {8u, 1u, 8u};
  }
  // If channels dim is sufficiently small, then bias towards width dim to
  // reduce the number of inactive invocations.
  if (global_workgroup_size[2u] < 2u) {
    return {64u, 1u, 1u};
  }
  return {16u, 1u, 4u};
}

utils::uvec3 int8_conv2d_dw_global_wg_size(
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

  return {C4 * W4 * H, 1, 1};
}

//
// Prepack nodes
//

ValueRef prepack_quantized_conv2d_dw_weight(
    ComputeGraph& graph,
    const QuantizationConfig& weight_quant_config,
    const ValueRef weight_data,
    const ValueRef kernel_size) {
  VK_CHECK_COND(weight_quant_config.nbits == 8);
  VK_CHECK_COND(weight_quant_config.is_symmetric);

  std::vector<int64_t> weight_orig_sizes = graph.sizes_of(weight_data);
  const int64_t ndim = graph.dim_of(weight_data);

  // For depthwise convolution, expect weight layout [K_h, aligned_K_w, OC]
  VK_CHECK_COND(ndim == 3);
  int64_t K_h = weight_orig_sizes.at(0);
  int64_t K_w = weight_orig_sizes.at(1);
  int64_t aligned_K_w = utils::align_up_4(K_w);
  int64_t OC = weight_orig_sizes.at(2);

  // The packing format packs the weight tensor into blocks of 4 output channels
  // (OC) and 4 kernel elements (K_h * aligned_K_w)
  int64_t OC_per_block = 4;
  int64_t K_per_block = 4;

  // To figure out the size of the output tensor, determine the number of blocks
  // along each dimension.
  const int64_t total_K_elements = K_h * aligned_K_w;
  const int64_t num_blocks_K = utils::div_up(total_K_elements, K_per_block);
  const int64_t num_blocks_OC = utils::div_up(OC, OC_per_block);

  // The blocks are arranged in a transposed manner, such that the transposed
  // weight block is indexed like packed_weights[k4][oc4] - this is to allow for
  // optimal memory coalescing when computing the depthwise convolution.
  int64_t output_height = num_blocks_K;
  // The base dtype of the packed tensor is int32 (each int32 contains 4x 8bit
  // values) and each block is represented as a ivec4. Therefore the width dim
  // of the packed tensor is multiplied by 4.
  int64_t output_width = num_blocks_OC * 4;

  // Store the original sizes of the weight data to pass to the shader
  utils::ivec3 orig_sizes = {
      utils::safe_downcast<int32_t>(K_h),
      utils::safe_downcast<int32_t>(K_w),
      utils::safe_downcast<int32_t>(OC)};

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
      utils::safe_downcast<uint32_t>(num_blocks_OC),
      utils::safe_downcast<uint32_t>(num_blocks_K),
      1u};

  std::string kernel_name = "pack_q8_conv2d_dw_weights";
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
       PushConstantDataInfo(&orig_sizes, sizeof(utils::ivec3))}));

  return packed_weight;
}

//
// Dispatch nodes
//

void add_conv2d_dw_q8ta_q8csw_q8to_4w4c_node(
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
    const ValueRef kernel_size,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef groups,
    const ValueRef packed_int8_output) {
  Conv2DParams conv_params = create_conv2d_params(
      graph,
      packed_int8_input,
      packed_int8_output,
      kernel_size,
      stride,
      padding,
      dilation,
      groups);

  // Verify this is actually a depthwise convolution
  const int64_t groups_val = graph.extract_scalar<int64_t>(groups);
  const int64_t in_channels = graph.size_at<int64_t>(-3, packed_int8_input);
  VK_CHECK_COND(groups_val == in_channels);

  float input_scale_val = graph.extract_scalar<float>(input_scale);
  int32_t input_zp_val = graph.extract_scalar<int32_t>(input_zp);

  float output_inv_scale_val = 1.0f / graph.extract_scalar<float>(output_scale);
  int32_t output_zp_val = graph.extract_scalar<int32_t>(output_zp);

  uint32_t apply_bias = 1;
  if (graph.val_is_none(bias_data)) {
    apply_bias = 0;
  }

  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&input_scale_val, sizeof(input_scale_val)),
      PushConstantDataInfo(&input_zp_val, sizeof(input_zp_val)),
      PushConstantDataInfo(&output_inv_scale_val, sizeof(output_inv_scale_val)),
      PushConstantDataInfo(&output_zp_val, sizeof(output_zp_val)),
  };

  std::string kernel_name = "conv2d_dw_q8ta_q8csw_q8to";
  add_storage_type_suffix(
      kernel_name, graph.storage_type_of(packed_int8_output));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(packed_weight));
  add_dtype_suffix(kernel_name, graph.dtype_of(packed_weight_scales));

  vkapi::ParamsBindList param_buffers = {
      graph.sizes_ubo(packed_int8_output), graph.sizes_ubo(packed_int8_input)};

  vkapi::SpecVarList spec_constants =
      GenerateSpecConstants(graph, conv_params, groups, apply_bias);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      int8_conv2d_dw_global_wg_size,
      default_pick_local_wg_size,
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
      {},
      // Resizing Logic
      nullptr));
}

void add_q8ta_conv2d_dw_node(
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
    const ValueRef kernel_size,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef groups,
    const ValueRef packed_int8_output) {
  Conv2DParams conv_params = create_conv2d_params(
      graph,
      packed_int8_input,
      packed_int8_output,
      kernel_size,
      stride,
      padding,
      dilation,
      groups);

  // Verify this is actually a depthwise convolution
  const int64_t groups_val = graph.extract_scalar<int64_t>(groups);
  const int64_t in_channels = graph.size_at<int64_t>(-3, packed_int8_input);
  VK_CHECK_COND(groups_val == in_channels);

  float input_scale_val = graph.extract_scalar<float>(input_scale);
  int32_t input_zp_val = graph.extract_scalar<int32_t>(input_zp);

  float output_inv_scale_val = 1.0f / graph.extract_scalar<float>(output_scale);
  int32_t output_zp_val = graph.extract_scalar<int32_t>(output_zp);

  uint32_t apply_bias = 1;
  if (graph.val_is_none(bias_data)) {
    apply_bias = 0;
  }

  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&input_scale_val, sizeof(input_scale_val)),
      PushConstantDataInfo(&input_zp_val, sizeof(input_zp_val)),
      PushConstantDataInfo(&output_inv_scale_val, sizeof(output_inv_scale_val)),
      PushConstantDataInfo(&output_zp_val, sizeof(output_zp_val)),
  };

  std::string kernel_name = "q8ta_conv2d_dw";
  add_dtype_suffix(kernel_name, graph.dtype_of(packed_weight_scales));

  // Pass metadata for both output and input tensors
  vkapi::ParamsBindList param_buffers = {
      graph.buffer_meta_ubo(packed_int8_output),
      graph.buffer_meta_ubo(packed_int8_input),
      graph.create_params_buffer(conv_params)};

  // Build spec constants: apply_bias + layout constants
  vkapi::SpecVarList spec_constants = {
      apply_bias,
      // Layout specialization constants
      graph.hashed_layout_of(packed_int8_input),
      graph.hashed_layout_of(packed_int8_output),
  };

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      pick_q8ta_conv2d_dw_global_wg_size,
      pick_q8ta_conv2d_dw_local_wg_size,
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

//
// High level operator impl
//

void q8ta_conv2d_dw(ComputeGraph& graph, const std::vector<ValueRef>& args) {
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
  const ValueRef kernel_size = args.at(idx++);
  const ValueRef stride = args.at(idx++);
  const ValueRef padding = args.at(idx++);
  const ValueRef dilation = args.at(idx++);
  const ValueRef groups = args.at(idx++);
  const ValueRef packed_int8_output = args.at(idx++);

  QuantizationConfig weight_quant_config(8, kPerChannel, {});

  // Prepack weight using depthwise-specific packing
  ValueRef packed_weight = prepack_quantized_conv2d_dw_weight(
      graph, weight_quant_config, weight_data, kernel_size);

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

  add_q8ta_conv2d_dw_node(
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
      kernel_size,
      stride,
      padding,
      dilation,
      groups,
      packed_int8_output);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(etvk.q8ta_conv2d_dw.default, q8ta_conv2d_dw);
}

} // namespace vkcompute
