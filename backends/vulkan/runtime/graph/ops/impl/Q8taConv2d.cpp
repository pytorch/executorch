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
// Workgroup size selection functions
//

/**
 * Computes a global workgroup size for q8ta_conv2d where:
 *   - For channels-fastest output (e.g., 4C): x = C4, y = H, z = W4
 *   - For width-fastest output (e.g., 4C1W): x = W4, y = H, z = C4
 *
 * The x/z assignment matches the shader's dynamic thread assignment based on
 * fastest_dim (dim_order[0]), ensuring consecutive threads access consecutive
 * elements along the fastest moving dimension for optimal memory coalescing.
 *
 * Each thread processes a 4Wx4C tile of output elements.
 */
utils::uvec3 pick_q8ta_conv2d_global_wg_size(
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

/**
 * Picks a local workgroup size for q8ta_conv2d with adaptive sizing based on
 * tensor dimensions. Uses experimentation results:
 *   - {4, 2, 8} for medium tensors: +57% improvement on 81x81
 *   - {8, 1, 8} for very large tensors: best baseline performance
 *   - {64, 1, 1} for narrow channel dimensions: minimize inactive invocations
 */
utils::uvec3 pick_q8ta_conv2d_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;

  const ValueRef output = args.at(0).refs.at(0);

  // Get actual tensor dimensions for adaptive sizing
  const uint32_t H = graph->size_at<uint32_t>(-2, output);

  // For very large tensors (H >= 100 and large x/z), use {8, 1, 8}
  // This configuration performed best for 128x128 tensors in experiments
  if (H >= 100 && global_workgroup_size[0u] >= 24 &&
      global_workgroup_size[2u] >= 24) {
    return {8u, 1u, 8u};
  }

  // For medium-sized tensors, use {4, 2, 8} for better height parallelism
  // This configuration showed +57% improvement on 81x81 tensors
  if (global_workgroup_size[0u] >= 4 && global_workgroup_size[1u] >= 2 &&
      global_workgroup_size[2u] >= 8) {
    return {4u, 2u, 8u};
  }

  // For tensors with sufficient x and z dimensions, use square configuration
  if (global_workgroup_size[0u] >= 6 && global_workgroup_size[2u] >= 6) {
    return {8u, 1u, 8u};
  }

  // If x dimension is very small, bias towards z dimension
  if (global_workgroup_size[0u] < 2u) {
    return {1u, 1u, 64u};
  }

  // If z dimension is very small, bias towards x dimension
  if (global_workgroup_size[2u] < 2u) {
    return {64u, 1u, 1u};
  }

  return {16u, 1u, 4u};
}

//
// Prepack nodes
//

ValueRef prepack_quantized_conv2d_weight(
    ComputeGraph& graph,
    const QuantizationConfig& weight_quant_config,
    const ValueRef weight_data,
    const ValueRef input,
    const ValueRef output,
    const ValueRef groups,
    const ValueRef kernel_size) {
  VK_CHECK_COND(weight_quant_config.nbits == 8);
  VK_CHECK_COND(weight_quant_config.is_symmetric);

  const int32_t groups_val = graph.get_int(groups);

  const int64_t OC = graph.size_at<int64_t>(-3, output);
  const int64_t IC = graph.size_at<int64_t>(-3, input) / groups_val;

  int64_t K_h;
  int64_t K_w;

  {
    const auto kernel_size_list = graph.get_int_list(kernel_size);
    K_h = kernel_size_list->at(0);
    K_w = kernel_size_list->at(1);
  }

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

void add_q8ta_conv2d_node(
    ComputeGraph& graph,
    const ValueRef packed_int8_input,
    const ValueRef packed_int8_input_im2col,
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
  (void)packed_int8_input_im2col; // Not used in general shader

  Conv2DParams conv_params = create_conv2d_params(
      graph,
      packed_int8_input,
      packed_int8_output,
      kernel_size,
      stride,
      padding,
      dilation,
      groups);

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

  // Select shader based on layout
  std::string kernel_name = "q8ta_conv2d";
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
      pick_q8ta_conv2d_global_wg_size,
      pick_q8ta_conv2d_local_wg_size,
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

void q8ta_conv2d(ComputeGraph& graph, const std::vector<ValueRef>& args) {
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

  // Prepack weight using the conv2d weight packing for the general shader
  ValueRef packed_weight = prepack_quantized_conv2d_weight(
      graph,
      weight_quant_config,
      weight_data,
      packed_int8_input,
      packed_int8_output,
      groups,
      kernel_size);

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

  // The general q8ta_conv2d shader does not use im2col, so pass input as im2col
  add_q8ta_conv2d_node(
      graph,
      packed_int8_input,
      packed_int8_input, // packed_int8_input_im2col - not used in general
                         // shader
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
  VK_REGISTER_OP(etvk.q8ta_conv2d.default, q8ta_conv2d);
}

} // namespace vkcompute
