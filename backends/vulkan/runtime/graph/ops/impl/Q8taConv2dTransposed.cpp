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

// Dedicated workgroup size functions for transposed convolution.
// Unlike regular conv2d, transposed conv with stride > 1 causes branch
// divergence along the height dimension (different rows have different
// stride-alignment patterns). Keeping local_y=1 ensures all threads in a
// workgroup process the same height row, maximizing branch coherence.

utils::uvec3 pick_q8ta_conv2d_transposed_global_wg_size(
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

  const uint32_t W4 = utils::div_up_4(W);
  const uint32_t C4 = utils::div_up_4(C);

  return {W4, H, C4};
}

utils::uvec3 pick_q8ta_conv2d_transposed_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  (void)graph;
  (void)args;

  // Always keep local_y=1 to avoid branch divergence between height rows.
  if (global_workgroup_size[0u] >= 6 && global_workgroup_size[2u] >= 6) {
    return {8u, 1u, 8u};
  }
  if (global_workgroup_size[0u] < 2u) {
    return {1u, 1u, 64u};
  }
  if (global_workgroup_size[2u] < 2u) {
    return {64u, 1u, 1u};
  }
  return {16u, 1u, 4u};
}

void add_q8ta_conv2d_transposed_node(
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
    const uint32_t activation_type,
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

  // Transposed convolution only supports dilation=1
  VK_CHECK_COND(
      conv_params.dilation[0] == 1 && conv_params.dilation[1] == 1,
      "q8ta_conv2d_transposed only supports dilation=1");

  // The implementation requires that for grouped convolutions, the input
  // channels per group is a multiple of 4.
  if (conv_params.groups > 1) {
    VK_CHECK_COND(conv_params.in_channels_per_group % 4 == 0);
  }

  // Validate packed dim info for input and output tensors
  VK_CHECK_COND(q8ta_conv2d_check_packed_dim_info(
      graph.packed_dim_info_of(packed_int8_input)));
  VK_CHECK_COND(q8ta_conv2d_check_packed_dim_info(
      graph.packed_dim_info_of(packed_int8_output)));

  // Validate dtype is kInt8x4
  VK_CHECK_COND(graph.dtype_of(packed_int8_input) == vkapi::kInt8x4);
  VK_CHECK_COND(graph.dtype_of(packed_int8_output) == vkapi::kInt8x4);

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

  const bool use_hw_dot =
      graph.context()->adapter_ptr()->supports_int8_dot_product();
  std::string kernel_name =
      use_hw_dot ? "q8ta_conv2d_transposed" : "q8ta_conv2d_transposed_fallback";
  add_dtype_suffix(kernel_name, graph.dtype_of(packed_weight_scales));

  // Pass metadata for both output and input tensors
  vkapi::ParamsBindList param_buffers = {
      graph.buffer_meta_ubo(packed_int8_output),
      graph.buffer_meta_ubo(packed_int8_input),
      graph.create_params_buffer(conv_params)};

  // Build spec constants: apply_bias, activation_type + layout constants
  vkapi::SpecVarList spec_constants = {
      apply_bias,
      activation_type,
      // Layout specialization constants
      graph.hashed_layout_of(packed_int8_input),
      graph.hashed_layout_of(packed_int8_output),
  };

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      pick_q8ta_conv2d_transposed_global_wg_size,
      pick_q8ta_conv2d_transposed_local_wg_size,
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

void q8ta_conv2d_transposed(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
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
  args.at(idx++); // output_padding: only affects output size, not shader
  const ValueRef dilation = args.at(idx++);
  const ValueRef groups = args.at(idx++);
  const ValueRef activation = args.at(idx++);
  const ValueRef packed_int8_output = args.at(idx++);

  uint32_t activation_type_val = static_cast<uint32_t>(
      activation_type_from_string(graph.extract_string(activation)));

  QuantizationConfig weight_quant_config(8, kPerChannel, {});

  // Reuse the conv2d weight packing (after the pattern matcher reshapes the
  // transposed weight to (OC, KH*KW*IC_per_group), the weight layout is
  // identical to regular conv2d)
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
  // not provided
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

  add_q8ta_conv2d_transposed_node(
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
      activation_type_val,
      packed_int8_output);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.q8ta_conv2d_transposed.default, q8ta_conv2d_transposed);
}

} // namespace vkcompute
