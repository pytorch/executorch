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
#include <executorch/backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

//
// Shader dispatch utilities
//

utils::uvec3 pick_q8ta_im2col_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;

  const ValueRef im2col_output = args.at(0).refs.at(0);

  std::vector<int64_t> im2col_sizes = graph->sizes_of(im2col_output);
  const uint32_t K = utils::safe_downcast<uint32_t>(im2col_sizes[0]);
  const uint32_t H = utils::safe_downcast<uint32_t>(im2col_sizes[1]);
  const uint32_t W = utils::safe_downcast<uint32_t>(im2col_sizes[2]);

  const uint32_t K4 = utils::div_up_4(K);
  const uint32_t W4 = utils::div_up_4(W);

  // Each thread handles one 4x4 block in the output
  return {K4 * W4 * H, 1, 1};
}

utils::uvec3 pick_q8ta_im2col_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)graph;
  (void)shader;
  (void)args;
  (void)resize_args;
  (void)global_workgroup_size;

  return {64, 1, 1};
}

//
// Im2col calculation utilities
//

std::vector<int64_t> calculate_q8ta_im2col_sizes(
    ComputeGraph* graph,
    const ValueRef& input,
    const ValueRef& output,
    const ValueRef& kernel_size,
    const ValueRef& groups) {
  std::vector<int64_t> in_sizes = graph->sizes_of(input);
  const int64_t in_channels = utils::val_at(-3, in_sizes);

  std::vector<int64_t> out_sizes = graph->sizes_of(output);
  const int64_t out_height = utils::val_at(-2, out_sizes);
  const int64_t out_width = utils::val_at(-1, out_sizes);

  const int64_t groups_val = graph->extract_scalar<int64_t>(groups);
  const int64_t in_channels_per_group = in_channels / groups_val;

  const auto kernel_size_list = graph->get_int_list(kernel_size);

  // Align to next multiple of 4 to ensure data loads align nicely with
  // texel boundaries
  const int64_t flattened_kernel_len = utils::align_up_4(
      in_channels_per_group * kernel_size_list->at(0) *
      kernel_size_list->at(1));

  // K -> flattened convolution window (repeated for each group)
  const int64_t K = flattened_kernel_len * groups_val;
  // M -> number of elements in 2D output plane
  const int64_t W = utils::align_up_4(out_width);
  const int64_t H = out_height;

  return {K, H, W};
}

//
// Dispatch nodes
//

void add_q8ta_im2col_node(
    ComputeGraph& graph,
    const ValueRef packed_int8_input,
    const ValueRef kernel_size,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef groups,
    const ValueRef packed_int8_output,
    const ValueRef packed_int8_im2col,
    const int32_t zp) {
  // Validate packed dim info for input and output tensors
  VK_CHECK_COND(q8ta_conv2d_check_packed_dim_info(
      graph.packed_dim_info_of(packed_int8_input)));
  // The the output tensor must be in 4W4C layout
  VK_CHECK_COND(q8ta_conv2d_check_4w4c_packed_dim_info(
      graph.packed_dim_info_of(packed_int8_im2col)));

  Conv2DParams conv_params = create_conv2d_params(
      graph,
      packed_int8_input,
      packed_int8_output,
      kernel_size,
      stride,
      padding,
      dilation,
      groups);

  // At the moment, the im2col path only supports non-grouped convolutions
  VK_CHECK_COND(conv_params.groups == 1);
  // The implementation also requires that input channels is a multiple of 4
  VK_CHECK_COND(conv_params.in_channels_per_group % 4 == 0);

  std::string kernel_name = "q8ta_im2col_4w4c";

  vkapi::ParamsBindList param_buffers = {
      graph.buffer_meta_ubo(packed_int8_im2col),
      graph.buffer_meta_ubo(packed_int8_input),
      graph.create_params_buffer(conv_params)};

  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&zp, sizeof(zp)),
  };

  // Build spec constants: apply_bias + layout constants (for generic shader)
  vkapi::SpecVarList spec_constants = {
      1u,
      graph.hashed_layout_of(packed_int8_im2col),
      graph.hashed_layout_of(packed_int8_input),
  };

  // // Add layout specialization constants (only for generic shader)
  // if (!use_4w4c_path) {
  //   spec_constants.append(graph.hashed_layout_of(packed_int8_input));
  //   spec_constants.append(graph.hashed_layout_of(packed_int8_im2col));
  // }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      pick_q8ta_im2col_global_wg_size,
      pick_q8ta_im2col_local_wg_size,
      // Inputs and Outputs
      {{packed_int8_im2col, vkapi::kWrite}, {packed_int8_input, vkapi::kRead}},
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

//
// High level operator impl
//

void q8ta_conv2d_im2col(
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
  const ValueRef dilation = args.at(idx++);
  const ValueRef groups = args.at(idx++);
  const ValueRef packed_int8_output = args.at(idx++);

  QuantizationConfig weight_quant_config(8, kPerChannel, {});

  // Prepack weight using linear weight packing (for im2col approach)
  ValueRef packed_weight =
      prepack_quantized_linear_weight(graph, weight_quant_config, weight_data);

  ValueRef packed_weight_sums = prepack_standard(
      graph, weight_sums_data, utils::kBuffer, utils::kWidthPacked);

  ValueRef packed_weight_scales = prepack_standard(
      graph, weight_scales_data, utils::kBuffer, utils::kWidthPacked);

  // Create dummy tensor to fill bias binding slot if not provided
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

  // Calculate im2col output sizes
  std::vector<int64_t> im2col_sizes = calculate_q8ta_im2col_sizes(
      &graph, packed_int8_input, packed_int8_output, kernel_size, groups);

  // Create temporary tensor for im2col output (4W4C layout)
  TmpTensor packed_int8_im2col(
      &graph,
      im2col_sizes,
      vkapi::kInt8x4,
      utils::kBuffer,
      utils::kPackedInt8_4W4C);

  int32_t zp = graph.extract_scalar<int32_t>(input_zp);

  // Step 1: Perform im2col transformation
  add_q8ta_im2col_node(
      graph,
      packed_int8_input,
      kernel_size,
      stride,
      padding,
      dilation,
      groups,
      packed_int8_output,
      packed_int8_im2col,
      zp);

  // Step 2: Perform pointwise convolution on the im2col result
  add_q8ta_conv2d_pw_node(
      graph,
      packed_int8_im2col,
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
  VK_REGISTER_OP(etvk.q8ta_conv2d_im2col.default, q8ta_conv2d_im2col);
}

} // namespace vkcompute
