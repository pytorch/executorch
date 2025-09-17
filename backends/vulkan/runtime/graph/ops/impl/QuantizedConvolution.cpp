/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

//
// Utility functions
//

struct Conv2DParams {
  utils::ivec2 kernel_size;
  utils::ivec2 stride;
  utils::ivec2 padding;
  utils::ivec2 dilation;
  int32_t groups;
  int32_t out_channels_per_group;
  int32_t in_channels_per_group;
  int32_t logical_K_per_group;
  int32_t K_per_group;
  int32_t K4_per_group;
  int32_t logical_K;
  int32_t K;
  int32_t K4;
};

Conv2DParams create_conv2d_params(
    ComputeGraph& graph,
    const ValueRef& conv_input,
    const ValueRef& conv_output,
    const ValueRef& kernel_size,
    const ValueRef& stride,
    const ValueRef& padding,
    const ValueRef& dilation,
    const ValueRef& groups) {
  const auto kernel_size_list = graph.get_int_list(kernel_size);
  const auto stride_list = graph.get_int_list(stride);
  const auto padding_list = graph.get_int_list(padding);
  const auto dilation_list = graph.get_int_list(dilation);
  const int32_t groups_val = graph.get_int(groups);

  // Pre-compute input and output channels per group

  std::vector<int64_t> out_sizes = graph.sizes_of(conv_output);
  const int32_t out_channels = utils::val_at(-3, out_sizes);
  const int32_t out_channels_per_group = out_channels / groups_val;

  std::vector<int64_t> in_sizes = graph.sizes_of(conv_input);
  const int32_t in_channels = utils::val_at(-3, in_sizes);
  const int32_t in_channels_per_group = in_channels / groups_val;

  // Pre-compute the number of elements along the K dimension per group. This
  // quantity is aligned to the next multiple of 4 to ensure data loads are
  // aligned to texel boundaries.

  const int32_t logical_K_per_group =
      kernel_size_list->at(0) * kernel_size_list->at(1) * in_channels_per_group;
  const int32_t K_per_group = utils::align_up_4(logical_K_per_group);
  const int32_t K4_per_group = K_per_group / 4;

  // Pre-compute the "theoretical" size of the K dim of the input im2col matrix,
  // which represents the flattened convolution window used to compute an output
  // element. This is used for bounds checking.

  const int32_t logical_K =
      kernel_size_list->at(0) * kernel_size_list->at(1) * in_channels;

  const int32_t K = K_per_group * groups_val;
  // Used for texel stride calculations
  const int32_t K4 = K / 4;

  return Conv2DParams{
      // Swap the order from HW to WH
      utils::make_ivec2({kernel_size_list->at(1), kernel_size_list->at(0)}),
      utils::make_ivec2({stride_list->at(1), stride_list->at(0)}),
      utils::make_ivec2({padding_list->at(1), padding_list->at(0)}),
      utils::make_ivec2({dilation_list->at(1), dilation_list->at(0)}),
      groups_val,
      out_channels_per_group,
      in_channels_per_group,
      logical_K_per_group,
      K_per_group,
      K4_per_group,
      logical_K,
      K,
      K4,
  };
}

std::vector<int64_t> calculate_input_im2col_sizes(
    ComputeGraph* graph,
    const ValueRef& input,
    const ValueRef& output,
    const ValueRef& kernel_size,
    const ValueRef& groups) {
  std::vector<int64_t> in_sizes = graph->sizes_of(input);
  const int64_t in_channels = utils::val_at(-3, in_sizes);

  std::vector<int64_t> out_sizes = graph->sizes_of(output);
  const int64_t batches = utils::val_at(-4, out_sizes);
  const int64_t out_height = utils::val_at(-2, out_sizes);
  const int64_t out_width = utils::val_at(-1, out_sizes);

  // Represents the number of channel groups
  const int64_t groups_val = graph->extract_scalar<int64_t>(groups);
  // No need to div_up because in_channels % groups_val = 0
  const int64_t in_channels_per_group = in_channels / groups_val;

  const auto kernel_size_list = graph->get_int_list(kernel_size);

  // Align to the next multiple of 4 to ensure that data loads align nicely with
  // texel boundaries. We want to ensure that the first data element of each
  // group is at the start of its texel.
  const int64_t flattened_kernel_len = utils::align_up_4(
      in_channels_per_group * kernel_size_list->at(0) *
      kernel_size_list->at(1));

  // K -> flattened convolution window (adjusted)
  const int64_t K = flattened_kernel_len * groups_val;
  // M -> number of elements in 2D output plane. This is aligned to the next
  // multiple of 4 since the im2col shader operates on 4x4 blocks.
  const int64_t M = utils::align_up_4(out_height * out_width * batches);

  return {M, K};
}

std::vector<int64_t> calculate_output_im2col_sizes(
    ComputeGraph* graph,
    const ValueRef& output) {
  std::vector<int64_t> out_sizes = graph->sizes_of(output);
  const int64_t batches = utils::val_at(-4, out_sizes);
  const int64_t out_channels = utils::val_at(-3, out_sizes);
  const int64_t out_height = utils::val_at(-2, out_sizes);
  const int64_t out_width = utils::val_at(-1, out_sizes);

  // N -> output channels
  const int64_t N = out_channels;
  // M -> number of elements in 2D output plane
  const int64_t M = out_height * out_width * batches;

  return {M, N};
}

//
// Shader dispatch utilities
//

utils::uvec3 im2col_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef input_image = args.at(1).refs.at(0);
  const ValueRef output_image = resize_args.at(0);
  const ValueRef kernel_size = resize_args.at(1);
  const ValueRef groups = resize_args.at(2);

  std::vector<int64_t> im2col_sizes = calculate_input_im2col_sizes(
      graph, input_image, output_image, kernel_size, groups);
  const uint32_t K = utils::safe_downcast<uint32_t>(im2col_sizes[1]);
  const uint32_t M = utils::safe_downcast<uint32_t>(im2col_sizes[0]);

  // 1 output tile is 4x4 elements
  const uint32_t K4 = utils::div_up(K, 4u);
  const uint32_t M4 = utils::div_up(M, 4u);

  return {K4, M4, 1};
}

utils::uvec3 col2im_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef output = args.at(0).refs.at(0);

  std::vector<int64_t> im2col_sizes =
      calculate_output_im2col_sizes(graph, output);
  const uint32_t N = utils::safe_downcast<uint32_t>(im2col_sizes[1]);
  const uint32_t M = utils::safe_downcast<uint32_t>(im2col_sizes[0]);

  // 1 output tile is 4x4 elements
  const uint32_t N4 = utils::div_up(N, 4u);
  const uint32_t M4 = utils::div_up(M, 4u);

  return {N4, M4, 1};
}

//
// Dispatch nodes
//

void add_input_im2col_node(
    ComputeGraph& graph,
    const ValueRef input_image,
    const ValueRef kernel_size,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef groups,
    const ValueRef output_image,
    const ValueRef input_im2col) {
  Conv2DParams conv_params = create_conv2d_params(
      graph,
      input_image,
      output_image,
      kernel_size,
      stride,
      padding,
      dilation,
      groups);

  std::string kernel_name = "im2col";
  add_storage_type_suffix(kernel_name, graph.storage_type_of(input_im2col));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(input_image));
  add_dtype_suffix(kernel_name, graph.dtype_of(output_image));

  vkapi::ParamsBindList param_buffers = {
      graph.sizes_ubo(input_im2col),
      graph.sizes_ubo(input_image),
      graph.sizes_ubo(output_image),
      graph.create_params_buffer(conv_params)};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      im2col_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{input_im2col, vkapi::kWrite}, {input_image, vkapi::kRead}},
      // Shader params buffers
      param_buffers,
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize args
      {output_image, kernel_size, groups},
      // Resizing Logic
      nullptr));
}

void add_quantize_and_pack_im2col_node(
    ComputeGraph& graph,
    const ValueRef input_image,
    const ValueRef input_scale,
    const ValueRef input_zp,
    const ValueRef kernel_size,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef groups,
    const ValueRef output_image,
    const ValueRef input_int_im2col) {
  Conv2DParams conv_params = create_conv2d_params(
      graph,
      input_image,
      output_image,
      kernel_size,
      stride,
      padding,
      dilation,
      groups);

  float inv_scale = 1.0f / graph.extract_scalar<float>(input_scale);
  int32_t zp = graph.extract_scalar<int32_t>(input_zp);

  // Get shader for quantized conv2d linear tiled
  std::string kernel_name = "quantize_and_pack_im2col";
  add_storage_type_suffix(kernel_name, graph.storage_type_of(input_int_im2col));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(input_image));
  add_dtype_suffix(kernel_name, graph.dtype_of(output_image));

  vkapi::ShaderInfo shader = VK_KERNEL_FROM_STR(kernel_name);

  vkapi::ParamsBindList param_buffers = {
      graph.sizes_ubo(input_int_im2col),
      graph.sizes_ubo(input_image),
      graph.sizes_ubo(output_image),
      graph.create_params_buffer(conv_params)};

  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&inv_scale, sizeof(inv_scale)),
      PushConstantDataInfo(&zp, sizeof(zp)),
  };

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      im2col_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{input_int_im2col, vkapi::kWrite}, {input_image, vkapi::kRead}},
      // Shader params buffers
      param_buffers,
      // Push Constants
      push_constants,
      // Specialization Constants
      {},
      // Resize args
      {output_image, kernel_size, groups},
      // Resizing Logic
      nullptr));
}

void add_conv2d_q8csw_linear_node(
    ComputeGraph& graph,
    const ValueRef input_im2col,
    const ValueRef input_image,
    const ValueRef packed_weight,
    const ValueRef packed_weight_scales,
    const ValueRef bias_data,
    const ValueRef packed_bias,
    const ValueRef kernel_size,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef groups,
    const ValueRef output_image) {
  Conv2DParams conv_params = create_conv2d_params(
      graph,
      input_image,
      output_image,
      kernel_size,
      stride,
      padding,
      dilation,
      groups);

  // One limitation of the current implementation is that for grouped convs,
  // the number of output_image channels per group must be a multiple of 4. One
  // loaded 4x4 weight tile must all belong to the same group.
  if (conv_params.groups > 1) {
    VK_CHECK_COND(conv_params.out_channels_per_group % 4 == 0);
  }

  std::string kernel_name = "conv2d_q8csw_linear_tiled";
  add_storage_type_suffix(kernel_name, graph.storage_type_of(output_image));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(input_im2col));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(packed_weight));
  add_dtype_suffix(kernel_name, graph.dtype_of(output_image));
  vkapi::ShaderInfo shader = VK_KERNEL_FROM_STR(kernel_name);

  vkapi::ParamsBindList param_buffers = {
      graph.sizes_ubo(output_image),
      graph.sizes_ubo(input_image),
      graph.create_params_buffer(conv_params)};

  uint32_t apply_bias = 1;
  if (graph.val_is_none(bias_data)) {
    apply_bias = 0;
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      col2im_global_wg_size,
      quantized_linear_local_wg_size,
      // Inputs and Outputs
      {{output_image, vkapi::kWrite},
       {{input_im2col, packed_weight, packed_weight_scales, packed_bias},
        vkapi::kRead}},
      // Shader params buffers
      param_buffers,
      // Push Constants
      {},
      // Specialization Constants
      {apply_bias},
      // Resize args
      {},
      // Resizing Logic
      nullptr));
}

void add_conv2d_q8ta_q8csw_linear_node(
    ComputeGraph& graph,
    const ValueRef input_int_im2col,
    const ValueRef input_image,
    const ValueRef input_scale,
    const ValueRef input_zp,
    const ValueRef weight_data,
    const ValueRef packed_weight,
    const ValueRef packed_weight_sums,
    const ValueRef packed_weight_scales,
    const ValueRef bias_data,
    const ValueRef packed_bias,
    const ValueRef kernel_size,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef groups,
    const ValueRef output_image) {
  Conv2DParams conv_params = create_conv2d_params(
      graph,
      input_image,
      output_image,
      kernel_size,
      stride,
      padding,
      dilation,
      groups);

  // One limitation of the current implementation is that for grouped convs,
  // the number of output channels per group must be a multiple of 4. One loaded
  // 4x4 weight tile must all belong to the same group.
  if (conv_params.groups > 1) {
    VK_CHECK_COND(conv_params.out_channels_per_group % 4 == 0);
  }

  float scale = graph.extract_scalar<float>(input_scale);
  int32_t zp = graph.extract_scalar<int32_t>(input_zp);

  std::string kernel_name = "conv2d_q8ta_q8csw_linear_tiled";
  add_storage_type_suffix(kernel_name, graph.storage_type_of(output_image));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(input_int_im2col));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(packed_weight));
  add_dtype_suffix(kernel_name, graph.dtype_of(output_image));
  vkapi::ShaderInfo shader = VK_KERNEL_FROM_STR(kernel_name);

  vkapi::ParamsBindList param_buffers = {
      graph.sizes_ubo(output_image),
      graph.sizes_ubo(input_image),
      graph.create_params_buffer(conv_params)};

  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&scale, sizeof(scale)),
      PushConstantDataInfo(&zp, sizeof(zp)),
  };

  uint32_t apply_bias = 1;
  if (graph.val_is_none(bias_data)) {
    apply_bias = 0;
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      col2im_global_wg_size,
      quantized_linear_local_wg_size,
      // Inputs and Outputs
      {{output_image, vkapi::kWrite},
       {{input_int_im2col,
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
      {weight_data},
      // Resizing Logic
      nullptr));
}

//
// High level operator impl
//

void quantized_conv2d_impl(
    ComputeGraph& graph,
    const QuantizationConfig& input_quant_config,
    const QuantizationConfig& weight_quant_config,
    const ValueRef input_image,
    const ValueRef input_scale,
    const ValueRef input_zp,
    const ValueRef weight_data,
    const ValueRef weight_sums_data,
    const ValueRef weight_scales_data,
    const ValueRef bias_data,
    const ValueRef kernel_size,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef groups,
    const ValueRef output_image) {
  VK_CHECK_COND(weight_quant_config.granularity == kPerChannel);
  VK_CHECK_COND(weight_quant_config.nbits == 8);
  VK_CHECK_COND(weight_quant_config.is_symmetric);

  const ValueRef packed_weight =
      prepack_quantized_linear_weight(graph, weight_quant_config, weight_data);
  ValueRef packed_weight_scales = prepack_standard(
      graph, weight_scales_data, utils::kBuffer, utils::kWidthPacked);

  // Create a dummy tensor to fill the binding slot of the bias tensor if it is
  // not provided. This helps simplify dispatch logic and makes it so that
  // fewer shader variants need to be generated.
  TmpTensor dummy_bias(
      &graph,
      {},
      graph.dtype_of(output_image),
      utils::kBuffer,
      utils::kWidthPacked);

  ValueRef packed_bias = dummy_bias.vref;
  if (!graph.val_is_none(bias_data)) {
    packed_bias =
        prepack_standard(graph, bias_data, utils::kBuffer, utils::kWidthPacked);
  }

  std::vector<int64_t> input_im2col_sizes = calculate_input_im2col_sizes(
      &graph, input_image, output_image, kernel_size, groups);

  // Use weight only quantized conv2d if at least one is true:
  // 1. Device does not support int8 dot product
  // 2. Input is not quantized
  if (!graph.can_use_int8_dot_product() ||
      input_quant_config.granularity == kNoQuantization) {
    TmpTensor input_im2col(
        &graph,
        input_im2col_sizes,
        vkapi::kFloat,
        utils::kBuffer,
        utils::kWidthPacked);

    add_input_im2col_node(
        graph,
        input_image,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        output_image,
        input_im2col);

    add_conv2d_q8csw_linear_node(
        graph,
        input_im2col,
        input_image,
        packed_weight,
        packed_weight_scales,
        bias_data,
        packed_bias,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        output_image);
    return;
  } else {
    // Otherwise, use activation + weight quantized conv2d
    VK_CHECK_COND(input_quant_config.granularity == kPerTensor);
    VK_CHECK_COND(weight_quant_config.nbits == 8);
    VK_CHECK_COND(!weight_quant_config.is_dynamic);

    ValueRef packed_weight_sums = prepack_standard(
        graph, weight_sums_data, utils::kBuffer, utils::kWidthPacked);

    TmpTensor input_int_im2col(
        &graph,
        input_im2col_sizes,
        vkapi::kInt8x4,
        utils::kBuffer,
        utils::kPackedInt8_4H4W);

    add_quantize_and_pack_im2col_node(
        graph,
        input_image,
        input_scale,
        input_zp,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        output_image,
        input_int_im2col);

    add_conv2d_q8ta_q8csw_linear_node(
        graph,
        input_int_im2col,
        input_image,
        input_scale,
        input_zp,
        weight_data,
        packed_weight,
        packed_weight_sums,
        packed_weight_scales,
        bias_data,
        packed_bias,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        output_image);
    return;
  };
}

void conv2d_q8ta_q8csw(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef input_image = args.at(idx++);
  const ValueRef input_scale = args.at(idx++);
  const ValueRef input_zp = args.at(idx++);
  const ValueRef weight_data = args.at(idx++);
  const ValueRef weight_sums_data = args.at(idx++);
  const ValueRef weight_scales_data = args.at(idx++);
  const ValueRef bias_data = args.at(idx++);
  const ValueRef kernel_size = args.at(idx++);
  const ValueRef stride = args.at(idx++);
  const ValueRef padding = args.at(idx++);
  const ValueRef dilation = args.at(idx++);
  const ValueRef groups = args.at(idx++);
  const ValueRef output_image = args.at(idx++);

  const int64_t K = graph.size_at<int64_t>(-1, weight_data);

  QuantizationConfig input_quant_config(8, kPerTensor, {}, false);
  QuantizationConfig weight_quant_config(8, kPerChannel, {K});

  quantized_conv2d_impl(
      graph,
      input_quant_config,
      weight_quant_config,
      input_image,
      input_scale,
      input_zp,
      weight_data,
      weight_sums_data,
      weight_scales_data,
      bias_data,
      kernel_size,
      stride,
      padding,
      dilation,
      groups,
      output_image);
}

void conv2d_q8csw(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef input_image = args.at(idx++);
  const ValueRef weight_data = args.at(idx++);
  const ValueRef weight_scales_data = args.at(idx++);
  const ValueRef bias_data = args.at(idx++);
  const ValueRef kernel_size = args.at(idx++);
  const ValueRef stride = args.at(idx++);
  const ValueRef padding = args.at(idx++);
  const ValueRef dilation = args.at(idx++);
  const ValueRef groups = args.at(idx++);
  const ValueRef output_image = args.at(idx++);

  const int64_t K = graph.size_at<int64_t>(-1, weight_data);

  QuantizationConfig input_quant_config(32, kNoQuantization, {});
  QuantizationConfig weight_quant_config(8, kPerChannel, {K});

  quantized_conv2d_impl(
      graph,
      input_quant_config,
      weight_quant_config,
      input_image,
      kDummyValueRef, // input scale
      kDummyValueRef, // input zero point
      weight_data,
      kDummyValueRef, // weight sums
      weight_scales_data,
      bias_data,
      kernel_size,
      stride,
      padding,
      dilation,
      groups,
      output_image);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.conv2d_q8ta_q8csw.default, conv2d_q8ta_q8csw);
  VK_REGISTER_OP(et_vk.conv2d_q8csw.default, conv2d_q8csw);
}

} // namespace vkcompute
