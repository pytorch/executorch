/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/ConvolutionUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q8taConv2d.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q8taQuantizeDequantize.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/QuantizeDequantize.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/QuantizedConvolution.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/QuantizedLinear.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

//
// Utility functions
//

bool is_pointwise(ComputeGraph* graph, const ValueRef& kernel_size) {
  const auto kernel_size_list = graph->get_int_list(kernel_size);
  return kernel_size_list->at(0) == 1 && kernel_size_list->at(1) == 1;
}

bool is_s1p1d1(
    ComputeGraph* graph,
    const ValueRef& stride,
    const ValueRef& padding,
    const ValueRef& dilation) {
  const auto stride_list = graph->get_int_list(stride);
  const auto padding_list = graph->get_int_list(padding);
  const auto dilation_list = graph->get_int_list(dilation);
  if (stride_list->at(0) != 1 && stride_list->at(1) != 1) {
    return false;
  }
  if (padding_list->at(0) != 1 && padding_list->at(1) != 1) {
    return false;
  }
  if (dilation_list->at(0) != 1 && dilation_list->at(1) != 1) {
    return false;
  }
  return true;
}

bool is_s1p0d1_pointwise(
    ComputeGraph* graph,
    const ValueRef& kernel_size,
    const ValueRef& stride,
    const ValueRef& padding,
    const ValueRef& dilation) {
  if (is_pointwise(graph, kernel_size)) {
    const auto stride_list = graph->get_int_list(stride);
    const auto padding_list = graph->get_int_list(padding);
    const auto dilation_list = graph->get_int_list(dilation);
    if (stride_list->at(0) != 1 && stride_list->at(1) != 1) {
      return false;
    }
    if (padding_list->at(0) != 0 && padding_list->at(1) != 0) {
      return false;
    }
    if (dilation_list->at(0) != 1 && dilation_list->at(1) != 1) {
      return false;
    }
    return true;
  }
  return false;
}

bool should_use_im2col(
    ComputeGraph* graph,
    const ValueRef kernel_size,
    const ValueRef groups) {
  const auto kernel_size_list = graph->get_int_list(kernel_size);

  // Always use im2col for pointwise convolutions
  if (kernel_size_list->at(0) * kernel_size_list->at(1) == 1) {
    return true;
  }

  // For large kernel sizes, the im2col matrix will be too big. Not only will
  // this result in a larger footprint for the im2col matrix, but the cost of
  // performing the im2col procedure will also become prohibitive. In these
  // cases it is faster to just compute convolution directly without going
  // through im2col. Empirically, im2col works well for 3x3 convolution and
  // not for 5x5 convolution, so set the limit at 10.
  if (kernel_size_list->at(0) * kernel_size_list->at(1) > 10) {
    return false;
  }

  // Only use im2col for non-grouped convolutions; manual experimentation shows
  // that im2col becomes very slow when dealing with grouped convolutions. The
  // reason for this is likely that memory access in the im2col shader becomes
  // too non-linear due to needed to keep convolution groups contiguous in
  // in memory. This means that the channels of the input tensor (which are
  // originally contiguous in memory) will be split up during the im2col
  // procedure.
  return graph->get_int(groups) == 1;
}

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

std::vector<int64_t> calculate_packed_int8_input_im2col_sizes(
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

  // K -> flattened convolution window (repeated for each group)
  const int64_t K = flattened_kernel_len * groups_val;
  // M -> number of elements in 2D output plane. This is aligned to the next
  // multiple of 4 since the im2col shader operates on 4x4 blocks.
  const int64_t W = utils::align_up_4(out_width);
  const int64_t H = out_height;

  return {K, H, W};
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

utils::uvec3 im2col_packed_int8_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef input_im2col = args.at(0).refs.at(0);

  std::vector<int64_t> im2col_sizes = graph->sizes_of(input_im2col);
  const uint32_t K = utils::safe_downcast<uint32_t>(im2col_sizes[0]);
  const uint32_t H = utils::safe_downcast<uint32_t>(im2col_sizes[1]);
  const uint32_t W = utils::safe_downcast<uint32_t>(im2col_sizes[2]);

  const uint32_t K4 = utils::div_up(K, 4u);
  const uint32_t W4 = utils::div_up(W, 4u);

  return {K4 * W4 * H, 1, 1};
}

utils::uvec3 im2col_packed_int8_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  return {64, 1, 1};
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

utils::uvec3 pick_static_quantized_conv2d_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef packed_int8_output = args.at(0).refs.at(0);

  const uint32_t W = graph->size_at<uint32_t>(-1, packed_int8_output);
  const uint32_t H = graph->size_at<uint32_t>(-2, packed_int8_output);
  const uint32_t C = graph->size_at<uint32_t>(-3, packed_int8_output);

  uint32_t C_per_tile = 4;
  uint32_t W_per_tile = 4;

  if (shader.kernel_name.find("linear") != std::string::npos) {
    C_per_tile = 8;
  }

  const uint32_t num_W_tiles = utils::div_up(W, W_per_tile);
  const uint32_t num_C_tiles = utils::div_up(C, C_per_tile);

  return {num_C_tiles, num_W_tiles, H};
}

utils::uvec3 pick_static_quantized_conv2d_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  return pick_hw_square_wg_size(
      graph, shader, global_workgroup_size, args, resize_args);
}

//
// Dispatch nodes
//
vkapi::SpecVarList GenerateSpecConstants(
    ComputeGraph& graph,
    Conv2DParams& conv_params,
    const ValueRef& groups,
    uint32_t apply_bias) {
  uint32_t conv2d_params_stride_x = conv_params.stride[0];
  uint32_t conv2d_params_stride_y = conv_params.stride[1];
  uint32_t conv2d_params_padding_x = conv_params.padding[0];
  uint32_t conv2d_params_padding_y = conv_params.padding[1];
  uint32_t conv2d_params_dilation_x = conv_params.dilation[0];
  uint32_t conv2d_params_dilation_y = conv_params.dilation[1];
  uint32_t conv2d_params_kernel_size_x = conv_params.kernel_size[0];
  uint32_t conv2d_params_kernel_size_y = conv_params.kernel_size[1];
  uint32_t in_channels_per_group = conv_params.in_channels_per_group;
  uint32_t out_channels_per_group = conv_params.out_channels_per_group;
  uint32_t K4_per_group = conv_params.K4_per_group;
  uint32_t K4 = conv_params.K4;
  uint32_t K_per_group = conv_params.K_per_group;
  uint32_t logical_K_per_group = conv_params.logical_K_per_group;
  uint32_t logical_K = conv_params.logical_K;
  uint32_t groups_val = graph.get_int(groups);

  vkapi::SpecVarList spec_constants = {
      apply_bias,
      conv2d_params_stride_x,
      conv2d_params_stride_y,
      conv2d_params_padding_x,
      conv2d_params_padding_y,
      conv2d_params_dilation_x,
      conv2d_params_dilation_y,
      conv2d_params_kernel_size_x,
      conv2d_params_kernel_size_y,
      in_channels_per_group,
      out_channels_per_group,
      K4_per_group,
      K4,
      K_per_group,
      logical_K,
      logical_K_per_group,
      groups_val};

  return spec_constants;
}

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
      graph.sizes_ubo(output_image)};

  vkapi::SpecVarList spec_constants =
      GenerateSpecConstants(graph, conv_params, groups);

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
      spec_constants,
      // Resize args
      {output_image, kernel_size, groups},
      // Resizing Logic
      nullptr));
}

void add_input_im2col_packed_int8_node(
    ComputeGraph& graph,
    const ValueRef input,
    const ValueRef input_scale,
    const ValueRef input_zp,
    const ValueRef kernel_size,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef groups,
    const ValueRef output,
    const ValueRef input_im2col) {
  Conv2DParams conv_params = create_conv2d_params(
      graph, input, output, kernel_size, stride, padding, dilation, groups);

  float inv_scale = 1.0f / graph.extract_scalar<float>(input_scale);
  int32_t zp = graph.extract_scalar<int32_t>(input_zp);

  std::string kernel_name = "im2col_packed_int8";
  add_storage_type_suffix(kernel_name, graph.storage_type_of(input_im2col));

  vkapi::ParamsBindList param_buffers = {
      graph.sizes_ubo(input_im2col),
      graph.sizes_ubo(output),
      graph.sizes_ubo(input)};

  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&inv_scale, sizeof(inv_scale)),
      PushConstantDataInfo(&zp, sizeof(zp)),
  };

  vkapi::SpecVarList spec_constants =
      GenerateSpecConstants(graph, conv_params, groups);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      im2col_packed_int8_global_wg_size,
      im2col_packed_int8_local_wg_size,
      // Inputs and Outputs
      {{input_im2col, vkapi::kWrite}, {input, vkapi::kRead}},
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
      graph.sizes_ubo(output_image)};

  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&inv_scale, sizeof(inv_scale)),
      PushConstantDataInfo(&zp, sizeof(zp)),
  };

  vkapi::SpecVarList spec_constants =
      GenerateSpecConstants(graph, conv_params, groups);

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
      spec_constants,
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
      graph.sizes_ubo(output_image), graph.sizes_ubo(input_image)};

  uint32_t apply_bias = 1;
  if (graph.val_is_none(bias_data)) {
    apply_bias = 0;
  }

  vkapi::SpecVarList spec_constants =
      GenerateSpecConstants(graph, conv_params, groups, apply_bias);

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
      spec_constants,
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
      graph.sizes_ubo(output_image), graph.sizes_ubo(input_image)};

  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&scale, sizeof(scale)),
      PushConstantDataInfo(&zp, sizeof(zp)),
  };

  uint32_t apply_bias = 1;
  if (graph.val_is_none(bias_data)) {
    apply_bias = 0;
  }

  vkapi::SpecVarList spec_constants =
      GenerateSpecConstants(graph, conv_params, groups, apply_bias);

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
      spec_constants,
      // Resize args
      {weight_data},
      // Resizing Logic
      nullptr));
}

void add_conv2d_q8ta_q8csw_q8to_4w4c_node(
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

  // Use the optimized im2col or direct shader for 4W4C layout
  const bool use_im2col = should_use_im2col(&graph, kernel_size, groups);

  std::string kernel_name = use_im2col ? "conv2d_q8ta_q8csw_q8to_linear_tiled"
                                       : "conv2d_q8ta_q8csw_q8to";
  add_storage_type_suffix(
      kernel_name, graph.storage_type_of(packed_int8_output));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(packed_weight));
  add_dtype_suffix(kernel_name, graph.dtype_of(packed_weight_scales));

  vkapi::ParamsBindList param_buffers = {
      graph.sizes_ubo(packed_int8_output),
      graph.sizes_ubo(packed_int8_input_im2col)};

  vkapi::SpecVarList spec_constants =
      GenerateSpecConstants(graph, conv_params, groups, apply_bias);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      pick_static_quantized_conv2d_global_wg_size,
      pick_static_quantized_conv2d_local_wg_size,
      // Inputs and Outputs
      {{packed_int8_output, vkapi::kWrite},
       {{packed_int8_input_im2col,
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

void add_conv2d_q8ta_q8csw_q8to_node(
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
  // Check if the input/output layout is 4W4C (optimized path)
  const utils::GPUMemoryLayout inp_layout =
      graph.estimate_memory_layout_of(packed_int8_input);
  const utils::GPUMemoryLayout outp_layout =
      graph.estimate_memory_layout_of(packed_int8_output);

  const bool use_optimized_shader =
      (inp_layout == utils::kPackedInt8_4W4C &&
       outp_layout == utils::kPackedInt8_4W4C);

  if (use_optimized_shader) {
    add_conv2d_q8ta_q8csw_q8to_4w4c_node(
        graph,
        packed_int8_input,
        packed_int8_input_im2col,
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
  } else {
    add_q8ta_conv2d_node(
        graph,
        packed_int8_input,
        packed_int8_input_im2col,
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

// Implementation for statically quantized conv2d, which expects input, weight,
// and output tensors to all have packed int8 dtype/memory layout.
void static_quantized_conv2d_impl(
    ComputeGraph& graph,
    const QuantizationConfig& input_quant_config,
    const QuantizationConfig& weight_quant_config,
    const QuantizationConfig& output_quant_config,
    const ValueRef packed_int8_input,
    const ValueRef input_scale,
    const ValueRef input_zp,
    const ValueRef weight_data,
    const ValueRef weight_sums_data,
    const ValueRef weight_scales_data,
    const ValueRef output_scale,
    const ValueRef output_zp,
    const ValueRef bias_data,
    const ValueRef kernel_size,
    const ValueRef stride,
    const ValueRef padding,
    const ValueRef dilation,
    const ValueRef groups,
    const ValueRef packed_int8_output) {
  // Currently, only certain quantization configs are supported
  VK_CHECK_COND(input_quant_config.granularity == kPerTensor);
  VK_CHECK_COND(input_quant_config.nbits == 8);

  VK_CHECK_COND(weight_quant_config.granularity == kPerChannel);
  VK_CHECK_COND(weight_quant_config.nbits == 8);
  VK_CHECK_COND(weight_quant_config.is_symmetric);

  VK_CHECK_COND(output_quant_config.granularity == kPerTensor);
  VK_CHECK_COND(output_quant_config.nbits == 8);

  // Check for depthwise conv
  const int64_t groups_val = graph.extract_scalar<int64_t>(groups);
  const int64_t in_channels = graph.size_at<int64_t>(-3, packed_int8_input);

  // Depthwise convs have a specialized implementation, since the regular conv
  // implementations requires that the number of input and output channels per
  // groups is a multiple of 4. This is so that all values that are part of the
  // same 4Wx4C block have the same group index.
  const bool is_depthwise = (groups_val == in_channels);

  // Check if input/output layouts are 4W4C (optimized im2col path)
  const utils::GPUMemoryLayout inp_layout =
      graph.estimate_memory_layout_of(packed_int8_input);
  const utils::GPUMemoryLayout outp_layout =
      graph.estimate_memory_layout_of(packed_int8_output);
  const bool is_optimized_layout =
      (inp_layout == utils::kPackedInt8_4W4C &&
       outp_layout == utils::kPackedInt8_4W4C);

  // Only use im2col path for 4W4C layouts
  const bool use_im2col =
      is_optimized_layout && should_use_im2col(&graph, kernel_size, groups);
  // For pointwise convolution with stride = 1, padding = 0, dilation = 1, the
  // input tensor is already equivalent to its im2col representation. In this
  // case we can skip the im2col procedure and pass in the input image to the
  // convolution_as_matmul implementation directly.
  const bool is_optimizable_pw =
      is_s1p0d1_pointwise(&graph, kernel_size, stride, padding, dilation);

  ValueRef packed_weight;
  if (is_depthwise) {
    packed_weight = prepack_quantized_conv2d_dw_weight(
        graph, weight_quant_config, weight_data, kernel_size);
  } else if (use_im2col) {
    packed_weight = prepack_quantized_linear_weight(
        graph, weight_quant_config, weight_data);
  } else {
    packed_weight = prepack_quantized_conv2d_weight(
        graph,
        weight_quant_config,
        weight_data,
        packed_int8_input,
        packed_int8_output,
        groups,
        kernel_size);
  }

  ValueRef packed_weight_sums = prepack_standard(
      graph, weight_sums_data, utils::kBuffer, utils::kWidthPacked);

  ValueRef packed_weight_scales = prepack_standard(
      graph, weight_scales_data, utils::kBuffer, utils::kWidthPacked);

  // See quantized_conv2d_impl for why this is needed
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

  // Depthwise conv path
  if (is_depthwise) {
    add_conv2d_dw_q8ta_q8csw_q8to_4w4c_node(
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
    return;
  }

  std::vector<int64_t> input_im2col_sizes =
      calculate_packed_int8_input_im2col_sizes(
          &graph, packed_int8_input, packed_int8_output, kernel_size, groups);

  ValueRef packed_int8_input_im2col = packed_int8_input;
  if (use_im2col && !is_optimizable_pw) {
    TmpTensor packed_int8_input_im2col_tensor(
        &graph,
        input_im2col_sizes,
        vkapi::kInt8x4,
        graph.storage_type_of(packed_int8_input),
        utils::kPackedInt8_4W4C);

    packed_int8_input_im2col = packed_int8_input_im2col_tensor.vref;

    add_input_im2col_packed_int8_node(
        graph,
        packed_int8_input,
        input_scale,
        input_zp,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        packed_int8_output,
        packed_int8_input_im2col);
  }

  add_conv2d_q8ta_q8csw_q8to_node(
      graph,
      packed_int8_input,
      packed_int8_input_im2col,
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

void conv2d_q8ta_q8csw_q8to(
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

  QuantizationConfig input_quant_config(8, kPerTensor, {});
  QuantizationConfig weight_quant_config(8, kPerChannel, {});
  QuantizationConfig output_quant_config(8, kPerTensor, {});

  static_quantized_conv2d_impl(
      graph,
      input_quant_config,
      weight_quant_config,
      output_quant_config,
      packed_int8_input,
      input_scale,
      input_zp,
      weight_data,
      weight_sums_data,
      weight_scales_data,
      output_scale,
      output_zp,
      bias_data,
      kernel_size,
      stride,
      padding,
      dilation,
      groups,
      packed_int8_output);
}

//
// Test operators
//

void test_conv2d_q8ta_q8csw_q8to(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef fp_input = args.at(idx++);
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
  const ValueRef layout_int = args.at(idx++);
  const ValueRef fp_output = args.at(idx++);

  // Extract the layout parameter and cast to GPUMemoryLayout
  int32_t layout_value = graph.extract_scalar<int32_t>(layout_int);
  utils::GPUMemoryLayout layout =
      static_cast<utils::GPUMemoryLayout>(layout_value);

  TmpTensor packed_int8_input(
      &graph, graph.sizes_of(fp_input), vkapi::kInt8x4, utils::kBuffer, layout);

  TmpTensor packed_int8_output(
      &graph,
      graph.sizes_of(fp_output),
      vkapi::kInt8x4,
      utils::kBuffer,
      layout);

  add_q8ta_quantize_node(
      graph, fp_input, input_scale, input_zp, packed_int8_input);

  std::vector<ValueRef> conv2d_args = {
      packed_int8_input,
      input_scale,
      input_zp,
      weight_data,
      weight_sums_data,
      weight_scales_data,
      output_scale,
      output_zp,
      bias_data,
      kernel_size,
      stride,
      padding,
      dilation,
      groups,
      packed_int8_output};

  conv2d_q8ta_q8csw_q8to(graph, conv2d_args);

  add_q8ta_dequantize_node(
      graph, packed_int8_output, output_scale, output_zp, fp_output);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.conv2d_q8ta_q8csw.default, conv2d_q8ta_q8csw);
  VK_REGISTER_OP(et_vk.conv2d_q8csw.default, conv2d_q8csw);
  VK_REGISTER_OP(
      etvk.test_conv2d_q8ta_q8csw_q8to.default, test_conv2d_q8ta_q8csw_q8to);
  VK_REGISTER_OP(et_vk.conv2d_q8ta_q8csw_q8to.default, conv2d_q8ta_q8csw_q8to);
  VK_REGISTER_OP(
      et_vk.conv2d_q8ta_q8csw_q8to_dw.default, conv2d_q8ta_q8csw_q8to);
}

} // namespace vkcompute
