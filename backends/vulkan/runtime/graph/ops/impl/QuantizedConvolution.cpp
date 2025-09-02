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
  // M -> number of elements in 2D output plane
  const int64_t M = out_height * out_width * batches;

  return {M, K};
}

utils::uvec3 im2col_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef input = args.at(1).refs.at(0);
  const ValueRef output = resize_args.at(0);
  const ValueRef kernel_size = resize_args.at(1);
  const ValueRef groups = resize_args.at(2);

  std::vector<int64_t> im2col_sizes =
      calculate_input_im2col_sizes(graph, input, output, kernel_size, groups);
  const uint32_t K = utils::safe_downcast<uint32_t>(im2col_sizes[1]);
  const uint32_t M = utils::safe_downcast<uint32_t>(im2col_sizes[0]);

  // 1 output tile is 4x4 elements
  const uint32_t K4 = utils::div_up(K, 4u);
  const uint32_t M4 = utils::div_up(M, 4u);

  return {K4, M4, 1};
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

void add_input_im2col_node(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  // Extract arguments
  int32_t idx = 0;
  const ValueRef input = args.at(idx++);
  const ValueRef kernel_size = args.at(idx++);
  const ValueRef stride = args.at(idx++);
  const ValueRef padding = args.at(idx++);
  const ValueRef dilation = args.at(idx++);
  const ValueRef groups = args.at(idx++);
  const ValueRef output = args.at(idx++);
  const ValueRef im2col_matrix = args.at(idx++);

  Conv2DParams conv_params = create_conv2d_params(
      graph, input, output, kernel_size, stride, padding, dilation, groups);

  // Get shader for quantized conv2d linear tiled
  std::string kernel_name = "im2col";
  add_storage_type_suffix(kernel_name, graph.storage_type_of(im2col_matrix));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(input));
  add_dtype_suffix(kernel_name, graph.dtype_of(output));

  vkapi::ShaderInfo shader = VK_KERNEL_FROM_STR(kernel_name);

  vkapi::ParamsBindList param_buffers = {
      graph.sizes_ubo(im2col_matrix),
      graph.sizes_ubo(input),
      graph.sizes_ubo(output),
      graph.create_params_buffer(conv_params)};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      im2col_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{im2col_matrix, vkapi::kWrite}, {input, vkapi::kRead}},
      // Shader params buffers
      param_buffers,
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize args
      {output, kernel_size, groups},
      // Resizing Logic
      nullptr));
}

void add_quantize_and_pack_im2col_node(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  // Extract arguments
  int32_t idx = 0;
  const ValueRef input = args.at(idx++);
  const ValueRef input_scale = args.at(idx++);
  const ValueRef input_zp = args.at(idx++);
  const ValueRef kernel_size = args.at(idx++);
  const ValueRef stride = args.at(idx++);
  const ValueRef padding = args.at(idx++);
  const ValueRef dilation = args.at(idx++);
  const ValueRef groups = args.at(idx++);
  const ValueRef output = args.at(idx++);
  const ValueRef quantized_im2col_matrix = args.at(idx++);

  Conv2DParams conv_params = create_conv2d_params(
      graph, input, output, kernel_size, stride, padding, dilation, groups);

  float inv_scale = 1.0f / graph.extract_scalar<float>(input_scale);
  int32_t zp = graph.extract_scalar<int32_t>(input_zp);

  // Get shader for quantized conv2d linear tiled
  std::string kernel_name = "quantize_and_pack_im2col";
  add_storage_type_suffix(
      kernel_name, graph.storage_type_of(quantized_im2col_matrix));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(input));
  add_dtype_suffix(kernel_name, graph.dtype_of(output));

  vkapi::ShaderInfo shader = VK_KERNEL_FROM_STR(kernel_name);

  vkapi::ParamsBindList param_buffers = {
      graph.sizes_ubo(quantized_im2col_matrix),
      graph.sizes_ubo(input),
      graph.sizes_ubo(output),
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
      {{quantized_im2col_matrix, vkapi::kWrite}, {input, vkapi::kRead}},
      // Shader params buffers
      param_buffers,
      // Push Constants
      push_constants,
      // Specialization Constants
      {},
      // Resize args
      {output, kernel_size, groups},
      // Resizing Logic
      nullptr));
}

void add_conv2d_q8csw_linear_tiled_node(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  // Extract arguments
  int32_t idx = 0;
  const ValueRef input_im2col = args.at(idx++);
  const ValueRef input = args.at(idx++);
  const ValueRef packed_weight = args.at(idx++);
  const ValueRef packed_weight_scales = args.at(idx++);
  const ValueRef bias = args.at(idx++);
  const ValueRef packed_bias = args.at(idx++);
  const ValueRef kernel_size = args.at(idx++);
  const ValueRef stride = args.at(idx++);
  const ValueRef padding = args.at(idx++);
  const ValueRef dilation = args.at(idx++);
  const ValueRef groups = args.at(idx++);
  const ValueRef output = args.at(idx++);
  const ValueRef original_weight = args.at(idx++); // For resize args

  Conv2DParams conv_params = create_conv2d_params(
      graph, input, output, kernel_size, stride, padding, dilation, groups);

  // One limitation of the current implementation is that for grouped convs,
  // the number of output channels per group must be a multiple of 4. One loaded
  // 4x4 weight tile must all belong to the same group.
  if (conv_params.groups > 1) {
    VK_CHECK_COND(conv_params.out_channels_per_group % 4 == 0);
  }

  std::string kernel_name = "conv2d_q8csw_linear_tiled";
  add_storage_type_suffix(kernel_name, graph.storage_type_of(output));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(input_im2col));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(packed_weight));
  add_dtype_suffix(kernel_name, graph.dtype_of(output));
  vkapi::ShaderInfo shader = VK_KERNEL_FROM_STR(kernel_name);

  vkapi::ParamsBindList param_buffers = {
      graph.sizes_ubo(output),
      graph.sizes_ubo(input),
      graph.create_params_buffer(conv_params)};

  uint32_t apply_bias = 1;
  if (graph.val_is_none(bias)) {
    apply_bias = 0;
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      col2im_global_wg_size,
      quantized_linear_local_wg_size,
      // Inputs and Outputs
      {{output, vkapi::kWrite},
       {{input_im2col, packed_weight, packed_weight_scales, packed_bias},
        vkapi::kRead}},
      // Shader params buffers
      param_buffers,
      // Push Constants
      {},
      // Specialization Constants
      {apply_bias},
      // Resize args
      {original_weight},
      // Resizing Logic
      nullptr));
}

void add_conv2d_q8ta_q8csw_linear_tiled_node(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  // Extract arguments
  int32_t idx = 0;
  const ValueRef quantized_input_im2col = args.at(idx++);
  const ValueRef input = args.at(idx++);
  const ValueRef input_scale = args.at(idx++);
  const ValueRef input_zp = args.at(idx++);
  const ValueRef packed_weight = args.at(idx++);
  const ValueRef packed_weight_sums = args.at(idx++);
  const ValueRef packed_weight_scales = args.at(idx++);
  const ValueRef bias = args.at(idx++);
  const ValueRef packed_bias = args.at(idx++);
  const ValueRef kernel_size = args.at(idx++);
  const ValueRef stride = args.at(idx++);
  const ValueRef padding = args.at(idx++);
  const ValueRef dilation = args.at(idx++);
  const ValueRef groups = args.at(idx++);
  const ValueRef output = args.at(idx++);
  const ValueRef original_weight = args.at(idx++); // For resize args

  Conv2DParams conv_params = create_conv2d_params(
      graph, input, output, kernel_size, stride, padding, dilation, groups);

  // One limitation of the current implementation is that for grouped convs,
  // the number of output channels per group must be a multiple of 4. One loaded
  // 4x4 weight tile must all belong to the same group.
  if (conv_params.groups > 1) {
    VK_CHECK_COND(conv_params.out_channels_per_group % 4 == 0);
  }

  float scale = graph.extract_scalar<float>(input_scale);
  int32_t zp = graph.extract_scalar<int32_t>(input_zp);

  std::string kernel_name = "conv2d_q8ta_q8csw_linear_tiled";
  add_storage_type_suffix(kernel_name, graph.storage_type_of(output));
  add_storage_type_suffix(
      kernel_name, graph.storage_type_of(quantized_input_im2col));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(packed_weight));
  add_dtype_suffix(kernel_name, graph.dtype_of(output));
  vkapi::ShaderInfo shader = VK_KERNEL_FROM_STR(kernel_name);

  vkapi::ParamsBindList param_buffers = {
      graph.sizes_ubo(output),
      graph.sizes_ubo(input),
      graph.create_params_buffer(conv_params)};

  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&scale, sizeof(scale)),
      PushConstantDataInfo(&zp, sizeof(zp)),
  };

  uint32_t apply_bias = 1;
  if (graph.val_is_none(bias)) {
    apply_bias = 0;
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      col2im_global_wg_size,
      quantized_linear_local_wg_size,
      // Inputs and Outputs
      {{output, vkapi::kWrite},
       {{quantized_input_im2col,
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
      {original_weight},
      // Resizing Logic
      nullptr));
}

/*
 * Computes weight only quantized conv2d with the conv2d_q8csw_linear_tiled
 * shader. The input image will first be converted to matrix form using the
 * im2col procedure. The convolution is performed via matrix multiplication, but
 * the output is written directly as image format which circumvents the need for
 * a separate step to convert the output matrix back to image format. This
 * implementation will be used when accelerated int8 dot product is not
 * available on a particular device, in which case there is no benefit from
 * quantizing the input tensor.
 */
void conv2d_q8csw_linear_tiled_impl(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef input = args.at(idx++);
  const ValueRef input_scale = args.at(idx++);
  (void)input_scale;
  const ValueRef input_zp = args.at(idx++);
  (void)input_zp;
  const ValueRef weight = args.at(idx++);
  const ValueRef weight_sums = args.at(idx++);
  (void)weight_sums;
  const ValueRef weight_scales = args.at(idx++);
  const ValueRef bias = args.at(idx++);
  const ValueRef kernel_size = args.at(idx++);
  const ValueRef stride = args.at(idx++);
  const ValueRef padding = args.at(idx++);
  const ValueRef dilation = args.at(idx++);
  const ValueRef groups = args.at(idx++);
  const ValueRef orig_OC = args.at(idx++);
  (void)orig_OC;
  const ValueRef output = args.at(idx++);

  const ValueRef packed_weight = prepack_quantized_linear_weight(graph, weight);
  ValueRef packed_weight_scales = prepack_standard(
      graph, weight_scales, utils::kBuffer, utils::kWidthPacked);

  // Create a dummy tensor to fill the binding slot of the bias tensor if it is
  // not provided. This helps simplify dispatch logic and makes it so that
  // fewer shdaer variants need to be generated.
  TmpTensor dummy_bias(
      &graph, {}, graph.dtype_of(output), utils::kBuffer, utils::kWidthPacked);

  ValueRef packed_bias = dummy_bias.vref;
  if (!graph.val_is_none(bias)) {
    packed_bias =
        prepack_standard(graph, bias, utils::kBuffer, utils::kWidthPacked);
  }

  std::vector<int64_t> input_im2col_sizes =
      calculate_input_im2col_sizes(&graph, input, output, kernel_size, groups);

  TmpTensor input_im2col_matrix(
      &graph,
      input_im2col_sizes,
      vkapi::kFloat,
      utils::kBuffer,
      utils::kWidthPacked);

  std::vector<ValueRef> im2col_args = {
      input,
      kernel_size,
      stride,
      padding,
      dilation,
      groups,
      output,
      input_im2col_matrix};

  add_input_im2col_node(graph, im2col_args);

  std::vector<ValueRef> conv2d_linear_args = {
      input_im2col_matrix,
      input,
      packed_weight,
      packed_weight_scales,
      bias,
      packed_bias,
      kernel_size,
      stride,
      padding,
      dilation,
      groups,
      output,
      weight};

  add_conv2d_q8csw_linear_tiled_node(graph, conv2d_linear_args);
}

void conv2d_q8ta_q8csw_linear_tiled_impl(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef input = args.at(idx++);
  const ValueRef input_scale = args.at(idx++);
  const ValueRef input_zp = args.at(idx++);
  const ValueRef weight = args.at(idx++);
  const ValueRef weight_sums = args.at(idx++);
  const ValueRef weight_scales = args.at(idx++);
  const ValueRef bias = args.at(idx++);
  const ValueRef kernel_size = args.at(idx++);
  const ValueRef stride = args.at(idx++);
  const ValueRef padding = args.at(idx++);
  const ValueRef dilation = args.at(idx++);
  const ValueRef groups = args.at(idx++);
  const ValueRef orig_OC = args.at(idx++);
  (void)orig_OC;
  const ValueRef output = args.at(idx++);

  const ValueRef packed_weight = prepack_quantized_linear_weight(graph, weight);
  ValueRef packed_weight_scales = prepack_standard(
      graph, weight_scales, utils::kBuffer, utils::kWidthPacked);
  ValueRef packed_weight_sums =
      prepack_standard(graph, weight_sums, utils::kBuffer, utils::kWidthPacked);

  // Create a dummy tensor to fill the binding slot of the bias tensor if it is
  // not provided. This helps simplify dispatch logic and makes it so that
  // fewer shdaer variants need to be generated.
  TmpTensor dummy_bias(
      &graph, {}, graph.dtype_of(output), utils::kBuffer, utils::kWidthPacked);

  ValueRef packed_bias = dummy_bias.vref;
  if (!graph.val_is_none(bias)) {
    packed_bias =
        prepack_standard(graph, bias, utils::kBuffer, utils::kWidthPacked);
  }

  std::vector<int64_t> input_im2col_sizes =
      calculate_input_im2col_sizes(&graph, input, output, kernel_size, groups);

  const int64_t num_blocks_M = utils::div_up_4(input_im2col_sizes.at(0));
  const int64_t num_blocks_K = utils::div_up_4(input_im2col_sizes.at(1));

  TmpTensor quantized_input_im2col_matrix(
      &graph,
      {num_blocks_M, num_blocks_K * 4},
      vkapi::kInt,
      utils::kBuffer,
      utils::kWidthPacked);

  std::vector<ValueRef> quantize_and_pack_im2col_args = {
      input,
      input_scale,
      input_zp,
      kernel_size,
      stride,
      padding,
      dilation,
      groups,
      output,
      quantized_input_im2col_matrix};

  add_quantize_and_pack_im2col_node(graph, quantize_and_pack_im2col_args);

  std::vector<ValueRef> conv2d_linear_args = {
      quantized_input_im2col_matrix,
      input,
      input_scale,
      input_zp,
      packed_weight,
      packed_weight_sums,
      packed_weight_scales,
      bias,
      packed_bias,
      kernel_size,
      stride,
      padding,
      dilation,
      groups,
      output,
      weight};

  add_conv2d_q8ta_q8csw_linear_tiled_node(graph, conv2d_linear_args);
}

void conv2d_q8ta_q8csw(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  // If accelerated int8 dot product is available, quantize the input tensor
  // to allow for faster arithmetic throughput.
  if (graph.can_use_int8_dot_product()) {
    conv2d_q8ta_q8csw_linear_tiled_impl(graph, args);
  }
  // Otherwise, dequantize the weight tensor and do math in fp32.
  else {
    conv2d_q8csw_linear_tiled_impl(graph, args);
  }
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.conv2d_q8ta_q8csw.default, conv2d_q8ta_q8csw);
  VK_REGISTER_OP(
      et_vk.conv2d_q8ta_q8csw.conv2d_q8csw_linear_tiled,
      conv2d_q8csw_linear_tiled_impl);
  VK_REGISTER_OP(
      et_vk.conv2d_q8ta_q8csw.conv2d_q8ta_q8csw_linear_tiled,
      conv2d_q8ta_q8csw_linear_tiled_impl);
}

} // namespace vkcompute
