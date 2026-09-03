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
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/KernelUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

#include <algorithm>
#include <limits>

namespace vkcompute {

Q8taConv2dStreamPlan make_q8ta_conv2d_stream_plan(
    const int64_t batch,
    const int64_t flattened_kernel_size,
    const int64_t out_height,
    const int64_t out_width,
    const int64_t scratch_budget_bytes) {
  Q8taConv2dStreamPlan plan{};
  if (batch <= 0 || flattened_kernel_size <= 0 || out_height <= 0 ||
      out_width <= 0 || scratch_budget_bytes <= 0) {
    return plan;
  }

  constexpr int64_t kAlignment = 4;
  if (out_width >
      std::numeric_limits<int64_t>::max() - (kAlignment - 1)) {
    return plan;
  }
  plan.aligned_out_width =
      (out_width + kAlignment - 1) / kAlignment * kAlignment;
  if (flattened_kernel_size >
      std::numeric_limits<int64_t>::max() / plan.aligned_out_width) {
    return plan;
  }
  const int64_t bytes_per_row =
      flattened_kernel_size * plan.aligned_out_width;
  if (bytes_per_row > scratch_budget_bytes ||
      batch > std::numeric_limits<int64_t>::max() / out_height) {
    return plan;
  }

  const int64_t total_rows = batch * out_height;
  if (total_rows > std::numeric_limits<int32_t>::max()) {
    return plan;
  }
  const int64_t max_rows_per_tile =
      std::min(total_rows, scratch_budget_bytes / bytes_per_row);
  plan.num_tiles = total_rows / max_rows_per_tile +
      static_cast<int64_t>(total_rows % max_rows_per_tile != 0);
  plan.rows_per_tile = total_rows / plan.num_tiles +
      static_cast<int64_t>(total_rows % plan.num_tiles != 0);
  plan.scratch_bytes = plan.rows_per_tile * bytes_per_row;
  plan.feasible = true;
  return plan;
}

//
// Shader dispatch utilities
//

GlobalWorkGrid pick_q8ta_im2col_full_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  VK_CHECK_COND(graph != nullptr);
  const ValueRef output = args.at(0).refs.at(0);
  const uint32_t K4 = utils::div_up_4(graph->size_at<uint32_t>(-3, output));
  const uint32_t H = graph->size_at<uint32_t>(-2, output);
  const uint32_t W4 = utils::div_up_4(graph->size_at<uint32_t>(-1, output));
  const uint32_t N = graph->size_at<uint32_t>(-4, output);
  return graph->create_linear_gwg(utils::safe_downcast<uint32_t>(
      static_cast<uint64_t>(K4) * H * W4 * N));
}

GlobalWorkGrid pick_q8ta_im2col_streaming_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  VK_CHECK_COND(graph != nullptr);
  const ValueRef im2col_output = args.at(0).refs.at(0);
  const uint32_t K = graph->size_at<uint32_t>(-3, im2col_output);
  const uint32_t rows_per_tile =
      graph->size_at<uint32_t>(-2, im2col_output);
  const uint32_t W = graph->size_at<uint32_t>(-1, im2col_output);

  const ValueRef input = resize_args.at(0);
  const ValueRef kernel_size = resize_args.at(1);
  const ValueRef stride = resize_args.at(2);
  const ValueRef padding = resize_args.at(3);
  const ValueRef dilation = resize_args.at(4);
  const int64_t row_offset =
      graph->extract_scalar<int64_t>(resize_args.at(6));

  const std::vector<int64_t> input_sizes = graph->sizes_of(input);
  const int64_t batch = utils::val_at(-4, input_sizes);
  const std::vector<int64_t> out_hw = calc_out_sizes_hw(
      *graph,
      input_sizes,
      kernel_size,
      /*kernel_size_only=*/true,
      {stride, padding, dilation, dilation},
      /*transposed=*/false);
  const int64_t total_rows = batch * out_hw.at(0);
  if (row_offset >= total_rows) {
    return graph->create_linear_gwg(0u);
  }
  const uint32_t live_rows = utils::safe_downcast<uint32_t>(
      std::min<int64_t>(rows_per_tile, total_rows - row_offset));

  const uint32_t K4 = utils::div_up_4(K);
  const uint32_t W4 = utils::div_up_4(W);

  // Each thread handles one 4x4 block in the output
  return graph->create_linear_gwg(
      utils::safe_downcast<uint32_t>(
          static_cast<uint64_t>(K4) * W4 * live_rows));
}

LocalWorkGroup pick_q8ta_im2col_lwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const GlobalWorkGrid& gwg,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)graph;
  (void)shader;
  (void)args;
  (void)resize_args;
  (void)gwg;

  return LocalWorkGroup(64u, 1u, 1u);
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
  const int64_t batch = utils::val_at(-4, in_sizes);
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

  return {batch, K, H, W};
}

//
// Resize
//

// resize_args = { input, kernel_size, stride, padding, dilation, groups,
//                 row_offset }
//
// The streaming scratch tensor is [1, K, rows_per_tile, align_up_4(W_out)].
// K and rows_per_tile are fixed; only W_out tracks the current input shape.
void resize_q8ta_im2col_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  VK_CHECK_COND(graph != nullptr);
  const ValueRef im2col_out = args.at(0).refs.at(0);
  const ValueRef in = resize_args.at(0);
  const ValueRef kernel_size = resize_args.at(1);
  const ValueRef stride = resize_args.at(2);
  const ValueRef padding = resize_args.at(3);
  const ValueRef dilation = resize_args.at(4);
  const ValueRef groups = resize_args.at(5);

  const std::vector<int64_t> in_sizes = graph->sizes_of(in);
  // Conv output width from the current input.
  const std::vector<int64_t> out_hw = calc_out_sizes_hw(
      *graph,
      in_sizes,
      kernel_size,
      /*kernel_size_only=*/true,
      {stride, padding, dilation, dilation},
      /*transposed=*/false);
  const int64_t out_width = out_hw.at(1);

  // K (flattened conv window) is shape-independent — recompute from channels +
  // kernel exactly as calculate_q8ta_im2col_sizes does.
  const int64_t in_channels = utils::val_at(-3, in_sizes);
  const int64_t groups_val = graph->extract_scalar<int64_t>(groups);
  const int64_t in_channels_per_group = in_channels / groups_val;
  const auto kernel_size_list = graph->get_int_list(kernel_size);
  const int64_t flattened_kernel_len = utils::align_up_4(
      in_channels_per_group * kernel_size_list->at(0) *
      kernel_size_list->at(1));
  const int64_t K = flattened_kernel_len * groups_val;
  const int64_t W = utils::align_up_4(out_width);

  const int64_t rows_per_tile = graph->size_at<int64_t>(-2, im2col_out);

  graph->virtual_resize(im2col_out, {1, K, rows_per_tile, W});
}

void resize_q8ta_im2col_full_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  VK_CHECK_COND(graph != nullptr);
  const ValueRef im2col_out = args.at(0).refs.at(0);
  const ValueRef in = resize_args.at(0);
  const ValueRef kernel_size = resize_args.at(1);
  const ValueRef stride = resize_args.at(2);
  const ValueRef padding = resize_args.at(3);
  const ValueRef dilation = resize_args.at(4);
  const ValueRef groups = resize_args.at(5);

  const std::vector<int64_t> in_sizes = graph->sizes_of(in);
  const std::vector<int64_t> out_hw = calc_out_sizes_hw(
      *graph,
      in_sizes,
      kernel_size,
      /*kernel_size_only=*/true,
      {stride, padding, dilation, dilation},
      /*transposed=*/false);
  const int64_t in_channels = utils::val_at(-3, in_sizes);
  const int64_t groups_val = graph->extract_scalar<int64_t>(groups);
  const int64_t in_channels_per_group = in_channels / groups_val;
  const auto kernel_size_list = graph->get_int_list(kernel_size);
  const int64_t flattened_kernel_len = utils::align_up_4(
      in_channels_per_group * kernel_size_list->at(0) *
      kernel_size_list->at(1));
  const int64_t K = flattened_kernel_len * groups_val;

  graph->virtual_resize(
      im2col_out,
      {utils::val_at(-4, in_sizes),
       K,
       out_hw.at(0),
       utils::align_up_4(out_hw.at(1))});
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
    const int32_t zp,
    const Q8taIm2ColMode mode,
    const int32_t stream_row_offset) {
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

  // The implementation requires that input channels per group is a multiple of
  // 4
  VK_CHECK_COND(conv_params.in_channels_per_group % 4 == 0);

  const bool is_streaming = mode == Q8taIm2ColMode::kStreaming;
  std::string kernel_name =
      is_streaming ? "q8ta_im2col_streaming" : "q8ta_im2col";

  vkapi::ParamsBindList param_buffers = {
      graph.buffer_meta_ubo(packed_int8_im2col),
      graph.buffer_meta_ubo(packed_int8_input),
      graph.create_params_buffer(conv_params)};

  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&zp, sizeof(zp)),
  };
  if (is_streaming) {
    push_constants.emplace_back(
        &stream_row_offset, sizeof(stream_row_offset));
  }

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

  std::vector<ValueRef> resize_args = {
      packed_int8_input, kernel_size, stride, padding, dilation, groups};
  if (is_streaming) {
    resize_args.push_back(graph.add_scalar<int64_t>(stream_row_offset));
  }

  const auto pick_gwg = is_streaming ? pick_q8ta_im2col_streaming_gwg
                                     : pick_q8ta_im2col_full_gwg;
  const auto resize_fn = is_streaming ? resize_q8ta_im2col_node
                                      : resize_q8ta_im2col_full_node;

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      pick_gwg,
      pick_q8ta_im2col_lwg,
      // Inputs and Outputs
      {{packed_int8_im2col, vkapi::kWrite}, {packed_int8_input, vkapi::kRead}},
      // Shader params buffers
      param_buffers,
      // Push Constants
      push_constants,
      // Specialization Constants
      spec_constants,
      resize_args,
      resize_fn));
}

//
// High level operator impl
//

void q8ta_conv2d_im2col_with_kernel(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args,
    const Q8taConv2dPwKernel kernel) {
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
  const ValueRef activation = args.at(idx++);
  const ValueRef packed_int8_output = args.at(idx++);

  const std::vector<int64_t> full_im2col_sizes = calculate_q8ta_im2col_sizes(
      &graph, packed_int8_input, packed_int8_output, kernel_size, groups);
  const Q8taConv2dStreamPlan stream_plan = make_q8ta_conv2d_stream_plan(
      full_im2col_sizes.at(0),
      full_im2col_sizes.at(1),
      full_im2col_sizes.at(2),
      full_im2col_sizes.at(3),
      kQ8taConv2dIm2ColScratchBudgetBytes);
  if (!stream_plan.feasible) {
    q8ta_conv2d_general(graph, args);
    return;
  }

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

  uint32_t activation_type_val = static_cast<uint32_t>(
      activation_type_from_string(graph.extract_string(activation)));

  const bool is_streaming = stream_plan.num_tiles > 1;
  const std::vector<int64_t> im2col_sizes = is_streaming
      ? std::vector<int64_t>{
            1,
            full_im2col_sizes.at(1),
            stream_plan.rows_per_tile,
            stream_plan.aligned_out_width}
      : full_im2col_sizes;

  TmpTensor packed_int8_im2col(
      &graph,
      im2col_sizes,
      vkapi::kInt8x4,
      utils::kBuffer,
      utils::kPackedInt8_4W4C);

  int32_t zp = graph.extract_scalar<int32_t>(input_zp);
  const int32_t groups_val = graph.extract_scalar<int32_t>(groups);

  if (!is_streaming) {
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
        zp,
        Q8taIm2ColMode::kFull,
        /*stream_row_offset=*/0);

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
        activation_type_val,
        packed_int8_output,
        groups_val,
        Q8taConv2dPwMode::kIm2Col,
        kernel,
        packed_int8_input,
        kernel_size,
        stride,
        padding,
        dilation);
    return;
  }

  // Reuse one fixed-size scratch buffer across all row tiles. Interleaved
  // write/read dispatches insert the barrier before the next tile overwrites
  // it.
  for (int64_t tile = 0; tile < stream_plan.num_tiles; ++tile) {
    const int32_t row_offset = utils::safe_downcast<int32_t>(
        tile * stream_plan.rows_per_tile);

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
        zp,
        Q8taIm2ColMode::kStreaming,
        row_offset);

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
        activation_type_val,
        packed_int8_output,
        groups_val,
        Q8taConv2dPwMode::kStreamingIm2Col,
        kernel,
        packed_int8_input,
        kernel_size,
        stride,
        padding,
        dilation,
        row_offset);
  }
}

void q8ta_conv2d_im2col(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  q8ta_conv2d_im2col_with_kernel(
      graph, args, Q8taConv2dPwKernel::kAuto);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.q8ta_conv2d_im2col.default, q8ta_conv2d_im2col);
}

} // namespace vkcompute
