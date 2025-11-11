/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/QuantizeDequantize.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

//
// General utilities
//

bool is_gemv(ComputeGraph* graph, const ValueRef& fp_input) {
  return graph->size_at<uint32_t>(-2, fp_input) == 1;
}

//
// Dispatch utilities (Linear)
//

std::tuple<int64_t, int64_t> get_quantized_input_num_blocks(
    ComputeGraph& graph,
    const ValueRef input) {
  std::vector<int64_t> input_sizes = graph.sizes_of(input);
  const int64_t ndim = graph.dim_of(input);

  const int64_t M = input_sizes.at(ndim - 2);
  const int64_t K = input_sizes.at(ndim - 1);

  const int64_t num_blocks_M = utils::div_up(M, int64_t(4));
  const int64_t num_blocks_K = utils::div_up(K, int64_t(4));

  return std::make_tuple(num_blocks_M, num_blocks_K);
}

utils::uvec3 quantize_and_pack_4h4w_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef input = args.at(1).refs.at(0);
  int64_t num_blocks_M, num_blocks_K;
  std::tie(num_blocks_M, num_blocks_K) =
      get_quantized_input_num_blocks(*graph, input);

  return {
      utils::safe_downcast<uint32_t>(num_blocks_K),
      utils::safe_downcast<uint32_t>(num_blocks_M),
      1u};
}

vkapi::ShaderInfo pick_quantize_and_pack_4h4w_with_group_sums_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef packed_int_input = args.at(0).refs.at(0);
  const ValueRef fp_input = args.at(1).refs.at(0);
  const ValueRef group_size = resize_args.at(0);

  const int64_t group_size_val = graph->extract_scalar<int64_t>(group_size);

  std::string shader_name = "quantize_and_pack_4h4w_with_group_sums";
  if (group_size_val >= 128) {
    shader_name += "_o2w32";
  } else {
    shader_name += "_o4w16";
  }

  add_storage_type_suffix(
      shader_name, graph->storage_type_of(packed_int_input));
  add_storage_type_suffix(shader_name, graph->storage_type_of(fp_input));
  add_dtype_suffix(shader_name, graph->dtype_of(fp_input));

  return VK_KERNEL_FROM_STR(shader_name);
}

utils::uvec3 pick_quantize_and_pack_4h4w_with_group_sums_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef fp_input = args.at(1).refs.at(0);
  // For gemv cases, skip the quantize and pack input step in favor of computing
  // the quantized linear as a weight only quantized linear operation. The
  // rationale for this is that gemv is a memory bound operation and may not
  // necessarily benefit from quantizing the input and computing with integer
  // accumulation.
  if (is_gemv(graph, fp_input)) {
    return {0u, 0u, 0u};
  }

  const ValueRef group_size = resize_args.at(0);
  int64_t num_blocks_M, num_blocks_K;
  std::tie(num_blocks_M, num_blocks_K) =
      get_quantized_input_num_blocks(*graph, fp_input);

  const int64_t group_size_val = graph->extract_scalar<int64_t>(group_size);
  const int64_t blocks_per_group = group_size_val / 4;

  const int64_t num_groups = num_blocks_K / blocks_per_group;

  return {
      utils::safe_downcast<uint32_t>(num_groups),
      utils::safe_downcast<uint32_t>(num_blocks_M),
      1u};
}

utils::uvec3 pick_quantize_and_pack_4h4w_with_group_sums_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;

  const ValueRef fp_input = args.at(1).refs.at(0);
  // For gemv, skip the quantize input step since the quantized linear is
  // computed as a weight only quantized linear operation.
  if (is_gemv(graph, fp_input)) {
    return {1u, 1u, 1u};
  }

  uint32_t groups_per_wg = 2u;
  uint32_t workers_per_group = 32u;

  if (shader.kernel_name.find("o4w16") != std::string::npos) {
    groups_per_wg = 4u;
    workers_per_group = 16u;
  }

  return {groups_per_wg, 1u, workers_per_group};
}

//
// Dispatch logic (Linear)
//

void add_quantize_and_pack_4h4w_node(
    ComputeGraph& graph,
    const QuantizationConfig& input_quant_config,
    const ValueRef fp_input,
    const ValueRef packed_input_scale,
    const ValueRef packed_input_zp,
    const ValueRef input_scale_data,
    const ValueRef input_zp_data,
    const ValueRef packed_int_input,
    const ValueRef group_size) {
  // Only certain quantization types supported at the moment
  VK_CHECK_COND(input_quant_config.granularity == kPerTensor);

  int64_t num_blocks_M, num_blocks_K;
  std::tie(num_blocks_M, num_blocks_K) =
      get_quantized_input_num_blocks(graph, fp_input);

  float inv_scale = 1.0f / graph.extract_scalar<float>(input_scale_data);
  int32_t zp = graph.extract_scalar<int32_t>(input_zp_data);

  std::string shader_name = "quantize_and_pack_4h4w_per_tensor";
  add_storage_type_suffix(shader_name, graph.storage_type_of(packed_int_input));
  add_storage_type_suffix(shader_name, graph.storage_type_of(fp_input));
  add_dtype_suffix(shader_name, graph.dtype_of(fp_input));

  vkapi::ParamsBindList param_buffers = {graph.sizes_ubo(fp_input)};

  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&inv_scale, sizeof(inv_scale)),
      PushConstantDataInfo(&zp, sizeof(zp)),
  };

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(shader_name),
      quantize_and_pack_4h4w_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{packed_int_input, vkapi::kWrite}, {fp_input, vkapi::kRead}},
      // Shader params buffers
      param_buffers,
      // Push Constants
      push_constants,
      // Specialization Constants
      {},
      // Resize args
      {}));
}

void add_quantize_and_pack_4h4w_with_group_sums_node(
    ComputeGraph& graph,
    const QuantizationConfig& input_quant_config,
    const ValueRef fp_input,
    const ValueRef int_input_sums,
    const ValueRef packed_input_scales,
    const ValueRef packed_input_zps,
    const ValueRef packed_int_input,
    const ValueRef group_size) {
  // Only certain quantization types supported at the moment
  VK_CHECK_COND(input_quant_config.granularity == kPerChannel);

  int64_t num_blocks_M, num_blocks_K;
  std::tie(num_blocks_M, num_blocks_K) =
      get_quantized_input_num_blocks(graph, fp_input);

  vkapi::ParamsBindList param_buffers = {graph.sizes_ubo(fp_input)};

  const int32_t group_size_val = graph.extract_scalar<int32_t>(group_size);
  const int32_t blocks_per_group = utils::div_up(group_size_val, int32_t(4));

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_quantize_and_pack_4h4w_with_group_sums_shader,
      pick_quantize_and_pack_4h4w_with_group_sums_global_wg_size,
      pick_quantize_and_pack_4h4w_with_group_sums_local_wg_size,
      // Inputs and Outputs
      {{{packed_int_input, int_input_sums}, vkapi::kWrite},
       {{fp_input, packed_input_scales, packed_input_zps}, vkapi::kRead}},
      // Shader params buffers
      param_buffers,
      // Push Constants
      {},
      // Specialization Constants
      {blocks_per_group},
      // Resize args
      {group_size}));
}

//
// Dispatch utilities (Conv2d)
//

utils::uvec3 pick_quantize_and_pack_4w4c_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef fp_input = args.at(1).refs.at(0);

  const uint32_t W = graph->size_at<uint32_t>(-1, fp_input);
  const uint32_t H = graph->size_at<uint32_t>(-2, fp_input);
  const uint32_t C = graph->size_at<uint32_t>(-3, fp_input);

  const uint32_t W4 = utils::div_up_4(W);
  const uint32_t C4 = utils::div_up_4(C);

  return {W4, H, C4};
}

utils::uvec3 pick_unpack_4w4c_and_dequantize_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef fp_output = args.at(0).refs.at(0);

  const uint32_t W = graph->size_at<uint32_t>(-1, fp_output);
  const uint32_t H = graph->size_at<uint32_t>(-2, fp_output);
  const uint32_t C = graph->size_at<uint32_t>(-3, fp_output);

  const uint32_t W4 = utils::div_up_4(W);
  const uint32_t C4 = utils::div_up_4(C);

  return {W4, H, C4};
}

//
// Dispatch logic (Conv2d)
//

void add_quantize_and_pack_4w4c_node(
    ComputeGraph& graph,
    const ValueRef fp_input,
    const ValueRef input_scale,
    const ValueRef input_zp,
    const ValueRef packed_int8_input) {
  float inv_scale = 1.0f / graph.extract_scalar<float>(input_scale);
  int32_t zp = graph.extract_scalar<int32_t>(input_zp);

  // Get shader for quantized conv2d linear tiled
  std::string kernel_name = "quantize_and_pack_4w4c_per_tensor";
  add_storage_type_suffix(
      kernel_name, graph.storage_type_of(packed_int8_input));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(fp_input));
  add_dtype_suffix(kernel_name, graph.dtype_of(fp_input));

  vkapi::ShaderInfo shader = VK_KERNEL_FROM_STR(kernel_name);

  vkapi::ParamsBindList param_buffers = {graph.sizes_ubo(fp_input)};

  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&inv_scale, sizeof(inv_scale)),
      PushConstantDataInfo(&zp, sizeof(zp)),
  };

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      pick_quantize_and_pack_4w4c_global_wg_size,
      pick_wc_square_wg_size,
      // Inputs and Outputs
      {{packed_int8_input, vkapi::kWrite}, {fp_input, vkapi::kRead}},
      // Shader params buffers
      param_buffers,
      // Push Constants
      push_constants,
      // Specialization Constants
      {},
      // Resize args
      {},
      // Resizing Logic
      nullptr));
}

void add_unpack_4w4c_and_dequantize_node(
    ComputeGraph& graph,
    const ValueRef packed_int8_output,
    const ValueRef output_scale,
    const ValueRef output_zp,
    const ValueRef fp_output) {
  float scale = graph.extract_scalar<float>(output_scale);
  int32_t zp = graph.extract_scalar<int32_t>(output_zp);

  // Get shader for quantized conv2d linear tiled
  std::string kernel_name = "unpack_4w4c_and_dequantize_per_tensor";
  add_storage_type_suffix(kernel_name, graph.storage_type_of(fp_output));
  add_storage_type_suffix(
      kernel_name, graph.storage_type_of(packed_int8_output));
  add_dtype_suffix(kernel_name, graph.dtype_of(fp_output));

  vkapi::ShaderInfo shader = VK_KERNEL_FROM_STR(kernel_name);

  vkapi::ParamsBindList param_buffers = {graph.sizes_ubo(fp_output)};

  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&scale, sizeof(scale)),
      PushConstantDataInfo(&zp, sizeof(zp)),
  };

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      pick_unpack_4w4c_and_dequantize_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{fp_output, vkapi::kWrite}, {packed_int8_output, vkapi::kRead}},
      // Shader params buffers
      param_buffers,
      // Push Constants
      push_constants,
      // Specialization Constants
      {},
      // Resize args
      {},
      // Resizing Logic
      nullptr));
}

//
// Operator Entrypoints
//

void quantize_per_tensor_impl(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int32_t arg_idx = 0;
  const ValueRef fp_input = args[arg_idx++];
  const ValueRef scale = args[arg_idx++];
  const ValueRef zero_point = args[arg_idx++];
  const ValueRef quant_min = args[arg_idx++];
  (void)quant_min;
  const ValueRef quant_max = args[arg_idx++];
  (void)quant_max;
  const ValueRef dtype = args[arg_idx++];
  (void)dtype;

  const ValueRef int8_output = args[arg_idx++];

  VK_CHECK_COND(
      graph.estimate_memory_layout_of(int8_output) == utils::kPackedInt8_4W4C);

  add_quantize_and_pack_4w4c_node(
      graph, fp_input, scale, zero_point, int8_output);
}

void dequantize_per_tensor_impl(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int32_t arg_idx = 0;
  const ValueRef int8_input = args[arg_idx++];
  const ValueRef scale = args[arg_idx++];
  const ValueRef zero_point = args[arg_idx++];
  const ValueRef quant_min = args[arg_idx++];
  (void)quant_min;
  const ValueRef quant_max = args[arg_idx++];
  (void)quant_max;
  const ValueRef dtype = args[arg_idx++];
  (void)dtype;
  const ValueRef output_dtype = args[arg_idx++];
  (void)output_dtype;

  const ValueRef fp_output = args[arg_idx++];

  VK_CHECK_COND(
      graph.estimate_memory_layout_of(int8_input) == utils::kPackedInt8_4W4C);

  add_unpack_4w4c_and_dequantize_node(
      graph, int8_input, scale, zero_point, fp_output);
}

void qdq8ta_conv2d_input(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef fp_input = args.at(idx++);
  const ValueRef scale = args.at(idx++);
  const ValueRef zero_point = args.at(idx++);
  const ValueRef fp_output = args.at(idx++);

  TmpTensor packed_int8_input(
      &graph,
      graph.sizes_of(fp_input),
      vkapi::kInt8x4,
      utils::kBuffer,
      utils::kPackedInt8_4W4C);

  add_quantize_and_pack_4w4c_node(
      graph, fp_input, scale, zero_point, packed_int8_input);

  add_unpack_4w4c_and_dequantize_node(
      graph, packed_int8_input, scale, zero_point, fp_output);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(
      quantized_decomposed.quantize_per_tensor.default,
      quantize_per_tensor_impl);
  VK_REGISTER_OP(
      quantized_decomposed.dequantize_per_tensor.default,
      dequantize_per_tensor_impl);
  VK_REGISTER_OP(etvk.qdq8ta_conv2d_input.default, qdq8ta_conv2d_input);
}

} // namespace vkcompute
