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

utils::uvec3 quantized_linear_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);

  std::vector<int64_t> out_sizes = graph->sizes_of(out);
  // height
  const uint32_t M = utils::val_at(-2, out_sizes);
  // width
  const uint32_t N = utils::val_at(-1, out_sizes);

  // 1 output tile is 4x4 elements
  const uint32_t M4 = utils::div_up(M, 4u);
  const uint32_t N4 = utils::div_up(N, 4u);

  return {N4, M4, 1};
}

utils::uvec3 quantized_linear_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)args;
  (void)resize_args;

  // Optimize local workgroup size for linear operations
  uint32_t local_wg_size_x = 1;
  uint32_t local_wg_size_y = 1;

  if (global_workgroup_size[1] % 8 == 0) {
    local_wg_size_y = 8;
  } else if (global_workgroup_size[1] % 4 == 0) {
    local_wg_size_y = 4;
  } else if (global_workgroup_size[1] % 2 == 0) {
    local_wg_size_y = 2;
  }

  // Adjust x dimension to maintain reasonable total workgroup size
  local_wg_size_x = std::min(64u / local_wg_size_y, global_workgroup_size[0]);

  return {local_wg_size_x, local_wg_size_y, 1};
}

ValueRef prepack_quantized_linear_weight(
    ComputeGraph& graph,
    const ValueRef qmat2_data) {
  std::vector<int64_t> qmat2_orig_sizes = graph.sizes_of(qmat2_data);
  const int64_t ndim = graph.dim_of(qmat2_data);

  // Input is [K, N]
  const int64_t K = qmat2_orig_sizes.at(ndim - 2);
  const int64_t N = qmat2_orig_sizes.at(ndim - 1);

  // N must be a multiple of 4 so data data loads are aligned nicely with texel
  // boundaries.
  VK_CHECK_COND(N % 4 == 0);

  // This packing format partitions the weight tensor into 4 wide x 4 high
  // blocks. To figure out the size of the output tensor, determine the number
  // of blocks along the width and height dims.
  const int64_t num_blocks_K = utils::div_up(K, int64_t(4));
  const int64_t num_blocks_N = utils::div_up(N, int64_t(4));

  // Each transposed block is 4 wide x 4 high. To maximize memory loading
  // efficiency, the packed weight tensor will use a base data type of uint32_t;
  // in terms of uint32_t, each block is 1 wide x 4 high. However, each block is
  // also flattened as it is stored, so that the whole block can be loaded at
  // once. As a result, the stored block will be 4 wide x 1 high.
  const int64_t output_height = num_blocks_K;
  const int64_t output_width = num_blocks_N * 4;

  // Store the original sizes of the tensor to pass to the shader
  utils::ivec2 orig_sizes{
      utils::safe_downcast<int32_t>(K), utils::safe_downcast<int32_t>(N)};

  std::vector<int64_t> qmat2_sizes{output_height, output_width};

  utils::StorageType storage_type = utils::kTexture2D;
  uint32_t max_extent = graph.context()->adapter_ptr()->max_texture2d_dim();
  if (output_width > max_extent * 4 || output_height > max_extent) {
    storage_type = utils::kBuffer;
  }

  ValueRef qmat2 = graph.add_tensor(
      qmat2_sizes, vkcompute::vkapi::kInt, storage_type, utils::kWidthPacked);

  // Global workgroup size: each thread writes out two adjacent blocks
  utils::uvec3 global_wg_size{
      utils::safe_downcast<uint32_t>(num_blocks_N),
      utils::safe_downcast<uint32_t>(num_blocks_K),
      1u};

  std::string kernel_name = "pack_q8_linear_weight";
  add_storage_type_suffix(kernel_name, storage_type);

  graph.prepack_nodes().emplace_back(new PrepackNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_wg_size,
      graph.create_local_wg_size(global_wg_size),
      // Inputs and Outputs
      qmat2_data,
      qmat2,
      // UBOs
      {},
      // Specialization Constants
      {},
      // Push Constants
      {graph.sizes_pc_of(qmat2),
       PushConstantDataInfo(&orig_sizes, sizeof(utils::ivec2))}));

  return qmat2;
}

struct InputQuantConstants {
  alignas(16) float inv_scale;
  alignas(16) int32_t zp;
};

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

utils::uvec3 quant_pack_input_global_wg_size(
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

DynamicDispatchNode make_quantize_and_pack_linear_input_node(
    ComputeGraph& graph,
    const ValueRef input,
    const ValueRef input_scale,
    const ValueRef input_zp,
    const ValueRef quantized_input) {
  int64_t num_blocks_M, num_blocks_K;
  std::tie(num_blocks_M, num_blocks_K) =
      get_quantized_input_num_blocks(graph, input);

  bool is_per_channel = graph.val_is_tensor(input_scale);

  float inv_scale = 1.0f;
  int32_t zp = 0;
  if (!is_per_channel) {
    inv_scale = 1.0f / graph.extract_scalar<float>(input_scale);
    zp = graph.extract_scalar<int32_t>(input_zp);
  }

  std::string shader_name = "quantize_and_pack_linear_input";
  if (is_per_channel) {
    shader_name += "_per_channel";
  } else {
    shader_name += "_per_tensor";
  }
  add_storage_type_suffix(shader_name, graph.storage_type_of(quantized_input));
  add_storage_type_suffix(shader_name, graph.storage_type_of(input));
  add_dtype_suffix(shader_name, graph.dtype_of(input));

  vkapi::ParamsBindList param_buffers = {graph.sizes_ubo(input)};

  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&inv_scale, sizeof(inv_scale)),
      PushConstantDataInfo(&zp, sizeof(zp)),
  };

  return DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(shader_name),
      quant_pack_input_global_wg_size,
      default_pick_local_wg_size,
      // Inputs and Outputs
      {{quantized_input, vkapi::kWrite}, {input, vkapi::kRead}},
      // Shader params buffers
      param_buffers,
      // Push Constants
      push_constants,
      // Specialization Constants
      {},
      // Resize args
      {});
}

DynamicDispatchNode make_linear_q8ta_qw_tiled_node(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  // Extract arguments
  int32_t idx = 0;
  const ValueRef input = args.at(idx++);
  const ValueRef input_scale = args.at(idx++);
  const ValueRef input_zp = args.at(idx++);
  const ValueRef packed_weight = args.at(idx++);
  const ValueRef packed_weight_sums = args.at(idx++);
  const ValueRef packed_weight_scales = args.at(idx++);
  const ValueRef bias = args.at(idx++);
  const ValueRef packed_bias = args.at(idx++);
  const ValueRef output = args.at(idx++);
  const ValueRef original_weight = args.at(idx++); // For resize args

  bool is_per_channel = graph.val_is_tensor(input_scale);

  float scale = 1.0f;
  int32_t zp = 0;
  if (!is_per_channel) {
    scale = graph.extract_scalar<float>(input_scale);
    zp = graph.extract_scalar<int32_t>(input_zp);
  }

  // Get shader for quantized linear
  std::string kernel_name = "linear_q8ta_q8csw_tiled";
  add_storage_type_suffix(kernel_name, graph.storage_type_of(output));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(input));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(packed_weight));
  add_dtype_suffix(kernel_name, graph.dtype_of(output));
  vkapi::ShaderInfo shader = VK_KERNEL_FROM_STR(kernel_name);

  vkapi::ParamsBindList param_buffers = {
      graph.sizes_ubo(output), graph.sizes_ubo(input)};

  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&scale, sizeof(scale)),
      PushConstantDataInfo(&zp, sizeof(zp)),
  };

  uint32_t apply_bias = 0;
  if (!graph.val_is_none(bias)) {
    apply_bias = 1;
  }

  // Add the compute node
  return DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      quantized_linear_global_wg_size,
      quantized_linear_local_wg_size,
      // Inputs and Outputs
      {{output, vkapi::kWrite},
       {{input,
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
      nullptr);
}

DynamicDispatchNode make_linear_qw_node(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  // Extract arguments
  int32_t idx = 0;
  const ValueRef input = args.at(idx++);
  const ValueRef packed_weight = args.at(idx++);
  const ValueRef packed_weight_scales = args.at(idx++);
  const ValueRef bias = args.at(idx++);
  const ValueRef packed_bias = args.at(idx++);
  const ValueRef output = args.at(idx++);
  const ValueRef original_weight = args.at(idx++); // For resize args

  // Get shader for quantized linear
  std::string kernel_name = "linear_q8csw_tiled";
  add_storage_type_suffix(kernel_name, graph.storage_type_of(output));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(input));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(packed_weight));
  add_dtype_suffix(kernel_name, graph.dtype_of(output));
  vkapi::ShaderInfo shader = VK_KERNEL_FROM_STR(kernel_name);

  vkapi::ParamsBindList param_buffers = {
      graph.sizes_ubo(output), graph.sizes_ubo(input)};

  uint32_t apply_bias = 0;
  if (!graph.val_is_none(bias)) {
    apply_bias = 1;
  }

  return DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      quantized_linear_global_wg_size,
      quantized_linear_local_wg_size,
      // Inputs and Outputs
      {{output, vkapi::kWrite},
       {{input, packed_weight, packed_weight_scales, packed_bias},
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
      nullptr);
}

/*
 * Allows orchestration of two compute shader dispatch paths:
 * 1. Quantize & pack input to int8, execute activation and weight quantized
 *    linear operator.
 * 2. Execute fp activation and weight quantized linear operator (skipping the
 *    quantize and pack input step)
 *
 * The reason for this split is twofold:
 * - Some devices may not support accelerated int8 dot product. In that case,
 *   there is no benefit to quantizing the activation tensor, as the goal with
 *   quantizing the activation is to achieve higher arithmetic throughput via
 *   the int8 dot product extensions.
 * - For LLMs, which switch between GEMM and GEMV input conditions when going
 *   from prefill to decode. GEMM is typically a compute bound operation, which
 *   will benefit from accelerated int8 accumulation. On the other hand, GEMV
 *   is usually memory bound, which means it may actually suffer from the extra
 *   cost of having to quantize and pack the input tensor. Therefore,
 *   linear_q8ta_qw is preferred for GEMM and linear_qw is preferred for GEMV.
 *
 * Note that dynamic shape is currently not supported, so switching paths
 * when input conditions go between GEMM -> GEMV is currently not implemented.
 * This will be implemented at a later date.
 */
struct QuantizedLinearNode : public ExecuteNode {
  friend class ComputeGraph;

  bool can_use_int8_dot_product = false;
  DynamicDispatchNode quantize_and_pack_input_node;
  DynamicDispatchNode linear_q8ta_qw_tiled_node;
  DynamicDispatchNode linear_qw_node;

  explicit QuantizedLinearNode(
      ComputeGraph& graph,
      const std::vector<ValueRef>& args,
      DynamicDispatchNode&& quant_pack_input,
      DynamicDispatchNode&& qaqw_tiled_linear,
      DynamicDispatchNode&& qw_linear,
      bool int8_dot_product_enabled)
      : ExecuteNode(),
        quantize_and_pack_input_node(quant_pack_input),
        linear_q8ta_qw_tiled_node(qaqw_tiled_linear),
        linear_qw_node(qw_linear) {
    if (int8_dot_product_enabled) {
      can_use_int8_dot_product = graph.can_use_int8_dot_product();
    }
  }

  void prepare_pipelines(ComputeGraph* graph) override {
    if (can_use_int8_dot_product) {
      quantize_and_pack_input_node.prepare_pipelines(graph);
      linear_q8ta_qw_tiled_node.prepare_pipelines(graph);
    }
    linear_qw_node.prepare_pipelines(graph);
  }

  void encode(ComputeGraph* graph) override {
    if (can_use_int8_dot_product) {
      quantize_and_pack_input_node.encode(graph);
      linear_q8ta_qw_tiled_node.encode(graph);
    } else {
      linear_qw_node.encode(graph);
    }
  }
};

/*
 * Implements activation and weight quantized linear. Currently, only the
 * following quantization configurations are supported:
 * - activation quantized to int8 with per tensor quant params
 * - weight quantized to int8 with per channel quant params
 */
void linear_q8ta_qw_impl(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args,
    const bool use_int8_dot_product = true) {
  int32_t idx = 0;
  const ValueRef input = args.at(idx++);
  const ValueRef input_scale = args.at(idx++);
  const ValueRef input_zp = args.at(idx++);
  const ValueRef weight = args.at(idx++);
  const ValueRef weight_sums = args.at(idx++);
  const ValueRef weight_scales = args.at(idx++);
  const ValueRef orig_OC = args.at(idx++);
  (void)orig_OC; // unused
  const ValueRef bias = args.at(idx++);
  const ValueRef output = args.at(idx++);

  bool is_per_channel = graph.val_is_tensor(input_scale);

  // Input validation
  std::vector<int64_t> input_sizes = graph.sizes_of(input);
  std::vector<int64_t> weight_sizes = graph.sizes_of(weight);

  const int64_t K = utils::val_at(-1, input_sizes);
  // K (input channels) must be a multiple of 4 to ensure that reading a group
  // of 4 input channels from the input tensor will be aligned on a texel
  // boundary.
  VK_CHECK_COND(K % 4 == 0);

  const int64_t N = utils::val_at(-1, input_sizes);
  // N (output channels) must be a multiple of 4 to ensure that reading a group
  // of 4 output channels from the weight/output tensor will be aligned on a
  // texel boundary.
  VK_CHECK_COND(N % 4 == 0);

  // Prepacking
  const ValueRef packed_weight = prepack_quantized_linear_weight(graph, weight);
  ValueRef packed_weight_scales = prepack_standard(
      graph, weight_scales, utils::kBuffer, utils::kWidthPacked);
  ValueRef packed_weight_sums =
      prepack_standard(graph, weight_sums, utils::kBuffer, utils::kWidthPacked);

  ValueRef packed_input_scale = input_scale;
  ValueRef packed_input_zp = input_zp;
  if (is_per_channel) {
    packed_input_scale = prepack_standard(
        graph, input_scale, utils::kBuffer, utils::kWidthPacked);
    packed_input_zp =
        prepack_standard(graph, input_zp, utils::kBuffer, utils::kWidthPacked);
  }

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

  int64_t num_blocks_M, num_blocks_K;
  std::tie(num_blocks_M, num_blocks_K) =
      get_quantized_input_num_blocks(graph, input);

  const int64_t quantized_input_height = num_blocks_M;
  const int64_t quantized_input_width = num_blocks_K * 4;

  TmpTensor quantized_packed_input(
      &graph,
      {quantized_input_height, quantized_input_width},
      vkapi::kInt,
      graph.storage_type_of(input),
      utils::kWidthPacked);

  DynamicDispatchNode quantize_and_pack_linear_node(
      make_quantize_and_pack_linear_input_node(
          graph,
          input,
          packed_input_scale,
          packed_input_zp,
          quantized_packed_input));

  std::vector<ValueRef> linear_args = {
      quantized_packed_input,
      packed_input_scale,
      packed_input_zp,
      packed_weight,
      packed_weight_sums,
      packed_weight_scales,
      bias,
      packed_bias,
      output,
      weight};

  DynamicDispatchNode linear_q8ta_qw_tiled_node(
      make_linear_q8ta_qw_tiled_node(graph, linear_args));

  linear_args = {
      input,
      packed_weight,
      packed_weight_scales,
      bias,
      packed_bias,
      output,
      weight};

  DynamicDispatchNode linear_qw_node(make_linear_qw_node(graph, linear_args));

  graph.execute_nodes().emplace_back(new QuantizedLinearNode(
      graph,
      linear_args,
      std::move(quantize_and_pack_linear_node),
      std::move(linear_q8ta_qw_tiled_node),
      std::move(linear_qw_node),
      use_int8_dot_product));
}

void linear_q8ta_q8csw(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  linear_q8ta_qw_impl(graph, args, true);
}

void linear_q8ta_q8csw_no_int8(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  linear_q8ta_qw_impl(graph, args, false);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.linear_q8ta_q8csw.default, linear_q8ta_q8csw);
  VK_REGISTER_OP(et_vk.linear_q8ta_q8csw.noint8, linear_q8ta_q8csw_no_int8);
}

} // namespace vkcompute
