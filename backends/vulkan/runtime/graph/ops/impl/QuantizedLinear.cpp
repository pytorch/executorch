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
// Shader dispatch utilities
//

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
  return pick_hw_square_wg_size(
      graph, shader, global_workgroup_size, args, resize_args);
}

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

//
// Prepacking nodes
//

ValueRef prepack_quantized_linear_weight(
    ComputeGraph& graph,
    const QuantizationConfig& weight_quant_config,
    const ValueRef qmat2_data) {
  VK_CHECK_COND(weight_quant_config.nbits == 8);

  std::vector<int64_t> qmat2_orig_sizes = graph.sizes_of(qmat2_data);
  const int64_t ndim = graph.dim_of(qmat2_data);

  // Input size is [N, K]. K will be guaranteed to be a multiple of 4.
  const int64_t K = qmat2_orig_sizes.at(ndim - 1);
  const int64_t N = qmat2_orig_sizes.at(ndim - 2);

  // Sanity check that assumption is correct
  VK_CHECK_COND(K % 4 == 0);

  // The packing format packs the weight tensor into units of 4 wide x 4 high
  // blocks. To figure out the size of the output tensor, determine the number
  // of blocks along each dimension.
  const int64_t num_blocks_K = utils::div_up(K, int64_t(4));
  const int64_t num_blocks_N = utils::div_up(N, int64_t(4));

  // The blocks are arranged in a transposed manner, such that the transposed
  // weight block is indexed like packed_weights[k4][n4] - this is to allow for
  // optimal memory coalescing when computing GEMM.
  const int64_t output_height = num_blocks_K;
  // The base dtype of the packed tensor is int32 (each int32 contains 4x 8bit
  // values) and each block is represented as a ivec4. Therefore the width dim
  // of the packed tensor is multiplied by 4.
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

//
// Dispatch nodes
//

/*
 * Shader dispatch for linear with quantized weight but fp activations.
 */
DynamicDispatchNode make_linear_qw_node(
    ComputeGraph& graph,
    const QuantizationConfig& weight_quant_config,
    const ValueRef fp_input,
    const ValueRef weight_data,
    const ValueRef packed_weight,
    const ValueRef packed_weight_scales,
    const ValueRef packed_weight_zeros,
    const ValueRef group_size,
    const ValueRef bias_data,
    const ValueRef packed_bias,
    const ValueRef output) {
  // Only certain quantization types supported at the moment
  VK_CHECK_COND(weight_quant_config.granularity == kPerChannel);
  VK_CHECK_COND(weight_quant_config.is_symmetric);
  VK_CHECK_COND(weight_quant_config.nbits == 8);

  std::string kernel_name = "linear_q8csw_tiled";
  add_storage_type_suffix(kernel_name, graph.storage_type_of(output));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(packed_weight));
  add_dtype_suffix(kernel_name, graph.dtype_of(output));
  vkapi::ShaderInfo shader = VK_KERNEL_FROM_STR(kernel_name);

  vkapi::ParamsBindList param_buffers = {
      graph.sizes_ubo(output), graph.sizes_ubo(fp_input)};

  uint32_t apply_bias = 1;
  if (graph.val_is_none(bias_data)) {
    apply_bias = 0;
  }

  return DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      quantized_linear_global_wg_size,
      quantized_linear_local_wg_size,
      // Inputs and Outputs
      {{output, vkapi::kWrite},
       {{fp_input, packed_weight, packed_weight_scales, packed_bias},
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
      nullptr);
}

DynamicDispatchNode make_quantize_and_pack_linear_input_node(
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

  std::string shader_name = "quantize_and_pack_linear_input_per_tensor";
  add_storage_type_suffix(shader_name, graph.storage_type_of(packed_int_input));
  add_storage_type_suffix(shader_name, graph.storage_type_of(fp_input));
  add_dtype_suffix(shader_name, graph.dtype_of(fp_input));

  vkapi::ParamsBindList param_buffers = {graph.sizes_ubo(fp_input)};

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
      {{packed_int_input, vkapi::kWrite}, {fp_input, vkapi::kRead}},
      // Shader params buffers
      param_buffers,
      // Push Constants
      push_constants,
      // Specialization Constants
      {},
      // Resize args
      {});
}

DynamicDispatchNode make_linear_qa_qw_node(
    ComputeGraph& graph,
    const QuantizationConfig& input_quant_config,
    const QuantizationConfig& weight_quant_config,
    const ValueRef fp_input,
    const ValueRef packed_int_input,
    const ValueRef packed_input_scale,
    const ValueRef packed_input_zp,
    const ValueRef input_scale_data,
    const ValueRef input_zp_data,
    const ValueRef weight_data,
    const ValueRef packed_weight,
    const ValueRef packed_weight_sums,
    const ValueRef packed_weight_scales,
    const ValueRef group_size,
    const ValueRef bias_data,
    const ValueRef packed_bias,
    const ValueRef output) {
  VK_CHECK_COND(input_quant_config.granularity == kPerTensor);
  VK_CHECK_COND(input_quant_config.nbits == 8);
  VK_CHECK_COND(weight_quant_config.granularity == kPerChannel);
  VK_CHECK_COND(weight_quant_config.is_symmetric);
  VK_CHECK_COND(weight_quant_config.nbits == 8);

  float scale = graph.extract_scalar<float>(input_scale_data);
  int32_t zp = graph.extract_scalar<int32_t>(input_zp_data);

  // Get shader for quantized linear
  std::string kernel_name = "linear_q8ta_q8csw_tiled";
  add_storage_type_suffix(kernel_name, graph.storage_type_of(output));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(packed_int_input));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(packed_weight));
  add_dtype_suffix(kernel_name, graph.dtype_of(output));
  vkapi::ShaderInfo shader = VK_KERNEL_FROM_STR(kernel_name);

  vkapi::ParamsBindList param_buffers = {
      graph.sizes_ubo(output), graph.sizes_ubo(packed_int_input)};

  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&scale, sizeof(scale)),
      PushConstantDataInfo(&zp, sizeof(zp)),
  };

  uint32_t apply_bias = 1;
  if (graph.val_is_none(bias_data)) {
    apply_bias = 0;
  }

  // Add the compute node
  return DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      quantized_linear_global_wg_size,
      quantized_linear_local_wg_size,
      // Inputs and Outputs
      {{output, vkapi::kWrite},
       {{packed_int_input,
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
      {fp_input},
      // Resizing Logic
      nullptr);
}

//
// High level operator impl
//

void quantized_linear_impl(
    ComputeGraph& graph,
    const QuantizationConfig& input_quant_config,
    const QuantizationConfig& weight_quant_config,
    const ValueRef fp_input,
    const ValueRef input_scale,
    const ValueRef input_zp,
    const ValueRef weight_data,
    const ValueRef weight_sums_data,
    const ValueRef weight_scales_data,
    const ValueRef weight_zeros_data,
    const ValueRef group_size,
    const ValueRef bias_data,
    const ValueRef output) {
  std::vector<int64_t> input_sizes = graph.sizes_of(fp_input);
  std::vector<int64_t> weight_sizes = graph.sizes_of(weight_data);

  const int64_t K = utils::val_at(-1, input_sizes);
  // K (input channels) must be a multiple of 4 to ensure that reading a group
  // of 4 input channels from the input tensor will be aligned on a texel
  // boundary.
  VK_CHECK_COND(K % 4 == 0);

  // Prepack weight data

  const ValueRef packed_weight =
      prepack_quantized_linear_weight(graph, weight_quant_config, weight_data);
  const ValueRef packed_weight_scales = prepack_standard(
      graph, weight_scales_data, utils::kBuffer, utils::kWidthPacked);
  // Weight affine quant not supported at the moment
  const ValueRef packed_weight_zeros = kDummyValueRef;

  // Prepack bias data

  // Create a dummy tensor to fill the binding slot of the bias tensor if it is
  // not provided. This helps simplify dispatch logic and makes it so that
  // fewer shdaer variants need to be generated.
  TmpTensor dummy_bias(
      &graph, {}, graph.dtype_of(output), utils::kBuffer, utils::kWidthPacked);

  ValueRef packed_bias = dummy_bias.vref;
  if (graph.val_is_not_none(bias_data)) {
    packed_bias =
        prepack_standard(graph, bias_data, utils::kBuffer, utils::kWidthPacked);
  }

  // Use weight only quantized linear if at least one is true:
  // 1. Device does not support int8 dot product
  // 2. Input is not quantized
  if (!graph.can_use_int8_dot_product() ||
      input_quant_config.granularity == kNoQuantization) {
    DynamicDispatchNode linear_qw_node(make_linear_qw_node(
        graph,
        weight_quant_config,
        fp_input,
        weight_data,
        packed_weight,
        packed_weight_scales,
        packed_weight_zeros,
        group_size,
        bias_data,
        packed_bias,
        output));

    graph.execute_nodes().emplace_back(new DynamicDispatchNode(linear_qw_node));
    return;
  } else {
    // Otherwise, use input and weight quantized linear computed with integer
    // accumulation

    // Input scale/zero point only used for activation & weight quantized linear
    ValueRef packed_input_scale = input_scale;
    ValueRef packed_input_zp = input_zp;
    if (graph.val_is_tref(input_scale)) {
      VK_CHECK_COND(graph.val_is_tref(packed_input_zp));
      packed_input_scale = prepack_standard(
          graph, input_scale, utils::kBuffer, utils::kWidthPacked);
      packed_input_zp = prepack_standard(
          graph, input_zp, utils::kBuffer, utils::kWidthPacked);
    }

    // Pre-computed per quant group weight sums are needed for int accumulation,
    // but not for weight only
    const ValueRef packed_weight_sums = prepack_standard(
        graph, weight_sums_data, utils::kBuffer, utils::kWidthPacked);

    // Allocate temporary tensor to store quantized and packed input

    int64_t num_blocks_M, num_blocks_K;
    std::tie(num_blocks_M, num_blocks_K) =
        get_quantized_input_num_blocks(graph, fp_input);

    const int64_t int_input_height = num_blocks_M;
    const int64_t int_input_width = num_blocks_K * 4;

    TmpTensor packed_int_input(
        &graph,
        {int_input_height, int_input_width},
        vkapi::kInt,
        utils::kBuffer,
        utils::kWidthPacked);

    DynamicDispatchNode quantize_and_pack_linear_node(
        make_quantize_and_pack_linear_input_node(
            graph,
            input_quant_config,
            fp_input,
            packed_input_scale,
            packed_input_zp,
            input_scale,
            input_zp,
            packed_int_input,
            group_size));

    graph.execute_nodes().emplace_back(
        new DynamicDispatchNode(quantize_and_pack_linear_node));

    DynamicDispatchNode linear_qa_qw_node(make_linear_qa_qw_node(
        graph,
        input_quant_config,
        weight_quant_config,
        fp_input,
        packed_int_input,
        packed_input_scale,
        packed_input_zp,
        input_scale,
        input_zp,
        weight_data,
        packed_weight,
        packed_weight_sums,
        packed_weight_scales,
        group_size,
        bias_data,
        packed_bias,
        output));

    graph.execute_nodes().emplace_back(
        new DynamicDispatchNode(linear_qa_qw_node));
  }
}

void linear_q8ta_q8csw(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef fp_input = args.at(idx++);
  const ValueRef input_scale = args.at(idx++);
  const ValueRef input_zp = args.at(idx++);
  const ValueRef weight_data = args.at(idx++);
  const ValueRef weight_sums_data = args.at(idx++);
  const ValueRef weight_scales_data = args.at(idx++);
  const ValueRef bias_data = args.at(idx++);
  const ValueRef output = args.at(idx++);

  const int64_t K = graph.size_at<int64_t>(-1, fp_input);

  QuantizationConfig input_quant_config(8, kPerTensor, {}, false);
  QuantizationConfig weight_quant_config(8, kPerChannel, {K});

  quantized_linear_impl(
      graph,
      input_quant_config,
      weight_quant_config,
      fp_input,
      input_scale,
      input_zp,
      weight_data,
      weight_sums_data,
      weight_scales_data,
      kDummyValueRef, // weight_zeros_data
      kDummyValueRef, // group_size
      bias_data,
      output);
}

void linear_q8csw(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef fp_input = args.at(idx++);
  const ValueRef weight_data = args.at(idx++);
  const ValueRef weight_scales_data = args.at(idx++);
  const ValueRef bias_data = args.at(idx++);
  const ValueRef output = args.at(idx++);

  const int64_t K = graph.size_at<int64_t>(-1, fp_input);

  QuantizationConfig input_quant_config(32, kNoQuantization, {});
  QuantizationConfig weight_quant_config(8, kPerChannel, {K});

  quantized_linear_impl(
      graph,
      input_quant_config,
      weight_quant_config,
      fp_input,
      kDummyValueRef, // input scale
      kDummyValueRef, // input zp
      weight_data,
      kDummyValueRef, // weight sums
      weight_scales_data,
      kDummyValueRef, // weight zeros
      kDummyValueRef, // group size
      bias_data,
      output);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.linear_q8ta_q8csw.default, linear_q8ta_q8csw);
  VK_REGISTER_OP(et_vk.linear_q8csw.default, linear_q8csw);
}

} // namespace vkcompute
