/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Q8taQuantizeDequantize.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/QuantizeDequantize.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

//
// Shader dispatch utilities
//

void resize_linear_q8ta_q8csw_nv_cm2_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;

  const ValueRef output = args.at(0).refs.at(0);
  const ValueRef input = args.at(1).refs.at(0);
  const ValueRef weight_data = extra_args.at(0);

  std::vector<int64_t> input_sizes = graph->sizes_of(input);
  std::vector<int64_t> weight_sizes = graph->sizes_of(weight_data);

  // input: [M, K], weight: [N, K] -> output: [M, N]
  const int64_t M = utils::val_at(-2, input_sizes);
  const int64_t N = utils::val_at(-2, weight_sizes);

  std::vector<int64_t> new_out_sizes(input_sizes.size());
  if (input_sizes.size() == 2) {
    new_out_sizes.at(0) = M;
    new_out_sizes.at(1) = N;
  } else {
    new_out_sizes.at(0) = input_sizes.at(0);
    new_out_sizes.at(1) = M;
    new_out_sizes.at(2) = N;
  }

  graph->virtual_resize(output, new_out_sizes);
}

utils::uvec3 linear_q8ta_q8csw_nv_cm2_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;

  const ValueRef output = args.at(0).refs.at(0);

  std::vector<int64_t> out_sizes = graph->sizes_of(output);
  // Width dimension (N = out_features)
  const uint32_t N = utils::val_at(-1, out_sizes);
  // Height dimension (M = batch size)
  const uint32_t M = utils::val_at(-2, out_sizes);

  // NV cooperative matrix 2 shader uses BM=16 x BN=16 tiles for int8
  const uint32_t BM = 16;
  const uint32_t BN = 16;

  const uint32_t blocks_m = utils::div_up(M, BM);
  const uint32_t blocks_n = utils::div_up(N, BN);

  // Each workgroup (32 threads = 1 subgroup) processes one BM x BN tile
  return {blocks_m*32, blocks_n, 1};
}

utils::uvec3 linear_q8ta_q8csw_nv_cm2_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)graph;
  (void)shader;
  (void)global_workgroup_size;
  (void)args;
  (void)resize_args;

  // NV cooperative matrix 2 with Subgroup scope uses 32 threads (1 subgroup)
  return {32, 1, 1};
}

//
// Prepacking
//

ValueRef prepack_int8_linear_weight_nv_cm2(
    ComputeGraph& graph,
    const QuantizationConfig& weight_quant_config,
    const ValueRef weight_data) {
  VK_CHECK_COND(weight_quant_config.nbits == 8);

  std::vector<int64_t> weight_sizes = graph.sizes_of(weight_data);
  const int64_t ndim = graph.dim_of(weight_data);

  // Weight tensor has shape [N, K] (out_features, in_features)
  const int64_t K = weight_sizes.at(ndim - 1);
  const int64_t N = weight_sizes.at(ndim - 2);

  // Calculate output sizes for prepacked weight
  // Output layout: [K, N4 * 4] where N4 = ceil(N / 4)
  const int64_t N4 = utils::div_up(N, int64_t(4));

  utils::StorageType storage_type = utils::kBuffer;

  std::vector<int64_t> packed_weight_sizes = {K, N4 * 4};

  ValueRef packed_weight = graph.add_tensor(
      packed_weight_sizes, vkapi::kInt, storage_type, utils::kWidthPacked);

  // Store original sizes for the shader
  utils::ivec2 orig_sizes = {
      utils::safe_downcast<int32_t>(K), utils::safe_downcast<int32_t>(N)};

  utils::uvec3 global_wg_size = {
      utils::safe_downcast<uint32_t>(N4),
      utils::safe_downcast<uint32_t>(K),
      1u};

  std::string kernel_name = "pack_q8_linear_weight";
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
       PushConstantDataInfo(&orig_sizes, sizeof(utils::ivec2))}));

  return packed_weight;
}

//
// Linear Dispatch
//

void add_linear_q8ta_q8csw_nv_cm2_node(
    ComputeGraph& graph,
    const ValueRef packed_int_input,
    const ValueRef weight_data,
    const ValueRef packed_weight,
    const ValueRef packed_weight_sums,
    const ValueRef packed_weight_scales,
    const ValueRef bias_data,
    const ValueRef packed_bias,
    const ValueRef input_scale,
    const ValueRef input_zero_point,
    const ValueRef output) {
  std::string kernel_name = "linear_q8ta_q8csw_nv_cm2";
  add_storage_type_suffix(kernel_name, graph.storage_type_of(output));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(packed_int_input));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(packed_weight));
  add_dtype_suffix(kernel_name, graph.dtype_of(output));

  vkapi::ParamsBindList param_buffers = {
      graph.sizes_ubo(output), graph.sizes_ubo(packed_int_input)};

  // Extract input_scale and input_zp for push constants
  float in_scale = graph.extract_scalar<float>(input_scale);
  int32_t in_zp = graph.extract_scalar<int32_t>(input_zero_point);

  struct PushConstants {
    float input_scale;
    int32_t input_zp;
  } push_data = {in_scale, in_zp};

  std::vector<PushConstantDataInfo> push_constants = {
      PushConstantDataInfo(&push_data, sizeof(push_data)),
  };

  int32_t apply_bias = graph.val_is_not_none(bias_data) ? 1 : 0;

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      linear_q8ta_q8csw_nv_cm2_global_wg_size,
      linear_q8ta_q8csw_nv_cm2_local_wg_size,
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
      {weight_data},
      // Resizing Logic
      resize_linear_q8ta_q8csw_nv_cm2_node));
}

//
// High-level operator implementation
//

void linear_q8ta_q8csw_nv_cm2_impl(
    ComputeGraph& graph,
    const ValueRef input_tensor,
    const ValueRef input_scale,
    const ValueRef input_zero_point,
    const ValueRef weight_data,
    const ValueRef weight_sums,
    const ValueRef weight_scales,
    const ValueRef bias,
    const ValueRef output) {
  // Check that VK_NV_cooperative_matrix2 extension is available
  VK_CHECK_COND(
      graph.context()->adapter_ptr()->supports_nv_cooperative_matrix2(),
      "linear_q8ta_q8csw_nv_cm2 requires VK_NV_cooperative_matrix2 extension "
      "which is not available on this device.");

  std::vector<int64_t> input_sizes = graph.sizes_of(input_tensor);
  VK_CHECK_COND(
      input_sizes.size() == 2 || input_sizes.size() == 3,
      "Input must be 2D or 3D tensor");

  // Prepack weight data - just upload to GPU buffer without transformation
  // Weight is in [N, K] format from input
  const ValueRef packed_weight = prepack_standard(
      graph, weight_data, utils::kBuffer, utils::kWidthPacked);

  // Prepack weight_sums
  const ValueRef packed_weight_sums = prepack_standard(
      graph, weight_sums, utils::kBuffer, utils::kWidthPacked);

  // Prepack weight_scales
  const ValueRef packed_weight_scales = prepack_standard(
      graph, weight_scales, utils::kBuffer, utils::kWidthPacked);

  // Prepack bias
  // Create a dummy tensor to fill the binding slot of the bias tensor if it is
  // not provided. This helps simplify dispatch logic and makes it so that
  // fewer shader variants need to be generated.
  TmpTensor dummy_bias(
      &graph, {}, graph.dtype_of(output), utils::kBuffer, utils::kWidthPacked);
  ValueRef packed_bias = dummy_bias.vref;
  if (graph.val_is_not_none(bias)) {
    packed_bias = prepack_standard(
        graph, bias, utils::kBuffer, utils::kWidthPacked);
  }

  // Check if input is float type and quantize to int8 if needed
  ValueRef quantized_input = input_tensor;
  vkapi::ScalarType input_dtype = graph.dtype_of(input_tensor);
  if (input_dtype == vkapi::kFloat || input_dtype == vkapi::kHalf) {
    // Create quantized int8 output tensor
    TmpTensor quantized_tensor(
        &graph,
        input_sizes,
        vkapi::kInt8x4,
        utils::kBuffer,
        utils::kPackedInt8_4W);

    // Add quantization node to convert float input to int8
    add_q8ta_quantize_node(
        graph,
        input_tensor,
        input_scale,
        input_zero_point,
        quantized_tensor.vref);

    quantized_input = quantized_tensor.vref;
  }

  add_linear_q8ta_q8csw_nv_cm2_node(
      graph,
      quantized_input,
      weight_data,
      packed_weight,
      packed_weight_sums,
      packed_weight_scales,
      bias,
      packed_bias,
      input_scale,
      input_zero_point,
      output);
}

//
// Registered operator entry point
//

void linear_q8ta_q8csw_nv_cm2(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  int32_t idx = 0;
  const ValueRef input_tensor = args.at(idx++);
  const ValueRef input_scale = args.at(idx++);
  const ValueRef input_zero_point = args.at(idx++);
  const ValueRef weight_data = args.at(idx++);
  const ValueRef weight_sums = args.at(idx++);
  const ValueRef weight_scales = args.at(idx++);
  const ValueRef bias = args.at(idx++);
  const ValueRef output = args.at(idx++);

  linear_q8ta_q8csw_nv_cm2_impl(
      graph,
      input_tensor,
      input_scale,
      input_zero_point,
      weight_data,
      weight_sums,
      weight_scales,
      bias,
      output);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(
      et_vk.linear_q8ta_q8csw_nv_cm2.default,
      linear_q8ta_q8csw_nv_cm2);
}

} // namespace vkcompute
