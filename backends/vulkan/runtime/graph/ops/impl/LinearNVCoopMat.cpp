/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

//
// Shader dispatch utilities
//

void resize_linear_tiled_node(
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

utils::uvec3 linear_tiled_nv_cm2_global_wg_size(
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

  // NV cooperative matrix 2 shader uses BM=16 x BN=16 tiles
  // Following ggml's dispatch pattern: x = blocks_m * k_split, y = blocks_n
  const uint32_t BM = 16;
  const uint32_t BN = 16;

  const uint32_t blocks_m = utils::div_up(M, BM);
  const uint32_t blocks_n = utils::div_up(N, BN);

  // x = blocks_m (row tiles), y = blocks_n (column tiles)
  return {blocks_m * 32, blocks_n, 1};
}

// Fixed local workgroup size for NV cooperative matrix 2 linear shader
// Must match the shader's layout(local_size_x = 32)
utils::uvec3 linear_tiled_nv_cm2_local_wg_size(
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

  // NV cooperative matrix 2 with Subgroup scope always uses 32 threads (subgroup size)
  // This matches the shader's layout(local_size_x = 32, local_size_y = 1, local_size_z = 1)
  return {32, 1, 1};
}

//
// Prepacking
//

ValueRef prepack_fp_linear_weight(
    ComputeGraph& graph,
    const ValueRef weight_data,
    const utils::StorageType output_storage_type) {
  std::vector<int64_t> weight_sizes = graph.sizes_of(weight_data);
  const int64_t ndim = graph.dim_of(weight_data);

  // Weight tensor has shape [N, K] (out_features, in_features)
  const int64_t K = weight_sizes.at(ndim - 1);
  const int64_t N = weight_sizes.at(ndim - 2);

  // Calculate output sizes
  // Output layout: [K, N4] where each element is a vec4 containing 4
  // consecutive N values for one K position
  const int64_t N4 = utils::div_up(N, int64_t(4));

  // Determine if we need to fall back to buffer storage
  utils::StorageType storage_type = output_storage_type;
  if (storage_type == utils::kTexture3D) {
    uint32_t max_extent = graph.context()->adapter_ptr()->max_texture2d_dim();
    if (N4 > max_extent || K > max_extent) {
      storage_type = utils::kBuffer;
    }
  }

  // Output tensor shape: [K, N4 * 4] for the prepacked weights
  // The width is N4 * 4 because each vec4 access reads 4 consecutive elements
  // (matching the standard convention where sizes are in terms of individual
  // elements, not vec4s)
  std::vector<int64_t> packed_weight_sizes = {K, N4 * 4};

  ValueRef packed_weight = graph.add_tensor(
      packed_weight_sizes,
      graph.dtype_of(weight_data),
      storage_type,
      utils::kWidthPacked);

  // Store original sizes for the shader
  utils::ivec2 orig_sizes = {
      utils::safe_downcast<int32_t>(K), utils::safe_downcast<int32_t>(N)};

  utils::uvec3 global_wg_size = {
      utils::safe_downcast<uint32_t>(N4),
      utils::safe_downcast<uint32_t>(K),
      1u};

  std::string kernel_name = "pack_fp_linear_weight";
  add_storage_type_suffix(kernel_name, storage_type);
  add_dtype_suffix(kernel_name, graph.dtype_of(weight_data));

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
      {PushConstantDataInfo(&orig_sizes, sizeof(utils::ivec2))}));

  return packed_weight;
}

//
// Linear Dispatch
//

void add_linear_tiled_node(
    ComputeGraph& graph,
    const ValueRef input,
    const ValueRef weight_data,
    const ValueRef packed_weight,
    const ValueRef bias_data,
    const ValueRef packed_bias,
    const ValueRef output) {
  // Use CM2 kernel for buffer storage (GL_NV_cooperative_matrix2)
  std::string kernel_name = "linear_tiled_nv_cm2";
  add_dtype_suffix(kernel_name, graph.dtype_of(output));

  vkapi::ParamsBindList param_buffers = {
      graph.sizes_ubo(output), graph.sizes_ubo(input)};

  int32_t apply_bias = graph.val_is_not_none(bias_data) ? 1 : 0;

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      linear_tiled_nv_cm2_global_wg_size,
      linear_tiled_nv_cm2_local_wg_size,
      // Inputs and Outputs
      {{output, vkapi::kWrite},
       {{input, packed_weight, packed_bias}, vkapi::kRead}},
      // Shader params buffers
      param_buffers,
      // Push Constants
      {},
      // Specialization Constants
      {apply_bias},
      // Resize args
      {weight_data},
      // Resizing Logic
      resize_linear_tiled_node));
}

//
// High-level operator implementation
//

void linear_nv_cm2_impl(
    ComputeGraph& graph,
    const ValueRef input,
    const ValueRef weight_data,
    const ValueRef bias_data,
    const ValueRef output) {
  // Check that VK_NV_cooperative_matrix2 extension is available
  // This is required for the linear_tiled_nv_cm2 shader
  VK_CHECK_COND(
      graph.context()->adapter_ptr()->supports_nv_cooperative_matrix2(),
      "linear_nv_cm2 requires VK_NV_cooperative_matrix2 extension which is "
      "not available on this device. Please use a device that supports "
      "VK_NV_cooperative_matrix2 or use a different linear implementation.");

  // Check input dimensions
  std::vector<int64_t> input_sizes = graph.sizes_of(input);
  VK_CHECK_COND(
      input_sizes.size() == 2 || input_sizes.size() == 3,
      "Input must be 2D or 3D tensor");

  // Determine storage type based on output
  utils::StorageType storage_type = graph.storage_type_of(output);

  // For the tiled implementation, we need the input to be width-packed
  // (i.e., K is along the width/x dimension)
  ValueRef input_W_packed = input;
  if (graph.estimate_memory_layout_of(input) != utils::kWidthPacked) {
    input_W_packed = graph.add_tensor_like(input, utils::kWidthPacked);
    auto viewFn = VK_GET_OP_FN("aten.view_copy.default");
    viewFn(graph, {input, graph.add_none(), input_W_packed});
  }

  // Prepack weight
  ValueRef packed_weight =
      prepack_fp_linear_weight(graph, weight_data, storage_type);

  // Create dummy bias tensor if bias is not provided
  TmpTensor dummy_bias(
      &graph, {}, graph.dtype_of(output), utils::kBuffer, utils::kWidthPacked);

  ValueRef packed_bias = dummy_bias.vref;
  if (graph.val_is_not_none(bias_data)) {
    packed_bias =
        prepack_standard(graph, bias_data, utils::kBuffer, utils::kWidthPacked);
  }

  add_linear_tiled_node(
      graph,
      input_W_packed,
      weight_data,
      packed_weight,
      bias_data,
      packed_bias,
      output);
}

//
// Registered operator entry point
//

void linear_nv_cm2(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  const ValueRef input = args.at(0);
  const ValueRef weight_data = args.at(1);
  const ValueRef bias_data = args.at(2);
  const ValueRef output = args.at(3);

  linear_nv_cm2_impl(graph, input, weight_data, bias_data, output);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(etvk.linear_nv_cm2.default, linear_nv_cm2);
}

} // namespace vkcompute
