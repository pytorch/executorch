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

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/ScalarUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void check_linear_qta8a_qga4w_args(
    ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef mat1_scale,
    const ValueRef mat1_zero_point,
    const ValueRef mat2_data,
    const ValueRef group_size,
    const ValueRef weight_scales,
    const ValueRef weight_zeros,
    const ValueRef out) {
  VK_CHECK_COND(graph.val_is_tensor(mat1));
  VK_CHECK_COND(graph.val_is_tensor(mat1_scale));
  VK_CHECK_COND(graph.val_is_tensor(mat1_zero_point));
  VK_CHECK_COND(graph.val_is_tref(mat2_data));
  VK_CHECK_COND(graph.val_is_tref(weight_scales));
  VK_CHECK_COND(graph.val_is_tref(weight_zeros));

  VK_CHECK_COND(graph.dim_of(mat1) <= 3);
  VK_CHECK_COND(graph.dim_of(mat2_data) == 2);
  VK_CHECK_COND(graph.dim_of(weight_scales) == 2);
  VK_CHECK_COND(graph.dim_of(weight_zeros) == 2);

  VK_CHECK_COND(graph.size_at<int>(-3, mat1) == 1);
  const int K = graph.size_at<int>(-1, mat1);
  VK_CHECK_COND(graph.size_at<int>(-1, mat2_data) * 2 == K);

  const int group_size_val = graph.extract_scalar<int>(group_size);
  VK_CHECK_COND(K % group_size_val == 0);
  // Due to the way weight packing works, group size needs to be a multiple of 8
  VK_CHECK_COND(group_size_val % 8 == 0);

  VK_CHECK_COND(graph.has_standard_axis_map(mat1));
  VK_CHECK_COND(graph.has_standard_axis_map(out));

  // Check that scale and zero_point tensors are buffer storage with width
  // packing
  VK_CHECK_COND(graph.is_buffer_storage(mat1_scale));
  VK_CHECK_COND(graph.packed_dim_of(mat1_scale) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.is_buffer_storage(mat1_zero_point));
  VK_CHECK_COND(graph.packed_dim_of(mat1_zero_point) == WHCN::kWidthDim);

  // Calculate number of tokens for input
  int64_t input_num_tokens = 1;
  const auto mat1_sizes = graph.sizes_of(mat1);
  for (size_t i = 0; i < mat1_sizes.size() - 1; i++) {
    input_num_tokens *= mat1_sizes[i];
  }

  // Verify scale and zero_point tensor sizes match number of tokens
  const auto mat1_scale_sizes = graph.sizes_of(mat1_scale);
  const auto mat1_zero_point_sizes = graph.sizes_of(mat1_zero_point);

  VK_CHECK_COND(
      utils::val_at<int64_t>(-1, mat1_scale_sizes) == input_num_tokens);
  VK_CHECK_COND(
      utils::val_at<int64_t>(-1, mat1_zero_point_sizes) == input_num_tokens);

  // Verify weight scales and zeros have the same shape
  const auto weight_scales_sizes = graph.sizes_of(weight_scales);
  const auto weight_zeros_sizes = graph.sizes_of(weight_zeros);
  VK_CHECK_COND(weight_scales_sizes == weight_zeros_sizes);
}

void resize_linear_qta8a_qga4w_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;

  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef mat1 = args.at(1).refs.at(0);
  const ValueRef mat2 = args.at(1).refs.at(1);

  const std::vector<int64_t> mat1_sizes = graph->sizes_of(mat1);
  const std::vector<int64_t> mat2_sizes = graph->sizes_of(mat2);

  const int64_t out_cols = utils::val_at(-2, mat1_sizes);
  const int64_t out_rows = utils::val_at(-1, mat2_sizes) * 2;

  std::vector<int64_t> new_out_sizes(3);
  if (mat1_sizes.size() == 2) {
    new_out_sizes.resize(2);
    new_out_sizes.at(0) = out_cols;
    new_out_sizes.at(1) = out_rows;
  } else {
    new_out_sizes.at(0) = mat1_sizes.at(0);
    new_out_sizes.at(1) = out_cols;
    new_out_sizes.at(2) = out_rows;
  }

  graph->virtual_resize(out, new_out_sizes);
}

/**
 * Determines if the cooperative algorithm should be used based on input tensor
 * dimensions. Apply the coop algorithm for vectors (GEMV cases), tiled for
 * matrices (GEMM cases).
 */
bool should_use_coop_algorithm_qta8a_qga4w(
    ComputeGraph* graph,
    const ValueRef& mat1) {
  const uint32_t M = graph->size_at<uint32_t>(-2, mat1);
  // Use coop algorithm for vectors (GEMV), tiled for larger matrices (GEMM)
  return M == 1;
}

vkapi::ShaderInfo pick_linear_qta8a_qga4w_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;

  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef mat1 = args.at(1).refs.at(0);
  const ValueRef mat2 = args.at(1).refs.at(1);

  const bool use_coop_algorithm =
      should_use_coop_algorithm_qta8a_qga4w(graph, mat1);

  std::string kernel_name = "linear_qta8a_qga4w";
  if (use_coop_algorithm) {
    kernel_name += "_coop";
  } else {
    kernel_name += "_tiled";
  }
  add_storage_type_suffix(kernel_name, graph->storage_type_of(out));
  add_storage_type_suffix(kernel_name, graph->storage_type_of(mat1));
  add_storage_type_suffix(kernel_name, graph->storage_type_of(mat2));
  add_dtype_suffix(kernel_name, graph->dtype_of(out));

  return VK_KERNEL_FROM_STR(kernel_name);
}

utils::uvec3 linear_qta8a_qga4w_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);

  const bool use_coop_algorithm =
      shader.kernel_name.find("_coop") != std::string::npos;

  // C = 1, H = 2, W = 3
  // global_wg_size = {round_up(C / 2f), round_up(H / 3f), W} --> (2W, 1H, 0C)
  // --> {1, 1, 3} global

  utils::uvec3 global_wg_size = graph->logical_limits_of(out);
  global_wg_size[0] = utils::div_up(global_wg_size[0], uint32_t(2));
  if (!use_coop_algorithm) { // GEMM - TILED
    global_wg_size[1] = utils::div_up(global_wg_size[1], uint32_t(3));
  }

  return global_wg_size;
}

utils::uvec3 linear_qta8a_qga4w_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)args;
  (void)resize_args;

  const bool use_coop_algorithm =
      shader.kernel_name.find("_coop") != std::string::npos;

  utils::uvec3 local_wg_size;
  if (use_coop_algorithm) { // GEMV - COOP
    local_wg_size = {8, 1, 8};
  } else { // GEMM - TILED
    local_wg_size = graph->create_local_wg_size(global_workgroup_size);
  }

  return local_wg_size;
}

void add_linear_qta8a_qga4w_node(
    ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef mat1_scale,
    const ValueRef mat1_zero_point,
    const ValueRef mat2_data,
    const ValueRef group_size,
    const ValueRef weight_scales_data,
    const ValueRef weight_zeros_data,
    const ValueRef out) {
  check_linear_qta8a_qga4w_args(
      graph,
      mat1,
      mat1_scale,
      mat1_zero_point,
      mat2_data,
      group_size,
      weight_scales_data,
      weight_zeros_data,
      out);
  const uint32_t group_size_val = graph.extract_scalar<int32_t>(group_size);

  ValueRef mat2 =
      prepack_int4_linear_weight_transposed_interleaved(graph, mat2_data);
  ValueRef weight_scales = prepack_standard(
      graph, weight_scales_data, utils::kBuffer, utils::kWidthPacked);
  ValueRef weight_zeros = prepack_standard(
      graph, weight_zeros_data, utils::kBuffer, utils::kWidthPacked);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_linear_qta8a_qga4w_shader,
      linear_qta8a_qga4w_global_wg_size,
      linear_qta8a_qga4w_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite},
       {{mat1, mat2, weight_scales, weight_zeros, mat1_scale, mat1_zero_point},
        vkapi::kRead}},
      // Shader params buffers
      {},
      // Push Constants
      {graph.sizes_pc_of(out),
       graph.sizes_pc_of(mat1),
       graph.sizes_pc_of(mat2)},
      // Specialization Constants
      {SV(group_size_val)},
      // Resize Args
      {},
      // Resizing Logic
      resize_linear_qta8a_qga4w_node));
}

void linear_qta8a_qga4w(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  return add_linear_qta8a_qga4w_node(
      graph,
      args[0], // quantized input (char tensor)
      args[1], // input_scale (float buffer tensor)
      args[2], // input_zero_point (int buffer tensor)
      args[3], // quantized weights (4-bit packed, byte)
      args[4], // group_size (int)
      args[5], // weight_scales (float tensor)
      args[6], // weight_zeros (int tensor)
      args[7] // float output tensor
  );
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.linear_qta8a_qga4w.default, linear_qta8a_qga4w);
}

} // namespace vkcompute
