/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/ScalarUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void check_q_4w_linear_args(
    ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef mat2_data,
    const ValueRef group_size,
    const ValueRef scales_and_zeros,
    const ValueRef out) {
  VK_CHECK_COND(graph.val_is_tensor(mat1));
  VK_CHECK_COND(graph.val_is_tref(mat2_data));
  VK_CHECK_COND(graph.val_is_tref(scales_and_zeros));

  VK_CHECK_COND(graph.dim_of(mat1) <= 3);
  VK_CHECK_COND(graph.dim_of(mat2_data) == 2);
  VK_CHECK_COND(graph.dim_of(scales_and_zeros) == 3);

  VK_CHECK_COND(graph.size_at<int>(-3, mat1) == 1);
  const int K = graph.size_at<int>(-1, mat1);
  VK_CHECK_COND(graph.size_at<int>(-1, mat2_data) * 2 == K);

  const int group_size_val = graph.extract_scalar<int>(group_size);
  VK_CHECK_COND(K % group_size_val == 0);
  // Due to the way weight packing works, group size needs to be a multiple of 8
  VK_CHECK_COND(group_size_val % 8 == 0);

  VK_CHECK_COND(graph.has_standard_axis_map(mat1));
  VK_CHECK_COND(graph.has_standard_axis_map(out));
}

void resize_q_4w_linear_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;

  vTensorPtr out = graph->get_tensor(args[0].refs[0]);
  vTensorPtr mat1 = graph->get_tensor(args[1].refs[0]);
  vTensorPtr mat2 = graph->get_tensor(args[1].refs[1]);

  const int out_cols = utils::val_at(-2, mat1->sizes());
  const int out_rows = utils::val_at(-1, mat2->sizes()) * 2;

  std::vector<int64_t> new_out_sizes(3);
  if (mat1->sizes().size() == 2) {
    new_out_sizes.resize(2);
    new_out_sizes.at(0) = out_cols;
    new_out_sizes.at(1) = out_rows;
  } else {
    new_out_sizes.at(0) = mat1->sizes().at(0);
    new_out_sizes.at(1) = out_cols;
    new_out_sizes.at(2) = out_rows;
  }

  out->virtual_resize(new_out_sizes);
}

ValueRef prepack_int4_linear_weight_transposed_interleaved(
    ComputeGraph& graph,
    const ValueRef qmat2_data) {
  std::vector<int64_t> qmat2_orig_sizes = graph.sizes_of(qmat2_data);
  const int64_t ndim = graph.dim_of(qmat2_data);

  const int64_t K = qmat2_orig_sizes.at(ndim - 1) * 2;
  const int64_t N = qmat2_orig_sizes.at(ndim - 2);
  const int64_t N_div2 = N / int64_t(2);

  utils::StorageType storage_type = utils::kTexture2D;
  uint32_t max_extent = graph.context()->adapter_ptr()->max_texture2d_dim();
  if (N_div2 > max_extent * 4 || K > max_extent) {
    storage_type = utils::kBuffer;
  }

  std::vector<int64_t> qmat2_sizes{K, N_div2};
  ValueRef qmat2 = graph.add_tensor(
      qmat2_sizes, vkcompute::vkapi::kByte, storage_type, utils::kWidthPacked);

  utils::uvec3 global_wg_size;
  global_wg_size = graph.logical_limits_of(qmat2);
  global_wg_size[1] = utils::div_up(global_wg_size[1], uint32_t(2));

  std::string kernel_name =
      graph.context()->adapter_ptr()->has_full_int8_buffers_support()
      ? "pack_int4_linear_weight_transposed_interleaved"
      : "pack_int4_linear_weight_transposed_interleaved_nobitw8buffer";
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
      {graph.sizes_pc_of(qmat2)}));

  return qmat2;
}

void add_q_4w_linear_node(
    ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef mat2_data,
    const ValueRef group_size,
    const ValueRef scales_and_zeros_data,
    const ValueRef out) {
  check_q_4w_linear_args(
      graph, mat1, mat2_data, group_size, scales_and_zeros_data, out);

  const uint32_t group_size_val = graph.extract_scalar<uint32_t>(group_size);

  bool use_coop_algorithm = false;
  // Apply the coop algorithm for gemv cases, i.e. mat1 is a vector as opposed
  // to a matrix.
  if (graph.size_at<uint32_t>(-2, mat1) == 1) {
    use_coop_algorithm = true;
  }

  ValueRef mat2 =
      prepack_int4_linear_weight_transposed_interleaved(graph, mat2_data);

  ValueRef scales_and_zeros = prepack_standard_hw_transposed(
      graph, scales_and_zeros_data, utils::kBuffer, utils::kWidthPacked);

  std::string kernel_name = "q_4w_linear";
  if (use_coop_algorithm) {
    kernel_name += "_coop";
  } else {
    kernel_name += "_tiled";
  }
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(mat1));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(mat2));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  utils::uvec3 global_wg_size = graph.logical_limits_of(out);
  global_wg_size[0] = utils::div_up(global_wg_size[0], uint32_t(2));
  utils::uvec3 local_wg_size = graph.create_local_wg_size(global_wg_size);

  if (use_coop_algorithm) {
    local_wg_size = {8, 1, 8};
  } else {
    global_wg_size[1] = utils::div_up(global_wg_size[1], uint32_t(3));
  }

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_wg_size,
      local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{mat1, mat2, scales_and_zeros}, vkapi::kRead}},
      // Shader params buffers
      {},
      // Specialization Constants
      {SV(group_size_val)},
      // Resizing Logic
      resize_q_4w_linear_node,
      {},
      // Push Constants
      {graph.sizes_pc_of(out),
       graph.sizes_pc_of(mat1),
       graph.sizes_pc_of(mat2)}));
}

void linear_weight_int4(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  return add_q_4w_linear_node(
      graph,
      args[0], // mat1
      args[1], // mat2
      args[2], // group_size
      args[3], // scales_and_zeros
      // There is an unused variable inner_k_tiles which is used to call
      // _convert_weight_to_int4pack in the AOT custom op, which is why the 4th
      // argument is skipped.
      args[5] // out
  );
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(et_vk.linear_weight_int4.default, linear_weight_int4);
}

} // namespace vkcompute
