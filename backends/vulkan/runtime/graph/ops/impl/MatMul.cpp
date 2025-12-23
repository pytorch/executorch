/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/MatMul.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/ScalarUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void check_matmul_args(
    const ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef mat2_data,
    const ValueRef out) {
  std::vector<int64_t> mat1_sizes = graph.sizes_of(mat1);
  std::vector<int64_t> mat2_sizes = graph.sizes_of(mat2_data);

  VK_CHECK_COND(mat1_sizes.size() == 2 || mat1_sizes.size() == 3);
  VK_CHECK_COND(mat1_sizes.size() == mat2_sizes.size());

  VK_CHECK_COND(graph.packed_dim_of(mat1) == graph.packed_dim_of(out));

  VK_CHECK_COND(utils::val_at(-1, mat1_sizes) == utils::val_at(-2, mat2_sizes));
}

void resize_matmul_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef mat1 = args.at(1).refs.at(0);
  const ValueRef mat2 = args.at(1).refs.at(1);

  bool mat2_is_transposed = graph->get_bool(resize_args.at(0));

  const std::vector<int64_t> mat1_sizes = graph->sizes_of(mat1);
  const std::vector<int64_t> mat2_sizes = graph->sizes_of(mat2);

  const int out_cols = utils::val_at(-2, mat1_sizes);
  const int out_rows = mat2_is_transposed ? utils::val_at(-2, mat2_sizes)
                                          : utils::val_at(-1, mat2_sizes);

  const int64_t out_dim = graph->dim_of(out);
  std::vector<int64_t> new_out_sizes(mat1_sizes);
  new_out_sizes.at(out_dim - 1) = out_rows;
  new_out_sizes.at(out_dim - 2) = out_cols;

  graph->virtual_resize(out, new_out_sizes);
}

/**
 * Custom global workgroup size function for naive buffer matmul operations.
 */
utils::uvec3 matmul_naive_buffer_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  return {
      graph->size_at<uint32_t>(-1, out),
      graph->size_at<uint32_t>(-2, out),
      graph->size_at<uint32_t>(-3, out) * graph->size_at<uint32_t>(-4, out)};
}

void add_matmul_naive_buffer_node(
    ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef mat2_data,
    const ValueRef out,
    const ValueRef mat2_is_transposed) {
  ValueRef mat2 = prepack_standard(
      graph,
      mat2_data,
      graph.storage_type_of(out),
      utils::kHeightPacked,
      /*passthrough = */ true);

  std::string kernel_name = "matmul_naive_buffer";
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  int mat2_is_transposed_val = (mat2_is_transposed != kDummyValueRef &&
                                graph.get_bool(mat2_is_transposed))
      ? 1
      : 0;

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      matmul_naive_buffer_global_wg_size,
      pick_hw_square_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{mat1, mat2}, vkapi::kRead}},
      // Shader params buffers
      {
          graph.sizes_ubo(out),
          graph.strides_ubo(out),
          graph.sizes_ubo(mat1),
          graph.strides_ubo(mat1),
          graph.sizes_ubo(mat2),
          graph.strides_ubo(mat2),
          graph.numel_ubo(out),
      },
      // Push Constants
      {},
      // Specialization Constants
      {mat2_is_transposed_val},
      // Resize Args
      {mat2_is_transposed},
      // Resizing Logic
      resize_matmul_node));
}

vkapi::ShaderInfo pick_matmul_naive_texture3d_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const bool is_transposed = graph->get_bool(resize_args.at(0));

  std::string kernel_name =
      is_transposed ? "matmul_transposed_naive" : "matmul_naive";
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph->storage_type_of(out));
  add_dtype_suffix(kernel_name, graph->dtype_of(out));

  return VK_KERNEL_FROM_STR(kernel_name);
}

void add_matmul_naive_texture3d_node(
    ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef mat2_data,
    const ValueRef out,
    const ValueRef mat2_is_transposed) {
  ValueRef mat2 = prepack_standard(
      graph,
      mat2_data,
      graph.storage_type_of(out),
      utils::kHeightPacked,
      /*passthrough = */ true);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_matmul_naive_texture3d_shader,
      default_pick_global_wg_size,
      pick_hw_square_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{mat1, mat2}, vkapi::kRead}},
      // Shader params buffers
      {},
      // Push Constants
      {graph.sizes_pc_of(out),
       graph.sizes_pc_of(mat1),
       graph.sizes_pc_of(mat2),
       graph.logical_limits_pc_of(out)},
      // Specialization Constants
      {graph.hashed_layout_of(out),
       graph.hashed_layout_of(mat1),
       graph.hashed_layout_of(mat2)},
      // Resize Args
      {mat2_is_transposed},
      // Resizing Logic
      resize_matmul_node));
}

vkapi::ShaderInfo pick_matmul_optimized_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef mat1_W_packed = resize_args.at(1);
  const bool mat2_is_transposed_val = graph->get_bool(resize_args.at(0));

  std::string kernel_name = mat2_is_transposed_val
      ? "matmul_transposed_optimized"
      : "matmul_optimized";

  std::vector<int64_t> mat1_sizes = graph->sizes_of(mat1_W_packed);
  size_t mat1_dims = mat1_sizes.size();
  if (mat1_dims == 3) {
    kernel_name = "batch_" + kernel_name;
  }
  if (mat1_sizes.at(mat1_dims - 2) < 8) {
    kernel_name += "_tile_row_2";
  } else {
    kernel_name += "_tile_row_4";
  }

  add_dtype_suffix(kernel_name, graph->dtype_of(out));

  return VK_KERNEL_FROM_STR(kernel_name);
}

utils::uvec3 matmul_optimized_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;

  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef mat1_W_packed = resize_args.at(1);

  const std::vector<int64_t> mat1_sizes = graph->sizes_of(mat1_W_packed);
  const size_t mat1_dims = mat1_sizes.size();

  utils::uvec3 global_size = graph->logical_limits_of(out);
  if (mat1_sizes.at(mat1_dims - 2) < 8) {
    // Use `logical_extents` instead of `image_extents` because the workgroup
    // axes need to correspond to tensor dimensions.
    global_size = utils::divup_vec(global_size, {4, 2, 1});
  } else {
    global_size = utils::divup_vec(global_size, {4, 4, 1});
  }

  return global_size;
}

void add_matmul_optimized_node(
    ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef mat2_data,
    const ValueRef out,
    const ValueRef mat2_is_transposed) {
  ValueRef mat2 = prepack_standard(
      graph,
      mat2_data,
      graph.storage_type_of(out),
      utils::kHeightPacked,
      /*passthrough = */ true);

  // Ensure mat1 is width packed
  TmpTensor mat1_tmp(
      &graph, graph.sizes_of(mat1), graph.dtype_of(mat1), utils::kWidthPacked);
  ValueRef mat1_W_packed = mat1;
  auto viewFn = VK_GET_OP_FN("aten.view_copy.default");
  if (graph.packed_dim_of(mat1) != WHCN::kWidthDim) {
    mat1_W_packed = mat1_tmp;
    viewFn(graph, {mat1, graph.add_none(), mat1_W_packed});
  }

  const bool mat2_is_transposed_val = graph.get_bool(mat2_is_transposed);

  // Ensure mat2 to height packed
  ValueRef mat2_packed = mat2;
  const utils::GPUMemoryLayout mat2_layout =
      mat2_is_transposed_val ? utils::kWidthPacked : utils::kHeightPacked;
  TmpTensor mat2_tmp(
      &graph, graph.sizes_of(mat2), graph.dtype_of(mat2), mat2_layout);
  if (graph.estimate_memory_layout_of(mat2) != mat2_layout) {
    mat2_packed = mat2_tmp;
    viewFn(graph, {mat2, graph.add_none(), mat2_packed});
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_matmul_optimized_shader,
      matmul_optimized_global_wg_size,
      pick_hw_square_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{mat1_W_packed, mat2_packed}, vkapi::kRead}},
      // Shader params buffers
      {
          graph.sizes_ubo(out),
          graph.sizes_ubo(mat1_W_packed),
          graph.sizes_ubo(mat2_packed),
      },
      // Push Constants
      {},
      // Specialization Constants
      {graph.hashed_layout_of(out),
       graph.hashed_layout_of(mat1_W_packed),
       graph.hashed_layout_of(mat2_packed)},
      // Resize Args
      {mat2_is_transposed, mat1_W_packed},
      // Resizing Logic
      resize_matmul_node));
}

void add_matmul_node(
    ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef mat2_data,
    const ValueRef out,
    const ValueRef mat2_is_transposed) {
  if (graph.is_buffer_storage(out)) {
    add_matmul_naive_buffer_node(
        graph, mat1, mat2_data, out, mat2_is_transposed);
  } else if (graph.packed_dim_of(mat1) == WHCN::kChannelsDim) {
    add_matmul_optimized_node(graph, mat1, mat2_data, out, mat2_is_transposed);
  } else if (graph.packed_dim_of(mat1) == WHCN::kWidthDim) {
    add_matmul_naive_texture3d_node(
        graph, mat1, mat2_data, out, mat2_is_transposed);
  } else {
    VK_THROW("Input texture should be channel packed or width packed.");
  }
}

void matmul(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  check_matmul_args(graph, args[0], args[1], args[2]);
  const ValueRef mat2_is_transposed = graph.add_scalar(false);
  return add_matmul_node(graph, args[0], args[1], args[2], mat2_is_transposed);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten.mm.default, matmul);
  VK_REGISTER_OP(aten.bmm.default, matmul);
}

} // namespace vkcompute
