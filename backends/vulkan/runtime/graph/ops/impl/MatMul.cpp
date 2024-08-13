/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

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

  VK_CHECK_COND(graph.memory_layout_of(mat1) == graph.memory_layout_of(out));

  VK_CHECK_COND(utils::val_at(-1, mat1_sizes) == utils::val_at(-2, mat2_sizes));
}

void resize_matmul_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  vTensorPtr out = graph->get_tensor(args[0].refs[0]);
  vTensorPtr mat1 = graph->get_tensor(args[1].refs[0]);
  vTensorPtr mat2 = graph->get_tensor(args[1].refs[1]);

  bool mat2_is_transposed = graph->get_bool(extra_args[0]);

  const int out_cols = utils::val_at(-2, mat1->sizes());
  const int out_rows = mat2_is_transposed ? utils::val_at(-2, mat2->sizes())
                                          : utils::val_at(-1, mat2->sizes());

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

void add_matmul_naive_node(
    ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef mat2_data,
    const ValueRef out,
    const ValueRef mat2_is_transposed) {
  ValueRef mat2 = prepack_if_tensor_ref(graph, mat2_data, utils::kHeightPacked);

  std::string kernel_name = graph.get_bool(mat2_is_transposed)
      ? "matmul_transposed_naive"
      : "matmul_naive";
  kernel_name.reserve(kShaderNameReserve);
  add_memory_layout_suffix(kernel_name, graph.memory_layout_of(mat1));
  add_memory_layout_suffix(kernel_name, graph.memory_layout_of(mat2));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      graph.create_global_wg_size(out),
      graph.create_local_wg_size(out),
      // Inputs and Outputs
      {{out, vkapi::MemoryAccessType::WRITE},
       {{mat1, mat2}, vkapi::MemoryAccessType::READ}},
      // Shader params buffers
      {
          graph.texture_limits_ubo(out),
          graph.sizes_ubo(mat1),
      },
      // Specialization Constants
      {},
      // Resizing Logic
      resize_matmul_node,
      {mat2_is_transposed}));
}

void add_matmul_optimized_node(
    ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef mat2_data,
    const ValueRef out,
    const ValueRef mat2_is_transposed) {
  ValueRef mat2 = prepack_if_tensor_ref(graph, mat2_data, utils::kHeightPacked);

  // Ensure mat1 is width packed
  ValueRef mat1_W_packed = graph.add_tensor_like(mat1, utils::kWidthPacked);
  auto viewFn = VK_GET_OP_FN("aten.view_copy.default");
  viewFn(graph, {mat1, graph.add_none(), mat1_W_packed});

  const bool mat2_is_transposed_val = graph.get_bool(mat2_is_transposed);

  // Ensure mat2 to height packed
  ValueRef mat2_packed = mat2;
  const utils::GPUMemoryLayout mat2_layout =
      mat2_is_transposed_val ? utils::kWidthPacked : utils::kHeightPacked;
  if (graph.memory_layout_of(mat2) != mat2_layout) {
    mat2_packed = graph.add_tensor_like(mat2, mat2_layout);
    viewFn(graph, {mat2, graph.add_none(), mat2_packed});
  }

  std::string kernel_name = mat2_is_transposed_val
      ? "matmul_transposed_optimized"
      : "matmul_optimized";

  std::vector<int64_t> mat1_sizes = graph.sizes_of(mat1_W_packed);
  int mat1_dims = mat1_sizes.size();
  if (mat1_dims == 3) {
    kernel_name = "batch_" + kernel_name;
  }
  if (mat1_sizes.at(mat1_dims - 2) < 8) {
    kernel_name += "_tile_row_2";
  } else {
    kernel_name += "_tile_row_4";
  }

  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  utils::uvec3 global_size;
  if (mat1_sizes.at(mat1_dims - 2) < 8) {
    global_size = utils::divup_vec(graph.image_extents_of(out), {4, 2, 1});
  } else {
    global_size = utils::divup_vec(graph.image_extents_of(out), {4, 4, 1});
  }
  utils::uvec3 local_size = adaptive_work_group_size(global_size);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_size,
      local_size,
      // Inputs and Outputs
      {{out, vkapi::MemoryAccessType::WRITE},
       {{mat1_W_packed, mat2_packed}, vkapi::MemoryAccessType::READ}},
      // Shader params buffers
      {
          graph.texture_limits_ubo(out),
          graph.sizes_ubo(out),
          graph.texture_limits_ubo(mat1_W_packed),
      },
      // Specialization Constants
      {},
      // Resizing Logic
      resize_matmul_node,
      {mat2_is_transposed}));
}

void add_matmul_node(
    ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef mat2_data,
    const ValueRef out,
    const ValueRef mat2_is_transposed) {
  if (graph.memory_layout_of(mat1) == utils::kChannelsPacked) {
    add_matmul_optimized_node(graph, mat1, mat2_data, out, mat2_is_transposed);
  } else if (graph.memory_layout_of(mat1) == utils::kWidthPacked) {
    add_matmul_naive_node(graph, mat1, mat2_data, out, mat2_is_transposed);
  } else {
    VK_THROW("Input should be channel packed or width packed.");
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
