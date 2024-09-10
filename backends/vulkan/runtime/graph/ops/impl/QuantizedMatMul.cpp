/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/Staging.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

void check_q_matmul_args(
    ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef mat2_data,
    const ValueRef group_size_data,
    const ValueRef scales_and_zeros,
    const ValueRef out) {
  const std::vector<int64_t> mat1_sizes = graph.sizes_of(mat1);
  const std::vector<int64_t> mat2_sizes = graph.sizes_of(mat2_data);
  const std::vector<int64_t> scales_and_zeros_sizes =
      graph.sizes_of(scales_and_zeros);

  const uint32_t group_size = graph.extract_scalar<uint32_t>(group_size_data);

  VK_CHECK_COND(mat1_sizes.size() == 2);
  VK_CHECK_COND(mat1_sizes.size() == mat2_sizes.size());

  VK_CHECK_COND(graph.memory_layout_of(mat1) == utils::kWidthPacked);
  VK_CHECK_COND(graph.memory_layout_of(mat2_data) == utils::kWidthPacked);
  VK_CHECK_COND(
      graph.memory_layout_of(scales_and_zeros) == utils::kWidthPacked);

  if (graph.storage_type_of(out) == utils::kBuffer) {
    VK_CHECK_COND(graph.memory_layout_of(out) == utils::kWidthPacked);
  } else {
    VK_CHECK_COND(graph.memory_layout_of(out) == utils::kChannelsPacked);
  }

  const int mat1_K = utils::val_at(-1, mat1_sizes);
  const int mat2_K = utils::val_at(-1, mat2_sizes) * 2;
  const int N = utils::val_at(-2, mat2_sizes);

  VK_CHECK_COND(mat1_K == mat2_K);

  VK_CHECK_COND(mat2_K % group_size == 0);

  const uint32_t k_groups = mat2_K / group_size;

  VK_CHECK_COND(scales_and_zeros_sizes.size() == 3);
  VK_CHECK_COND(utils::val_at(-1, scales_and_zeros_sizes) == k_groups);
  VK_CHECK_COND(utils::val_at(-2, scales_and_zeros_sizes) == N);
  VK_CHECK_COND(utils::val_at(-3, scales_and_zeros_sizes) == 2);

  // Match https://fburl.com/code/6ostkknm
  std::vector<uint32_t> valid_group_sizes = {32, 64, 128, 256};

  bool is_valid_group_size = false;
  for (auto valid_group_size : valid_group_sizes) {
    if (group_size == valid_group_size) {
      is_valid_group_size = true;
      break;
    }
  }

  VK_CHECK_COND(is_valid_group_size);
}

void resize_q_matmul_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;

  vTensorPtr out = graph->get_tensor(args[0].refs[0]);
  vTensorPtr mat1 = graph->get_tensor(args[1].refs[0]);
  vTensorPtr mat2 = graph->get_tensor(args[1].refs[1]);

  const int out_cols = utils::val_at(-2, mat1->sizes());
  const int out_rows = utils::val_at(-2, mat2->sizes());

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

void add_q_matmul_node(
    ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef mat2_data,
    const ValueRef group_size,
    const ValueRef scales_and_zeros_data,
    const ValueRef out) {
  auto storage_type = graph.storage_type_of(out);

  ValueRef mat2;

  if (storage_type == utils::kBuffer) {
    mat2 = prepack_buffer_if_tensor_ref(graph, mat2_data, utils::kWidthPacked);
  } else {
    mat2 = prepack_if_tensor_ref(graph, mat2_data, utils::kWidthPacked);
  }

  ValueRef scales_and_zeros =
      prepack_if_tensor_ref(graph, scales_and_zeros_data, utils::kWidthPacked);

  std::string kernel_name = "q_4w_linear";

  add_dtype_suffix(kernel_name, graph.dtype_of(out));
  add_storage_type_suffix(kernel_name, storage_type);

  const uint32_t group_size_val = graph.extract_scalar<uint32_t>(group_size);

  vkapi::ParamsBindList ubos({});
  if (storage_type == utils::kBuffer) {
    ubos.append(graph.sizes_ubo(out));
    ubos.append(graph.strides_ubo(out));
    ubos.append(graph.sizes_ubo(mat1));
    ubos.append(graph.strides_ubo(mat1));
    ubos.append(graph.strides_ubo(mat2));
    ubos.append(graph.strides_ubo(scales_and_zeros));
  } else {
    ubos.append(graph.sizes_ubo(out));
    ubos.append(graph.sizes_ubo(mat1));
    ubos.append(graph.strides_ubo(scales_and_zeros));
  }

  auto out_sizes = graph.sizes_of(out);
  uint32_t N = utils::val_at(-1, out_sizes);
  uint32_t M = utils::val_at(-2, out_sizes);

  utils::uvec3 global_wg_size = {N, M, 1};

  utils::uvec3 local_wg_size = adaptive_work_group_size(global_wg_size);

  graph.execute_nodes().emplace_back(new ExecuteNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_wg_size,
      local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::MemoryAccessType::WRITE},
       {{mat1, mat2, scales_and_zeros}, vkapi::MemoryAccessType::READ}},
      // Shader params buffers
      ubos,
      // Specialization Constants
      {SV(group_size_val)},
      // Resizing Logic
      resize_q_matmul_node,
      {}));
}

void int4pack_mm(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  check_q_matmul_args(graph, args[0], args[1], args[2], args[3], args[4]);
  return add_q_matmul_node(
      graph,
      args[0], // mat1
      args[1], // mat2
      args[2], // group_size
      args[3], // scales_and_zeros
      args[4] // out
  );
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten._weight_int4pack_mm.default, int4pack_mm);
}

} // namespace vkcompute
