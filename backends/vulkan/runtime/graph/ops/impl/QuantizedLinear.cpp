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

void check_qlinear_args(
    const ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef qmat2_data,
    const ValueRef scales,
    const ValueRef out) {
  std::vector<int64_t> mat1_sizes = graph.sizes_of(mat1);
  std::vector<int64_t> qmat2_sizes = graph.sizes_of(qmat2_data);
  std::vector<int64_t> scales_sizes = graph.sizes_of(scales);

  VK_CHECK_COND(mat1_sizes.size() == 2 || mat1_sizes.size() == 3);
  VK_CHECK_COND(qmat2_sizes.size() == 2);
  VK_CHECK_COND(scales_sizes.size() == 1);

  VK_CHECK_COND(graph.packed_dim_of(mat1) == graph.packed_dim_of(out));

  VK_CHECK_COND(
      utils::val_at(-1, mat1_sizes) == utils::val_at(-1, qmat2_sizes));
  VK_CHECK_COND(
      utils::val_at(-1, scales_sizes) == utils::val_at(-2, qmat2_sizes));
}

void resize_qlinear_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;

  vTensorPtr out = graph->get_tensor(args[0].refs[0]);
  vTensorPtr mat1 = graph->get_tensor(args[1].refs[0]);
  vTensorPtr qmat2 = graph->get_tensor(args[1].refs[1]);

  const int out_cols = utils::val_at(-2, mat1->sizes());
  const int out_rows = utils::val_at(-2, qmat2->sizes());

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

void add_q_8w_linear_node(
    ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef q_mat2_data,
    const ValueRef scales_data,
    const ValueRef out) {
  auto viewFn = VK_GET_OP_FN("aten.view_copy.default");
  ValueRef mat1_W_packed = mat1;
  ValueRef out_W_packed = out;
  if (!graph.is_buffer_storage(out) &&
      graph.packed_dim_of(mat1) != WHCN::kWidthDim) {
    // Ensure mat1 is width packed
    mat1_W_packed = graph.add_tensor_like(mat1, utils::kWidthPacked);
    viewFn(graph, {mat1, graph.add_none(), mat1_W_packed});
    // Ensure out is packed correctly
    out_W_packed = graph.add_tensor_like(out, utils::kWidthPacked);
  }
  ValueRef q_mat2 =
      prepack_if_tensor_ref(graph, q_mat2_data, utils::kWidthPacked);
  ValueRef scales =
      prepack_if_tensor_ref(graph, scales_data, utils::kWidthPacked);

  std::string kernel_name = "q_8w_linear";
  kernel_name.reserve(kShaderNameReserve);
  add_packed_dim_suffix(kernel_name, graph.packed_dim_of(mat1_W_packed));
  add_packed_dim_suffix(kernel_name, graph.packed_dim_of(q_mat2));
  add_dtype_suffix(kernel_name, graph.dtype_of(out_W_packed));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out_W_packed));

  vkapi::ParamsBindList ubos({});
  if (graph.is_buffer_storage(out_W_packed)) {
    ubos.append(
        {graph.sizes_ubo(out_W_packed),
         graph.strides_ubo(out_W_packed),
         graph.numel_ubo(out_W_packed),
         graph.sizes_ubo(mat1_W_packed),
         graph.strides_ubo(mat1),
         graph.strides_ubo(q_mat2),
         graph.strides_ubo(scales)});
  } else {
    ubos.append(
        {graph.logical_limits_ubo(out_W_packed),
         graph.sizes_ubo(mat1_W_packed)});
  }

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      graph.create_global_wg_size(out_W_packed),
      graph.create_local_wg_size(out_W_packed),
      // Inputs and Outputs
      {{out_W_packed, vkapi::MemoryAccessType::WRITE},
       {{mat1_W_packed, q_mat2, scales}, vkapi::MemoryAccessType::READ}},
      // Shader params buffers
      ubos,
      // Specialization Constants
      {},
      // Resizing Logic
      resize_qlinear_node));
  if (!graph.is_buffer_storage(out) &&
      graph.packed_dim_of(out) != WHCN::kWidthDim) {
    viewFn(graph, {out_W_packed, graph.add_none(), out});
  }
}

void weight_int8pack_mm(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  check_qlinear_args(graph, args[0], args[1], args[2], args[3]);
  return add_q_8w_linear_node(graph, args[0], args[1], args[2], args[3]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten._weight_int8pack_mm.default, weight_int8pack_mm);
}

} // namespace vkcompute
