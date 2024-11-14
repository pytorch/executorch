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

void check_q_8w_linear_args(
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

void resize_q_8w_linear_node(
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
  ValueRef q_mat2 = prepack_standard(
      graph, q_mat2_data, graph.storage_type_of(out), utils::kWidthPacked);
  ValueRef scales = prepack_standard(
      graph, scales_data, graph.storage_type_of(out), utils::kWidthPacked);

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

  // set global work group size to be 1 dimensional
  const utils::uvec3 wg_size = {
      static_cast<uint32_t>(graph.numel_of(out_W_packed)), 1, 1};

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      wg_size,
      graph.create_local_wg_size(wg_size),
      // Inputs and Outputs
      {{out_W_packed, vkapi::MemoryAccessType::WRITE},
       {{mat1_W_packed, q_mat2, scales}, vkapi::MemoryAccessType::READ}},
      // Shader params buffers
      ubos,
      // Specialization Constants
      {},
      // Resizing Logic
      resize_q_8w_linear_node));
  if (!graph.is_buffer_storage(out) &&
      graph.packed_dim_of(out) != WHCN::kWidthDim) {
    viewFn(graph, {out_W_packed, graph.add_none(), out});
  }
}

void add_q_8w_linear_optimized_node(
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

  utils::StorageType stype = graph.storage_type_of(out);
  ValueRef q_mat2 =
      prepack_standard(graph, q_mat2_data, stype, utils::kWidthPacked);
  ValueRef scales =
      prepack_standard(graph, scales_data, stype, utils::kWidthPacked);

  std::string kernel_name = "q_8w_linear_optimized";
  kernel_name.reserve(kShaderNameReserve);
  add_packed_dim_suffix(kernel_name, graph.packed_dim_of(mat1_W_packed));
  add_packed_dim_suffix(kernel_name, graph.packed_dim_of(q_mat2));
  std::vector<int64_t> mat1_sizes = graph.sizes_of(mat1_W_packed);
  const int mat1_dims = mat1_sizes.size();
  if (mat1_dims == 3) {
    kernel_name = "batch_" + kernel_name;
  }
  if (mat1_sizes.at(mat1_dims - 2) < 8) {
    kernel_name += "_tile_row_2";
  } else {
    kernel_name += "_tile_row_4";
  }

  add_dtype_suffix(kernel_name, graph.dtype_of(out_W_packed));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out_W_packed));

  vkapi::ParamsBindList ubos({});

  utils::uvec3 global_size;
  utils::uvec3 local_size;
  if (graph.is_buffer_storage(out)) {
    ubos.append(
        {graph.sizes_ubo(out_W_packed),
         graph.strides_ubo(out_W_packed),
         graph.numel_ubo(out_W_packed),
         graph.sizes_ubo(mat1_W_packed),
         graph.strides_ubo(mat1_W_packed),
         graph.strides_ubo(q_mat2),
         graph.strides_ubo(scales)});
    global_size = graph.create_global_wg_size(out_W_packed);
    local_size = graph.create_local_wg_size(out_W_packed);
  } else {
    global_size = graph.logical_limits_of(out_W_packed);
    ubos.append(
        {graph.logical_limits_ubo(out_W_packed),
         graph.sizes_ubo(mat1_W_packed)});
    if (mat1_sizes.at(mat1_dims - 2) < 8) {
      global_size = global_size = utils::divup_vec(global_size, {1, 2, 1});
    } else {
      global_size = utils::divup_vec(global_size, {1, 4, 1});
    }
    local_size = {16, 3, 1};
  }

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_size,
      local_size,
      // Inputs and Outputs
      {{out_W_packed, vkapi::MemoryAccessType::WRITE},
       {{mat1_W_packed, q_mat2, scales}, vkapi::MemoryAccessType::READ}},
      // Shader params buffers
      ubos,
      // Specialization Constants
      {}, // spec_vars,
      // Resizing Logic
      resize_q_8w_linear_node));

  if (!graph.is_buffer_storage(out)) {
    viewFn(graph, {out_W_packed, graph.add_none(), out});
  }
}

void weight_int8pack_mm(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  check_q_8w_linear_args(graph, args[0], args[1], args[2], args[3]);
  return add_q_8w_linear_node(graph, args[0], args[1], args[2], args[3]);
}

void check_q_4w_linear_args(
    ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef mat2_data,
    const ValueRef group_size,
    const ValueRef scales_and_zeros,
    const ValueRef out) {
  VK_CHECK_COND(graph.int16_shader_types_enabled());
  VK_CHECK_COND(graph.int8_buffers_enabled());

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

  VK_CHECK_COND(graph.packed_dim_of(mat1) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(out) == WHCN::kWidthDim);

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

void add_q_4w_linear_node(
    ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef mat2_data,
    const ValueRef group_size,
    const ValueRef scales_and_zeros_data,
    const ValueRef out) {
  check_q_4w_linear_args(
      graph, mat1, mat2_data, group_size, scales_and_zeros_data, out);

  utils::StorageType storage_type = graph.storage_type_of(out);

  ValueRef mat2 = prepack_direct_copy_buffer(graph, mat2_data);

  ValueRef scales_and_zeros = prepack_standard(
      graph,
      scales_and_zeros_data,
      graph.storage_type_of(out),
      utils::kWidthPacked);

  std::string kernel_name = "q_4w_linear";
  add_storage_type_suffix(kernel_name, storage_type);
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  const uint32_t group_size_val = graph.extract_scalar<uint32_t>(group_size);

  vkapi::ParamsBindList ubos({});
  ubos.append(graph.logical_limits_ubo(out));
  ubos.append(graph.sizes_ubo(mat1));
  ubos.append(graph.strides_ubo(mat2));
  ubos.append(graph.strides_ubo(scales_and_zeros));

  utils::uvec3 global_wg_size = graph.logical_limits_of(out);
  utils::uvec3 local_wg_size = graph.create_local_wg_size(global_wg_size);

  graph.execute_nodes().emplace_back(new DispatchNode(
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
      resize_q_4w_linear_node,
      {}));
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
  VK_REGISTER_OP(aten._weight_int8pack_mm.default, weight_int8pack_mm);
  VK_REGISTER_OP(et_vk.linear_weight_int4.default, linear_weight_int4);
}

} // namespace vkcompute
