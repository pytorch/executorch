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
  const int out_rows = utils::val_at(-1, qmat2->sizes());

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
  // Create temporary tensors to store the width packed versions of mat1 and out
  TmpTensor mat1_tmp(
      &graph, graph.sizes_of(mat1), graph.dtype_of(mat1), utils::kWidthPacked);
  TmpTensor out_tmp(
      &graph, graph.sizes_of(out), graph.dtype_of(out), utils::kWidthPacked);
  if (!graph.is_buffer_storage(out) &&
      graph.packed_dim_of(mat1) != WHCN::kWidthDim) {
    // Ensure mat1 is width packed
    mat1_W_packed = mat1_tmp;
    viewFn(graph, {mat1, graph.add_none(), mat1_W_packed});
    // Ensure out is packed correctly
    out_W_packed = out_tmp;
  }
  ValueRef q_mat2 = prepack_standard_hw_transposed(
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

  utils::uvec3 global_wg;
  if (graph.is_buffer_storage(out)) {
    global_wg = {static_cast<uint32_t>(graph.numel_of(out_W_packed)), 1, 1};
  } else {
    global_wg = graph.logical_limits_of(out_W_packed);
  }

  utils::uvec3 local_wg{8, 8, 1};
  int32_t out_W = graph.size_at<int32_t>(-1, out_W_packed);

  if (graph.is_buffer_storage(out_W_packed)) {
    local_wg[0] = 64;
    local_wg[1] = 1;
    local_wg[2] = 1;
  } else {
    if (out_W % 8 != 0) {
      if (out_W % 4 == 0) {
        local_wg[0] = 4;
        local_wg[1] = 16;
      } else {
        local_wg[0] = 2;
        local_wg[1] = 32;
      }
    }
  }

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_wg,
      local_wg,
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

void add_q_8w_linear_tiled_node(
    ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef q_mat2_data,
    const ValueRef scales_data,
    const ValueRef out) {
  utils::StorageType stype = graph.storage_type_of(out);
  ValueRef q_mat2 = prepack_standard_hw_transposed(
      graph, q_mat2_data, stype, utils::kWidthPacked);
  ValueRef scales =
      prepack_standard(graph, scales_data, stype, utils::kWidthPacked);

  std::string kernel_name = "q_8w_linear_tiled";
  kernel_name.reserve(kShaderNameReserve);
  std::vector<int64_t> mat1_sizes = graph.sizes_of(mat1);
  const int64_t M = utils::val_at(-2, mat1_sizes);
  int out_tile_nrows = 4;
  if (M % 6 == 0) {
    kernel_name += "_o4x6";
    out_tile_nrows = 6;
  } else {
    kernel_name += "_o4x4";
    out_tile_nrows = 4;
  }

  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  utils::uvec3 global_wg_size = graph.logical_limits_of(out);
  global_wg_size[1] = global_wg_size[1] / out_tile_nrows;

  utils::uvec3 local_wg_size{64, 1, 1};

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_wg_size,
      local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{mat1, q_mat2, scales}, vkapi::kRead}},
      // Shader params buffers
      {},
      // Specialization Constants
      {},
      // Resizing Logic
      resize_q_8w_linear_node,
      {},
      // Push Constants
      {{graph.sizes_pc_of(out), graph.sizes_pc_of(mat1)}}));
}

bool can_use_tiled_impl(
    ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef q_mat2_data,
    const ValueRef scales_data,
    const ValueRef out) {
  (void)q_mat2_data;
  (void)scales_data;

  // Check if mat1 is not a 3D tensor or that batches = 1
  // TODO(ssjia): Add support for batches in the tiled impl
  if (graph.dim_of(mat1) == 3 && graph.size_at<int>(-1, mat1) != 1) {
    return false;
  }
  // Check that K is a multiple of 4
  if (graph.size_at<int>(-1, mat1) % 4 != 0) {
    return false;
  }
  // Check that M is a multiple of 4 or 6
  if (graph.size_at<int>(-2, mat1) % 4 != 0 &&
      graph.size_at<int>(-2, mat1) % 6 != 0) {
    return false;
  }
  // Check that the storage type is texture
  // TODO(ssjia): Add support for buffer storage in the tiled impl
  if (graph.storage_type_of(out) != utils::kTexture3D) {
    return false;
  }
  // Check that the packed dim is the width dim
  if (graph.packed_dim_of(mat1) != WHCN::kWidthDim) {
    return false;
  }
  // Check that no special axis mapping is used for the input
  // TODO(ssjia): Add support for non-standard axis mapping in the tiled impl
  if (!graph.has_standard_axis_map(mat1)) {
    return false;
  }
  // Check that no special axis mapping is used for the output
  // TODO(ssjia): Add support for non-standard axis mapping in the tiled impl
  if (!graph.has_standard_axis_map(out)) {
    return false;
  }

  return true;
}

void weight_int8pack_mm(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  check_q_8w_linear_args(graph, args[0], args[1], args[2], args[3]);
  if (can_use_tiled_impl(graph, args[0], args[1], args[2], args[3])) {
    return add_q_8w_linear_tiled_node(
        graph, args[0], args[1], args[2], args[3]);
  }
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

  ValueRef mat1_W_packed = mat1;
  ValueRef out_W_packed = out;
  auto viewFn = VK_GET_OP_FN("aten.view_copy.default");
  // Create temporary tensors to store the width packed versions of mat1 and out
  TmpTensor mat1_tmp(
      &graph, graph.sizes_of(mat1), graph.dtype_of(mat1), utils::kWidthPacked);
  TmpTensor out_tmp(
      &graph, graph.sizes_of(out), graph.dtype_of(out), utils::kWidthPacked);
  if (storage_type == utils::kTexture3D) {
    if (!graph.is_buffer_storage(out) &&
        graph.packed_dim_of(mat1) != WHCN::kWidthDim) {
      // Ensure mat1 is width packed
      mat1_W_packed = mat1_tmp;
      viewFn(graph, {mat1, graph.add_none(), mat1_W_packed});
      // Ensure out is packed correctly
      out_W_packed = out_tmp;
    }
  }

  vkapi::ParamsBindList ubos({});
  ubos.append(graph.logical_limits_ubo(out_W_packed));
  ubos.append(graph.sizes_ubo(mat1_W_packed));
  ubos.append(graph.strides_ubo(mat2));
  ubos.append(graph.strides_ubo(scales_and_zeros));

  utils::uvec3 global_wg_size = graph.logical_limits_of(out_W_packed);
  utils::uvec3 local_wg_size = graph.create_local_wg_size(global_wg_size);

  graph.execute_nodes().emplace_back(new DispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      global_wg_size,
      local_wg_size,
      // Inputs and Outputs
      {{out_W_packed, vkapi::MemoryAccessType::WRITE},
       {{mat1_W_packed, mat2, scales_and_zeros},
        vkapi::MemoryAccessType::READ}},
      // Shader params buffers
      ubos,
      // Specialization Constants
      {SV(group_size_val)},
      // Resizing Logic
      resize_q_4w_linear_node,
      {}));
  if (!graph.is_buffer_storage(out) &&
      graph.packed_dim_of(out) != WHCN::kWidthDim) {
    viewFn(graph, {out_W_packed, graph.add_none(), out});
  }
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
