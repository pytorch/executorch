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

// Custom global workgroup size function for linear_qcs8w
utils::uvec3 linear_qcs8w_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  return {static_cast<uint32_t>(graph->numel_of(out)), 1, 1};
}

// Custom local workgroup size function for linear_qcs8w
utils::uvec3 linear_qcs8w_local_wg_size(
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
  return {64, 1, 1};
}

// Custom global workgroup size function for linear_qcsnw_tiled
utils::uvec3 linear_qcsnw_tiled_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef mat1 = args.at(1).refs.at(0);

  // Determine quantization bits from shader name
  int quant_nbits = 8;
  if (shader.kernel_name.find("qcs4w") != std::string::npos) {
    quant_nbits = 4;
  }

  std::vector<int64_t> mat1_sizes = graph->sizes_of(mat1);
  const int64_t M = utils::val_at(-2, mat1_sizes);
  uint32_t out_tile_nrows = 4;
  if (M % 6 == 0) {
    out_tile_nrows = 2;
  } else if (M % 4 == 0) {
    out_tile_nrows = 4;
  } else if (M % 1 == 0) {
    out_tile_nrows = 1;
  } else {
    out_tile_nrows = 4;
  }

  // Number of output texels in the output tile
  uint32_t out_tile_ntxcols = 2;
  if (quant_nbits == 4) {
    out_tile_ntxcols = 2;
  }

  utils::uvec3 out_limits = graph->logical_limits_of(out);
  uint32_t global_wg_x = utils::div_up(out_limits[0], out_tile_ntxcols);
  return {
      global_wg_x * (utils::div_up(out_limits[1], out_tile_nrows)),
      1,
      out_limits[2]};
}

// Custom local workgroup size function for linear_qcsnw_tiled
utils::uvec3 linear_qcsnw_tiled_local_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)graph;
  (void)global_workgroup_size;
  (void)args;
  (void)resize_args;

  // Check if using cooperative algorithm from shader name
  bool use_coop_algorithm =
      shader.kernel_name.find("_coop") != std::string::npos;

  if (use_coop_algorithm) {
    return {8, 1, 8};
  } else {
    return {64, 1, 1};
  }
}

void check_linear_qcsnw_args(
    const ComputeGraph& graph,
    const int quant_nbits,
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

  if (quant_nbits == 4) {
    VK_CHECK_COND(
        utils::val_at(-1, mat1_sizes) == utils::val_at(-1, qmat2_sizes) * 2);
    VK_CHECK_COND(
        utils::val_at(-1, scales_sizes) == utils::val_at(-2, qmat2_sizes));
  } else {
    VK_CHECK_COND(
        utils::val_at(-1, mat1_sizes) == utils::val_at(-1, qmat2_sizes));
    VK_CHECK_COND(
        utils::val_at(-1, scales_sizes) == utils::val_at(-2, qmat2_sizes));
  }

  if (graph.is_buffer_storage(out)) {
    VK_CHECK_COND(graph.is_contiguous(out));
  }
}

void resize_linear_qcsnw_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args) {
  (void)extra_args;

  const ValueRef out = args.at(0).refs.at(0);
  const ValueRef mat1 = args.at(1).refs.at(0);
  const ValueRef qmat2 = args.at(1).refs.at(1);

  const std::vector<int64_t> mat1_sizes = graph->sizes_of(mat1);
  const std::vector<int64_t> qmat2_sizes = graph->sizes_of(qmat2);

  const int out_cols = utils::val_at(-2, mat1_sizes);
  int out_rows = utils::val_at(-1, qmat2_sizes);
  // Byte dtype suggests 4-bit quantization in which case the weight tensor is
  // packed with 2 values per byte.
  if (graph->dtype_of(qmat2) == vkapi::kByte) {
    out_rows *= 2;
  }

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

void add_linear_qcs8w_node(
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

  std::string kernel_name = "linear_qcs8w";
  kernel_name.reserve(kShaderNameReserve);
  add_packed_dim_suffix(kernel_name, graph.packed_dim_of(mat1_W_packed));
  add_packed_dim_suffix(kernel_name, graph.packed_dim_of(q_mat2));
  add_dtype_suffix(kernel_name, graph.dtype_of(out_W_packed));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out_W_packed));

  std::vector<PushConstantDataInfo> pcs;
  if (graph.is_buffer_storage(out_W_packed)) {
    pcs = {
        graph.sizes_pc_of(out_W_packed),
        graph.strides_pc_of(out_W_packed),
        graph.sizes_pc_of(mat1_W_packed),
        graph.strides_pc_of(mat1),
        graph.strides_pc_of(q_mat2),
        graph.strides_pc_of(scales),
        graph.numel_pc_of(out_W_packed)};
  } else {
    pcs = {
        graph.logical_limits_pc_of(out_W_packed),
        graph.sizes_pc_of(mat1_W_packed),
        graph.sizes_pc_of(q_mat2)};
  }

  const utils::uvec3 global_wg = {
      static_cast<uint32_t>(graph.numel_of(out_W_packed)), 1, 1};
  const utils::uvec3 local_wg{64, 1, 1};

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      linear_qcs8w_global_wg_size,
      linear_qcs8w_local_wg_size,
      // Inputs and Outputs
      {{out_W_packed, vkapi::MemoryAccessType::WRITE},
       {{mat1_W_packed, q_mat2, scales}, vkapi::MemoryAccessType::READ}},
      // Shader params buffers
      {},
      // Push Constants
      pcs,
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      resize_linear_qcsnw_node));
  if (!graph.is_buffer_storage(out) &&
      graph.packed_dim_of(out) != WHCN::kWidthDim) {
    viewFn(graph, {out_W_packed, graph.add_none(), out});
  }
}

void add_linear_qcsnw_tiled_node(
    ComputeGraph& graph,
    const bool use_coop_algorithm,
    const int quant_nbits,
    const ValueRef mat1,
    const ValueRef q_mat2_data,
    const ValueRef scales_data,
    const ValueRef out) {
  uint32_t max_extent = graph.context()->adapter_ptr()->max_texture2d_dim();
  std::vector<int64_t> qmat2_orig_sizes = graph.sizes_of(q_mat2_data);
  const int64_t ndim = graph.dim_of(q_mat2_data);
  const int64_t K = qmat2_orig_sizes.at(ndim - 1);
  const int64_t N = qmat2_orig_sizes.at(ndim - 2);

  ValueRef q_mat2;
  if (quant_nbits == 4) {
    q_mat2 =
        prepack_int4_linear_weight_transposed_interleaved(graph, q_mat2_data);
  } else {
    utils::StorageType q_mat2_storage = utils::kTexture2D;
    if (N > max_extent * 4 || K > max_extent) {
      q_mat2_storage = utils::kBuffer;
    }

    q_mat2 = prepack_standard_hw_transposed(
        graph, q_mat2_data, q_mat2_storage, utils::kWidthPacked);
  }

  utils::StorageType scales_storage = utils::kTexture2D;
  if (N > max_extent) {
    scales_storage = utils::kBuffer;
  }
  ValueRef scales =
      prepack_standard(graph, scales_data, scales_storage, utils::kWidthPacked);

  std::string kernel_name;
  if (quant_nbits == 4) {
    kernel_name =
        use_coop_algorithm ? "linear_qcs4w_coop" : "linear_qcs4w_tiled";
  } else {
    kernel_name =
        use_coop_algorithm ? "linear_qcs8w_coop" : "linear_qcs8w_tiled";
  }
  kernel_name.reserve(kShaderNameReserve);
  add_storage_type_suffix(kernel_name, graph.storage_type_of(out));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(mat1));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(q_mat2));
  add_storage_type_suffix(kernel_name, graph.storage_type_of(scales));
  add_dtype_suffix(kernel_name, graph.dtype_of(out));

  std::vector<int64_t> mat1_sizes = graph.sizes_of(mat1);
  const int64_t M = utils::val_at(-2, mat1_sizes);
  uint32_t out_tile_nrows = 4;
  if (M % 6 == 0) {
    kernel_name += "_o4x2";
    out_tile_nrows = 2;
  } else if (M % 4 == 0) {
    kernel_name += "_o4x4";
    out_tile_nrows = 4;
  } else if (M % 1 == 0) {
    kernel_name += "_o4x1";
    out_tile_nrows = 1;
  } else {
    kernel_name += "_o4x4";
    out_tile_nrows = 4;
  }

  // Number of output texels in the output tile
  uint32_t out_tile_ntxcols = 2;
  if (quant_nbits == 4) {
    out_tile_ntxcols = 2;
  }

  utils::uvec3 out_limits = graph.logical_limits_of(out);
  uint32_t global_wg_x = utils::div_up(out_limits[0], out_tile_ntxcols);
  utils::uvec3 global_wg_size = {
      global_wg_x * (utils::div_up(out_limits[1], out_tile_nrows)),
      1,
      out_limits[2]};

  utils::uvec3 local_wg_size{64, 1, 1};
  if (use_coop_algorithm) {
    local_wg_size = {8, 1, 8};
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      VK_KERNEL_FROM_STR(kernel_name),
      linear_qcsnw_tiled_global_wg_size,
      linear_qcsnw_tiled_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{mat1, q_mat2, scales}, vkapi::kRead}},
      // Shader params buffers
      {},
      // Push Constants
      {{graph.sizes_pc_of(out),
        graph.sizes_pc_of(mat1),
        graph.sizes_pc_of(q_mat2)}},
      // Specialization Constants
      {},
      // Resize Args
      {},
      // Resizing Logic
      resize_linear_qcsnw_node));
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
  if (graph.dim_of(mat1) == 3 && graph.size_at<int>(0, mat1) != 1) {
    return false;
  }
  // Check that K is a multiple of 4
  if (graph.size_at<int>(-1, mat1) % 4 != 0) {
    return false;
  }
  // Check that N is a multiple of 4
  if (graph.size_at<int>(-1, out) % 4 != 0) {
    return false;
  }
  // Check that the packed dim is the width dim
  if (graph.packed_dim_of(mat1) != WHCN::kWidthDim &&
      graph.packed_dim_of(out) != WHCN::kWidthDim) {
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

bool can_use_coop_impl(ComputeGraph& graph, const ValueRef mat1) {
  // Do not use coop algorithm for Adreno 702; manual experimentation shows that
  // it performs worse than the tiled algorithm.
  // TODO(ssjia): Determine a more robust heuristic to determine when the coop
  // algorithm should be used, instead of depending on specific device identity.
  if (graph.device_is_adreno() && graph.device_name_contains("702")) {
    return false;
  }
  // Check that the computation is vector * matrix
  return (graph.size_at<int>(-2, mat1) == 1);
}

void weight_int8pack_mm(
    ComputeGraph& graph,
    const std::vector<ValueRef>& args) {
  check_linear_qcsnw_args(graph, 8, args[0], args[1], args[2], args[3]);
  if (can_use_tiled_impl(graph, args[0], args[1], args[2], args[3])) {
    bool use_coop_algorithm = can_use_coop_impl(graph, args[0]);
    return add_linear_qcsnw_tiled_node(
        graph, use_coop_algorithm, 8, args[0], args[1], args[2], args[3]);
  }
  return add_linear_qcs8w_node(graph, args[0], args[1], args[2], args[3]);
}

void linear_qcs4w(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  check_linear_qcsnw_args(graph, 4, args[0], args[1], args[2], args[3]);

  VK_CHECK_COND(can_use_tiled_impl(graph, args[0], args[1], args[2], args[3]));
  bool use_coop_algorithm = can_use_coop_impl(graph, args[0]);
  return add_linear_qcsnw_tiled_node(
      graph, use_coop_algorithm, 4, args[0], args[1], args[2], args[3]);
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(aten._weight_int8pack_mm.default, weight_int8pack_mm);
  VK_REGISTER_OP(et_vk.linear_qcs4w.default, linear_qcs4w);
}

} // namespace vkcompute
