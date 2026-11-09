/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/matmul/MatmulCoopmat.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/GemmCommon.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

static vkapi::ShaderInfo pick_matmul_coopmat_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  std::string kernel_name = "matmul_coopmat";
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, graph->dtype_of(out));
  return VK_KERNEL_FROM_STR(kernel_name);
}

static GlobalWorkGrid pick_matmul_coopmat_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  const auto out_sizes = graph->sizes_of(out);
  uint32_t M =
      utils::safe_downcast<uint32_t>(out_sizes.at(out_sizes.size() - 2));
  uint32_t N =
      utils::safe_downcast<uint32_t>(out_sizes.at(out_sizes.size() - 1));
  uint32_t num_tiles_n = utils::div_up(N, kCoopmatTileN);
  uint32_t num_tiles_m = utils::div_up(M, kCoopmatTileM);
  // Each workgroup processes one WG_TILE_M x WG_TILE_N output tile via
  // cooperative-matrix MMAs across its 4 subgroups. We want the dispatch
  // to launch exactly num_tiles_n x num_tiles_m workgroups.
  //
  // The framework computes the group count as
  //   group_count = div_up(gwg, lwg)
  // (see Context.cpp + Command.cpp). With lwg = (kCoopmatInvocations,
  // 1, 1), multiplying num_tiles_n by kCoopmatInvocations cancels the
  // div, yielding group_count.x = num_tiles_n.
  return GlobalWorkGrid(
      {num_tiles_n * kCoopmatInvocations, num_tiles_m, 1u},
      kTiledWorkGrid,
      LocalWorkGroup(kCoopmatInvocations, 1u, 1u));
}

void add_matmul_coopmat_node(
    ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef mat2,
    const ValueRef out) {
  VK_CHECK_COND(graph.packed_dim_of(mat1) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(mat2) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(out) == WHCN::kWidthDim);
  VK_CHECK_COND(
      graph.storage_type_of(out) == utils::kBuffer,
      "matmul_coopmat requires buffer storage");

  ValueRef has_bias_ref = graph.add_scalar(false);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_matmul_coopmat_shader,
      pick_matmul_coopmat_gwg,
      pick_required_lwg,
      // Inputs and Outputs — same binding order as matmul_vec
      {{out, vkapi::kWrite}, {{mat1, mat2}, vkapi::kRead}},
      // Shader params buffers — same UBOs as matmul_vec
      {graph.sizes_ubo(mat1), graph.sizes_ubo(mat2)},
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {has_bias_ref},
      // Resizing Logic
      resize_matmul_tiled_node));
}

} // namespace vkcompute
