/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/linear/LinearCoopmat.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/GemmCommon.h>

#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

static vkapi::ShaderInfo pick_linear_coopmat_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  const ValueRef out = args.at(0).refs.at(0);
  bool has_bias = graph->get_bool(resize_args.at(1));
  std::string kernel_name = has_bias ? "linear_coopmat_bias" : "linear_coopmat";
  kernel_name.reserve(kShaderNameReserve);
  add_dtype_suffix(kernel_name, graph->dtype_of(out));
  return VK_KERNEL_FROM_STR(kernel_name);
}

static GlobalWorkGrid pick_linear_coopmat_gwg(
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

void add_linear_coopmat_node(
    ComputeGraph& graph,
    const ValueRef input,
    const ValueRef packed_weight,
    const ValueRef packed_bias,
    bool has_bias,
    const ValueRef out,
    int32_t weight_B) {
  // weight_B must be 1 — the shader is 2D and dispatch z-dim is hardcoded to
  // 1, so batched weights would only compute the first batch. The
  // is_coopmat_eligible() gate rejects batched outputs upstream; this
  // VK_CHECK_COND turns silent miscompute into a hard fail for any direct
  // caller that bypasses the gate.
  VK_CHECK_COND(
      weight_B == 1, "linear_coopmat does not support batched weights");
  VK_CHECK_COND(graph.packed_dim_of(input) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(out) == WHCN::kWidthDim);
  VK_CHECK_COND(
      graph.storage_type_of(out) == utils::kBuffer,
      "linear_coopmat requires buffer storage");

  std::vector<int64_t> out_sizes = graph.sizes_of(out);
  int32_t orig_N =
      utils::safe_downcast<int32_t>(out_sizes.at(out_sizes.size() - 1));
  ValueRef orig_N_ref = graph.add_scalar(static_cast<int64_t>(orig_N));
  ValueRef has_bias_ref = graph.add_scalar(has_bias);

  std::vector<ValueRef> read_inputs = {input, packed_weight};
  if (has_bias) {
    read_inputs.push_back(packed_bias);
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_linear_coopmat_shader,
      pick_linear_coopmat_gwg,
      pick_required_lwg,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {read_inputs, vkapi::kRead}},
      // Shader params buffers
      {graph.sizes_ubo(input), graph.sizes_ubo(out)},
      // Push Constants
      {},
      // Specialization Constants
      {},
      // Resize Args
      {orig_N_ref, has_bias_ref},
      // Resizing Logic
      resize_linear_node));
}

} // namespace vkcompute
