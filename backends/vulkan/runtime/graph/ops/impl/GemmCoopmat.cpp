/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/impl/GemmCoopmat.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/GemmCommon.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/ScalarUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/TensorUtils.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

// ── Linear coopmat ──

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

static utils::uvec3 pick_linear_coopmat_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  const auto out_sizes = graph->sizes_of(out);
  uint32_t M = out_sizes.at(out_sizes.size() - 2);
  uint32_t N = out_sizes.at(out_sizes.size() - 1);
  uint32_t num_tiles_n = utils::div_up(N, kCoopmatTileN);
  uint32_t num_tiles_m = utils::div_up(M, kCoopmatTileM);
  return {num_tiles_n * kCoopmatInvocations, num_tiles_m, 1};
}

static utils::uvec3 pick_linear_coopmat_local_wg_size(
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
  return {kCoopmatInvocations, 1, 1};
}

void add_linear_coopmat_node(
    ComputeGraph& graph,
    const ValueRef input,
    const ValueRef packed_weight,
    const ValueRef packed_bias,
    bool has_bias,
    const ValueRef out,
    int32_t weight_B) {
  (void)weight_B;
  VK_CHECK_COND(graph.packed_dim_of(input) == WHCN::kWidthDim);
  VK_CHECK_COND(graph.packed_dim_of(out) == WHCN::kWidthDim);
  VK_CHECK_COND(
      graph.storage_type_of(out) == utils::kBuffer,
      "linear_coopmat requires buffer storage");

  std::vector<int64_t> out_sizes = graph.sizes_of(out);
  int32_t orig_N = utils::safe_downcast<int32_t>(out_sizes.back());
  ValueRef orig_N_ref = graph.add_scalar(static_cast<int64_t>(orig_N));
  ValueRef has_bias_ref = graph.add_scalar(has_bias);

  std::vector<ValueRef> read_inputs = {input, packed_weight};
  if (has_bias) {
    read_inputs.push_back(packed_bias);
  }

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_linear_coopmat_shader,
      pick_linear_coopmat_global_wg_size,
      pick_linear_coopmat_local_wg_size,
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

// ── Matmul coopmat ──

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

static utils::uvec3 pick_matmul_coopmat_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  const auto out_sizes = graph->sizes_of(out);
  uint32_t M = out_sizes.at(out_sizes.size() - 2);
  uint32_t N = out_sizes.at(out_sizes.size() - 1);
  uint32_t num_tiles_n = utils::div_up(N, kCoopmatTileN);
  uint32_t num_tiles_m = utils::div_up(M, kCoopmatTileM);
  return {num_tiles_n * kCoopmatInvocations, num_tiles_m, 1};
}

static utils::uvec3 pick_matmul_coopmat_local_wg_size(
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
  return {kCoopmatInvocations, 1, 1};
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
      pick_matmul_coopmat_global_wg_size,
      pick_matmul_coopmat_local_wg_size,
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
