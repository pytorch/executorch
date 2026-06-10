/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/vulkan/runtime/graph/ops/OperatorRegistry.h>

#include <executorch/backends/vulkan/runtime/graph/ops/impl/Common.h>
#include <executorch/backends/vulkan/runtime/graph/ops/utils/ShaderNameUtils.h>

namespace vkcompute {

// Dispatch for the ported NVIDIA double-buffered coopmat GEMM reference
// (coopmat_mm_ref.glsl): D[M,N] = A[M,K] x B[K,N], all fp16 buffers,
// row-major. One workgroup per 128x128 output tile, 256 threads, subgroup
// size forced to 32 by the shader's REQUIRED_SUBGROUP_SIZE annotation.

constexpr uint32_t kRefTileM = 128;
constexpr uint32_t kRefTileN = 128;
constexpr uint32_t kRefTileK = 16;
constexpr uint32_t kRefInvocations = 256; // 8 subgroups x 32

static vkapi::ShaderInfo pick_coopmat_mm_ref_shader(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)graph;
  (void)args;
  (void)resize_args;
  return VK_KERNEL_FROM_STR("coopmat_mm_ref_half");
}

static utils::uvec3 pick_coopmat_mm_ref_global_wg_size(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args) {
  (void)shader;
  (void)resize_args;
  const ValueRef out = args.at(0).refs.at(0);
  const auto out_sizes = graph->sizes_of(out);
  const uint32_t M = out_sizes.at(out_sizes.size() - 2);
  const uint32_t N = out_sizes.at(out_sizes.size() - 1);
  // Same group-count cancellation trick as GemmCoopmat.cpp: the framework
  // divides by the local size, so multiplying tiles_n by kRefInvocations
  // yields exactly tiles_n x tiles_m workgroups.
  return {
      utils::div_up(N, kRefTileN) * kRefInvocations,
      utils::div_up(M, kRefTileM),
      1};
}

static utils::uvec3 pick_coopmat_mm_ref_local_wg_size(
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
  return {kRefInvocations, 1, 1};
}

void coopmat_mm_ref(ComputeGraph& graph, const std::vector<ValueRef>& args) {
  int idx = 0;
  const ValueRef mat1 = args.at(idx++);
  const ValueRef mat2 = args.at(idx++);
  const ValueRef out = args.at(idx++);

  VK_CHECK_COND(graph.dtype_of(out) == vkapi::kHalf);
  VK_CHECK_COND(graph.storage_type_of(out) == utils::kBuffer);
  VK_CHECK_COND(graph.storage_type_of(mat1) == utils::kBuffer);
  VK_CHECK_COND(graph.storage_type_of(mat2) == utils::kBuffer);

  const int32_t M = graph.size_at<int32_t>(-2, out);
  const int32_t N = graph.size_at<int32_t>(-1, out);
  const int32_t K = graph.size_at<int32_t>(-1, mat1);
  // No partial-tile or K-tail handling in the reference shader.
  VK_CHECK_COND(M % static_cast<int32_t>(kRefTileM) == 0);
  VK_CHECK_COND(N % static_cast<int32_t>(kRefTileN) == 0);
  VK_CHECK_COND(K % static_cast<int32_t>(kRefTileK) == 0);

  graph.execute_nodes().emplace_back(new DynamicDispatchNode(
      graph,
      pick_coopmat_mm_ref_shader,
      pick_coopmat_mm_ref_global_wg_size,
      pick_coopmat_mm_ref_local_wg_size,
      // Inputs and Outputs
      {{out, vkapi::kWrite}, {{mat1, mat2}, vkapi::kRead}},
      // Shader params buffers — none; all geometry is spec constants
      {},
      // Push Constants
      {},
      // Specialization Constants
      {K, N},
      // Resize Args
      {},
      // Resizing Logic
      nullptr));
}

REGISTER_OPERATORS {
  VK_REGISTER_OP(etvk.coopmat_mm_ref, coopmat_mm_ref);
}

} // namespace vkcompute
