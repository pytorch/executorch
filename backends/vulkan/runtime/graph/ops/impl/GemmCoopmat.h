/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

namespace vkcompute {

// Cooperative-matrix GEMM dispatch. The coopmat shader (coopmat_mm.glsl) is
// shared across linear/matmul; the two add_*_node entry points below differ
// only in input layout (prepacked weight vs runtime mat2) and bias handling.
//
// Tile dimensions match the coopmat_mm.yaml defaults; they participate in
// the eligibility check below because the shader has no partial-tile or
// K-tail handling — misaligned shapes must fall back to the tiled path.

constexpr uint32_t kCoopmatTileM = 64;
constexpr uint32_t kCoopmatTileN = 64;
constexpr uint32_t kCoopmatTileK = 32;
constexpr uint32_t kCoopmatInvocations = 256; // 4 subgroups x 64

// Whether the coopmat path can be dispatched for the given M/N/K shape.
// Caller is responsible for falling back to the tiled path when this returns
// false.
inline bool is_coopmat_eligible(
    ComputeGraph& graph,
    const ValueRef out,
    int64_t M,
    int64_t N,
    int64_t K) {
  // The shader operates on 2D buffers; dispatch z-dim is hardcoded to 1, so
  // batched outputs would silent-miscompute batch > 0. Reject any rank-3+
  // output conservatively.
  if (graph.dim_of(out) > 2) {
    return false;
  }
  // TODO: also gate on adapter->subgroup_size() == 64 once a subgroup-size
  // accessor lands on Adapter. The shader bakes a 4-subgroup x 64-thread =
  // 256-thread workgroup; on a subgroup-32 device this would silently
  // miscompute. Today's only in-tree adapter exposing
  // VK_KHR_cooperative_matrix is subgroup-64, so the assumption holds.
  return graph.context()->adapter_ptr()->supports_cooperative_matrix() &&
      graph.storage_type_of(out) == utils::kBuffer && M % kCoopmatTileM == 0 &&
      N % kCoopmatTileN == 0 && K % kCoopmatTileK == 0;
}

void add_linear_coopmat_node(
    ComputeGraph& graph,
    const ValueRef input,
    const ValueRef packed_weight,
    const ValueRef packed_bias,
    bool has_bias,
    const ValueRef out,
    int32_t weight_B = 1);

void add_matmul_coopmat_node(
    ComputeGraph& graph,
    const ValueRef mat1,
    const ValueRef mat2,
    const ValueRef out);

} // namespace vkcompute
