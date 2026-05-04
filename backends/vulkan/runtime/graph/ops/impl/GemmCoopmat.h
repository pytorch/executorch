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
//
// Three device-capability gates beyond simple shape alignment:
//   * 2D outputs only — dispatch z-dim is hardcoded to 1, so batched outputs
//     silent-miscompute batch > 0.
//   * subgroup_size() == 64 — the shader bakes a 4-subgroup x 64-thread =
//     256-thread workgroup; subgroup-32 devices would silently miscompute.
//   * !is_integrated_gpu() — the kernel is desktop-tuned (256-thread
//     workgroups, ~9.5 KB shared mem, fp32 accumulators).
inline bool is_coopmat_eligible(
    ComputeGraph& graph,
    const ValueRef out,
    int64_t M,
    int64_t N,
    int64_t K) {
  if (graph.dim_of(out) > 2) {
    return false;
  }
  const auto* adapter = graph.context()->adapter_ptr();
  return adapter->supports_cooperative_matrix() &&
      adapter->subgroup_size() == 64 && !adapter->is_integrated_gpu() &&
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
