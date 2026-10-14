/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>
#include <executorch/backends/vulkan/runtime/graph/ops/ExecuteNode.h>

namespace vkcompute {

// Helpers shared by the linear/matmul tiled and coopmat dispatch paths.

// Prepack a floating-point weight tensor for the linear/matmul kernels.
// Source layout: [N, K] (is_transposed=true) or [K, N] (is_transposed=false),
// optionally batched. Output layout: 4OC x 4IC blocked, packed as
// kWidthPacked. When force_buffer is true, the packed tensor uses buffer
// storage (required by the coopmat shader); otherwise texture2d is used
// when it fits within max_texture2d_dim.
ValueRef prepack_fp_linear_weight(
    ComputeGraph& graph,
    const ValueRef weight_data,
    bool is_transposed,
    int64_t B,
    bool force_buffer = false);

// Resize logic for linear-shaped output: takes M from the input's penultimate
// dim, N from resize_args[0], optional batch from input's leading dim.
void resize_linear_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

// Resize logic for matmul (mat1 @ mat2): output rows from mat1, cols from
// mat2, batch dims propagated from mat1.
void resize_matmul_tiled_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

// Cooperative-matrix GEMM (coopmat_mm.glsl, shared by linear and matmul).
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

} // namespace vkcompute
