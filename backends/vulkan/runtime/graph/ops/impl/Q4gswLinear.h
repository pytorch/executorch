/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/api/api.h>

#include <executorch/backends/vulkan/runtime/graph/ComputeGraph.h>

namespace vkcompute {

//
// Shared constants and helpers — exposed so test/benchmark binaries (e.g.
// TestFpaQ4gswLinear.cpp) can build forced-shader dispatch paths that reuse
// the same prepack, resize, and workgroup-sizing logic as the production
// dispatchers below. Production callers do not need to touch these directly.
//

// fp32 GEMM tile shape — 4M x 8N per-thread tile, 8x8 LWG.
constexpr uint32_t kGemmTileM = 4u;
constexpr uint32_t kGemmTileN = 8u;

// fp16 tin GEMM tile shape — 8M x 4N per-thread tile, 1x128 LWG.
constexpr uint32_t kTinGemmTileM = 8u;
constexpr uint32_t kTinGemmTileN = 4u;

// Resize output [M, N] based on current fp_input M and packed_weight shape.
// extra_args = { weight_data_tref, fp_input }.
void resize_q4gsw_linear_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args);

// Prepack [N, K/2] uint8 weights into a W_4X8 block-packed nibble buffer of
// size [K/4, N/4] ivec2 elements (stored as 2 * K4 * N4 ints). Each ivec4
// covers a 4K x 8N block of nibbles.
ValueRef prepack_q4_w_4x8_nc_buffer(
    ComputeGraph& graph,
    const ValueRef weight_data);

// Prepack [K/gs, N] float scales into a dtype-matched buffer so the GEMM
// shader can read scales as vec4 (fp32) or f16vec4 (fp16) via the binding
// dtype.
ValueRef prepack_q4_scales(
    ComputeGraph& graph,
    const ValueRef weight_scales_data,
    vkapi::ScalarType dtype);

// Global/local workgroup pickers for the fp32 GEMM path.
utils::uvec3 pick_q4gsw_linear_gemm_global_wg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

utils::uvec3 pick_q4gsw_linear_gemm_local_wg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

// Global/local workgroup pickers for the fp16 tin GEMM path —
// {ceil(M/8), ceil(N/4), 1} global, {1, 128, 1} local.
utils::uvec3 pick_q4gsw_linear_tin_gemm_global_wg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

utils::uvec3 pick_q4gsw_linear_tin_gemm_local_wg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const utils::uvec3& global_workgroup_size,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

// Q4 group-symmetric-weight GEMM/GEMV optimized for Adreno.
//
// Each dispatcher registers two execute nodes that share a 6-binding layout
//     (output, fp_input, transposed_input, q4_weights, scales, bias)
// so one descriptor set matches every variant. The first node binds the
// dtype's GEMM shader and self-gates to {0,0,0} when M==1; the second node
// binds the adaptive nc-coop GEMV shader and self-gates to {0,0,0} when
// M!=1. The framework re-runs each node's pickers on every trigger_resize()
// so `virtual_resize` updates that cross the M==1 boundary are routed
// without baking in the initial-M decision.
//
//   - add_q4gsw_linear_w_4x8_node (fp32):
//       GEMM = `q4gsw_linear_gemm__w_4x8_nc` (reads fp_input).
//       The transposed_input binding is a 0-element dummy TmpTensor.
//
//   - add_q4gsw_linear_tin_w_4x8_node (fp16):
//       Preprocess transpose (self-gates to {0,0,0} when M==1) populates
//       transposed_input. GEMM = `q4gsw_linear_gemm__tin__w_4x8_nc`
//       (reads transposed_input).
void add_q4gsw_linear_tin_w_4x8_node(
    ComputeGraph& graph,
    const ValueRef fp_input,
    const ValueRef weight_data,
    const ValueRef weight_scales_data,
    const ValueRef group_size_ref,
    const ValueRef bias_data,
    const ValueRef output);

void add_q4gsw_linear_w_4x8_node(
    ComputeGraph& graph,
    const ValueRef fp_input,
    const ValueRef weight_data,
    const ValueRef weight_scales_data,
    const ValueRef group_size_ref,
    const ValueRef bias_data,
    const ValueRef output);

} // namespace vkcompute
