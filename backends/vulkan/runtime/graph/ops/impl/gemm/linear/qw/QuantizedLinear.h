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
#include <executorch/backends/vulkan/runtime/graph/ops/impl/gemm/GemmCommon.h>
#include <executorch/backends/vulkan/runtime/graph/ops/impl/utils/QuantizationConfig.h>

namespace vkcompute {

// Per-shader coopmat tile geometry (must match each shader's yaml).
// Workgroup size (wg_size) = SG_GRID_X * SG_GRID_Y * SUBGROUP_SIZE.
//   linear_q4gsw_coopmat       128x64x16, 2x2 subgroups x 32 (forced) -> 128
//   linear_dq8ca_q4gsw_coopmat 128x64x32, 2x2 subgroups x 64          -> 256
// (The int8-MMA shaders stay on wave64: int8 WMMA at forced subgroup 32
// crashes the Xclipse PAL compiler.)
struct CoopmatTileDims {
  uint32_t m;
  uint32_t n;
  uint32_t k;
  // Threads per workgroup = SG_GRID_X * SG_GRID_Y * SUBGROUP_SIZE. MUST match
  // the WG_SIZE the shader yaml resolves to, or the launched thread count won't
  // match the shader's staging passes (out-of-bounds).
  uint32_t wg_size;
};
// linear_qw_coopmat.yaml: 128x64, 2x2 subgroup grid, sg32 -> WG_SIZE 128.
constexpr CoopmatTileDims kQ4gswCoopmatDims = {128, 64, 16, 128};
// linear_dq8ca_qw_coopmat.yaml: 128x64, 2x2 grid, sg64 -> WG_SIZE 256.
constexpr CoopmatTileDims kDq8caQ4gswCoopmatDims = {128, 64, 32, 256};

void resize_linear_qw_node(
    ComputeGraph* graph,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& extra_args);

GlobalWorkGrid quantized_linear_gwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

LocalWorkGroup quantized_linear_lwg(
    ComputeGraph* graph,
    const vkapi::ShaderInfo& shader,
    const GlobalWorkGrid& gwg,
    const std::vector<ArgGroup>& args,
    const std::vector<ValueRef>& resize_args);

bool can_use_q4gsw_coopmat(
    ComputeGraph* graph,
    const ValueRef output,
    const ValueRef fp_input,
    int64_t group_size,
    const ValueRef bias,
    int64_t tile_m = kCoopmatTileM,
    int64_t tile_n = kCoopmatTileN,
    int64_t tile_k = kCoopmatTileK);

ValueRef prepack_quantized_linear_weight(
    ComputeGraph& graph,
    const QuantizationConfig& weight_quant_config,
    const ValueRef qmat2_data,
    const bool use_unsigned_dot = false);

void quantized_linear_impl(
    ComputeGraph& graph,
    const QuantizationConfig& input_quant_config,
    const QuantizationConfig& weight_quant_config,
    const ValueRef fp_input,
    const ValueRef input_scale,
    const ValueRef input_zp,
    const ValueRef weight_data,
    const ValueRef weight_sums_data,
    const ValueRef weight_scales_data,
    const ValueRef weight_zeros_data,
    const ValueRef group_size,
    const ValueRef bias_data,
    const ValueRef output);

} // namespace vkcompute
