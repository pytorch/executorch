/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * 8da4w int8 cooperative-matrix linear that reads row-major int8 activations
 * (utils::kPackedInt8_4W, written by quantize_and_pack_4w_with_group_sums),
 * used by the architecture-family path
 * (impl/gemm/linear/qw/dq8ca_q4gsw/Dq8caQ4gswLinearWmma.cpp). The 4h4w layout
 * is not row-major, so no coopMatLoad can address it; this kernel needs the
 * row-major packer. Template header only; the body is
 * linear_dq8ca_q4gsw_wmma_rowmajor_body.glslh, which:
 *   - stages A as raw 16-byte vectors (uvec4 global load -> uvec4 shared
 *     store) instead of coopMatLoad(global) -> coopMatStore(shared); needs
 *     WG_TILE_M * WG_TILE_K / 16 to be a multiple of the workgroup size;
 *   - lets one weight texel fetch feed both nibble parities of a component
 *     (output columns c and c + 4);
 *   - stages the texture3d drain band in the A shared-memory tile, which is
 *     dead after the last MMA.
 *
 * Preconditions (not checked in the shader), enforced by the selection in
 * CoopmatFamily.cpp: N % WG_TILE_N == 0, K % WG_TILE_K == 0,
 * group_size % WG_TILE_K == 0, no bias, batch 1. M is rounded up by the grid.
 */

#version 450 core

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
// A is bound as a scalar int8_t array so that a coopMatLoad from it has a
// matching component type.
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : enable

#define PRECISION ${PRECISION}

$if IO_STORAGE == "texture3d":
  #define IO_TEXTURE

layout(std430) buffer;

#include "common.glslh"

// Bindings, in the argument order of the dispatch node:
//   output(0), fp_input(1), packed_int8_input(2), int_input_sums(3),
//   input_scales(4), input_zps(5), packed_weight(6), weight_sums(7),
//   weight_scales(8), bias(9).
${layout_declare_tensor(B, "w", "t_output",              "half", IO_STORAGE, is_scalar_array=True)}
// t_input is unread here -- the activations arrive already quantized in
// t_packed_int8_input -- but stays declared so the binding layout matches the
// dispatch site.
${layout_declare_tensor(B, "r", "t_input",               "half", IO_STORAGE, is_scalar_array=False)}
// Row-major packed activations (row stride = K int8).
${layout_declare_tensor(B, "r", "t_packed_int8_input",   "int8", "buffer", is_scalar_array=True)}
// uvec4 view of binding 2 (t_packed_int8_input) for 16-byte A staging.
// Spelled layout(std430, set...) on purpose: gen_vulkan_spv.py collects the
// descriptor list from every line matching ^layout\(set, so the usual spelling
// would add a descriptor.
layout(std430, set = 0, binding = 2) buffer restrict readonly t_packed_int8_input_v4Buffer {
  uvec4 t_packed_int8_input_v4[];
};
${layout_declare_tensor(B, "r", "t_int8_input_sums",     "int",  "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_int8_input_scales",   "half", "texture3d")}
${layout_declare_tensor(B, "r", "t_int8_input_zps",      "int8", "texture3d")}
${layout_declare_tensor(B, "r", "t_packed_weight",       "int",  WEIGHT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_sums",         "int",  "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_weight_scales",       "half", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias",                "half", "buffer", is_scalar_array=True)}

${layout_declare_ubo(B, "ivec4", "output_sizes")}
${layout_declare_ubo(B, "ivec4", "input_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "apply_bias",   "0")}
${layout_declare_spec_const(C, "int", "K4_per_group", "0")}
${layout_declare_spec_const(C, "int", "num_groups_arg", "0")}
${layout_declare_spec_const(C, "int", "out_N_arg", "0")}

// Tile geometry
const uint MMA_M = ${MMA_M};
const uint MMA_N = ${MMA_N};
const uint MMA_K = ${MMA_K};

const uint WG_TILE_M = ${WG_TILE_M};
const uint WG_TILE_N = ${WG_TILE_N};
const uint WG_TILE_K = ${WG_TILE_K};

const uint SG_GRID_X = ${SG_GRID_X};
const uint SG_GRID_Y = ${SG_GRID_Y};
const uint SUBGROUP_SIZE = ${SUBGROUP_SIZE};

#include "linear_dq8ca_q4gsw_wmma_rowmajor_body.glslh"
