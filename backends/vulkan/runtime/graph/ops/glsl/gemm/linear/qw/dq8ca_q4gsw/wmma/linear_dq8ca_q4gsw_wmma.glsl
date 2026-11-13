/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * 8da4w (dynamically quantized int8 activations, 4-bit group-symmetric
 * weights) int8 cooperative-matrix linear for subgroup-16 devices with an
 * 8x16x32 int8 MMA, used by the architecture-family path
 * (impl/gemm/linear/qw/dq8ca_q4gsw/wmma/Dq8caQ4gswLinearWmma.cpp). Reads
 * the 4h4w int8 activations of quantize_and_pack_4h4w_with_group_sums.
 * Texel-wise B staging, multi-block A staging and an optional workgroup
 * order (ORDER_NB_MAX). Template header only; the body is
 * linear_dq8ca_q4gsw_wmma_body.glslh.
 *
 * ROWMAJOR_A reads row-major int8 activations instead (utils::kPackedInt8_4W,
 * written by quantize_and_pack_4w_with_group_sums): the 4h4w layout is not
 * row-major, so no coopMatLoad can address it. That variant stages A as raw
 * 16-byte vectors (needs WG_TILE_M * WG_TILE_K / 16 to be a multiple of the
 * workgroup size), lets one weight texel fetch feed both nibble parities of a
 * component (output columns c and c + 4), and stages the texture3d drain band
 * in the A shared-memory tile.
 *
 * Preconditions (not checked in the shader), enforced by the selection in
 * CoopmatFamily.cpp: M % WG_TILE_M == 0 (ROWMAJOR_A: M is rounded up by the
 * grid), N % WG_TILE_N == 0, K % WG_TILE_K == 0, group_size % WG_TILE_K == 0,
 * no bias, batch 1.
 */

#version 450 core

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : enable

#define PRECISION ${PRECISION}

// When the A staging thread map exactly covers the workgroup
// (A_ACTIVE_THREADS == WG_SIZE) the `a_active` guard is always true, but the
// driver compiler may not fold it because gl_LocalInvocationID.x's bound comes
// from a spec constant. Set A_MAP_FULL only for tiles where the equality holds.
$if A_MAP_FULL:
  #define A_ALWAYS_ACTIVE

$if IO_STORAGE == "texture3d":
  #define IO_TEXTURE

$if ROWMAJOR_A:
  #define ROWMAJOR_A

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
// ROWMAJOR_A: row-major (row stride = K int8), read as 16-byte vectors.
${layout_declare_tensor(B, "r", "t_packed_int8_input",   "uint" if ROWMAJOR_A else "int", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_int8_input_sums",     "int",  "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_int8_input_scales",   "half", "texture3d")}
${layout_declare_tensor(B, "r", "t_int8_input_zps",      "int8" if ZP_DTYPE_MODE == "zpint8" else "half", "texture3d")}
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
$if not ROWMAJOR_A:
  $if A_MULTI_BLOCK:
    #define A_MULTI_BLOCK
  const uint A_BLOCKS = ${A_BLOCKS};

  // Workgroup order: 0 = dispatch order; > 0 = blocks of at most ORDER_NB_MAX
  // N tiles, all M tiles of a block before the next block.
  #define ORDER_NB_MAX ${ORDER_NB_MAX}

#include "linear_dq8ca_q4gsw_wmma_body.glslh"
