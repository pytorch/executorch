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
 * (impl/gemm/linear/qw/dq8ca_q4gsw/Dq8caQ4gswLinearWmma.cpp). Reads the 4h4w
 * int8 activations of quantize_and_pack_4h4w_with_group_sums. Texel-wise B
 * staging, multi-block A staging and an optional workgroup order
 * (ORDER_NB_MAX). Template header only; the body is
 * linear_dq8ca_q4gsw_wmma_body.glslh.
 *
 * Preconditions (not checked in the shader), enforced by the selection in
 * CoopmatFamily.cpp: M % WG_TILE_M == 0, N % WG_TILE_N == 0,
 * K % WG_TILE_K == 0, group_size % WG_TILE_K == 0, no bias, batch 1.
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
${layout_declare_tensor(B, "r", "t_packed_int8_input",   "int",  "buffer", is_scalar_array=False)}
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
$if A_MULTI_BLOCK:
  #define A_MULTI_BLOCK
const uint A_BLOCKS = ${A_BLOCKS};

// Workgroup order: 0 = dispatch order; > 0 = blocks of at most ORDER_NB_MAX
// N tiles, all M tiles of a block before the next block.
#define ORDER_NB_MAX ${ORDER_NB_MAX}

#include "linear_dq8ca_q4gsw_wmma_body.glslh"
