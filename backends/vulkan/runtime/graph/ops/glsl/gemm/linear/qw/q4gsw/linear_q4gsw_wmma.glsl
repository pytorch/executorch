/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * fp16 q4gsw (4-bit group-symmetric weight) cooperative-matrix linear with
 * per-variant tile geometry, used by the architecture-family path
 * (impl/gemm/linear/qw/q4gsw/Q4gswLinearWmma.cpp).
 *
 * This file is the template header: bindings, spec constants and the tile
 * geometry. The kernel body is linear_q4gsw_wmma_body.glslh.
 *
 * Loop structure: double-buffered, prefetch-peeled. The prologue prefetches
 * chunk 0; each iteration does barrier -> prefetch(next) -> MMA(cur) ->
 * store(next).
 *
 * Feature defines (yaml parameters, all default off):
 *   B_COLMAJOR   B staged N-major in shared memory, ColumnMajor B coopMatLoad
 *   CSH_BAND     texture3d drain one SG_GRID_Y band at a time
 *   FRAG_LAYOUT  fragment-contiguous shared memory, no padding
 *   IMG_A        storage-image loads for A (texture3d IO only)
 *   MMA_M = 8    for devices whose fp16 cooperative matrix is 8x16x16 only
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
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : enable

#define PRECISION ${PRECISION}

$if IO_STORAGE == "texture3d":
  #define IO_TEXTURE

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_output",         "half", IO_STORAGE, is_scalar_array=True)}
$if IMG_A and IO_STORAGE == "texture3d":
  ${layout_declare_image(B, "r", "t_input", "half")}
$else:
  ${layout_declare_tensor(B, "r", "t_input",          "half", IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_packed_weight",  "int",  WEIGHT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_scales",  "half", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias",           "half", "buffer", is_scalar_array=True)}

${layout_declare_ubo(B, "ivec4", "output_sizes")}
${layout_declare_ubo(B, "ivec4", "input_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "apply_bias",   "0")}
${layout_declare_spec_const(C, "int", "K4_per_group", "0")}
${layout_declare_spec_const(C, "int", "num_groups_arg", "0")}
${layout_declare_spec_const(C, "int", "out_N_arg", "0")}

#define ACC_T float16_t

$if B_COLMAJOR:
  #define B_COLMAJOR

$if CSH_BAND:
  #define CSH_BAND

$if FRAG_LAYOUT:
  #define FRAG_LAYOUT

$if IMG_A and IO_STORAGE == "texture3d":
  #define IMG_A

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

#include "linear_q4gsw_wmma_body.glslh"
