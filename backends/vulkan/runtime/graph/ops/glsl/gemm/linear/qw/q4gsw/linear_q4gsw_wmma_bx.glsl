/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * fp16 q4gsw cooperative-matrix linear with texel-wise weight staging and an
 * optional fp32 accumulator, used by the architecture-family path
 * (impl/gemm/linear/qw/q4gsw/Q4gswLinearWmma.cpp) for long-K shapes where
 * fp16 accumulation loses accuracy. Template header only; the body is
 * linear_q4gsw_wmma_bx_body.glslh.
 *
 * Feature defines (yaml parameters, all default off):
 *   ACC_FP32    fp32 accumulator
 *   CSH_IN_ASH  texture3d drain staged in the A shared-memory tile, which is
 *               dead after the last MMA
 *
 * Preconditions as for linear_q4gsw_wmma.glsl.
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

$if ACC_FP32:
  #define ACC_FP32

#ifdef ACC_FP32
#define ACC_T float
#else
#define ACC_T float16_t
#endif

$if CSH_IN_ASH:
  #define CSH_IN_ASH

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

#include "linear_q4gsw_wmma_bx_body.glslh"
