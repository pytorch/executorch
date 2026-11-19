/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : require
#define VEC4_T ${buffer_gvec_type(DTYPE, 4)}
#define HAS_FLOAT16_ACCUMULATION

#define OUT_TILE ${OUT_TILE}
#define IN_TILE ${OUT_TILE + KERNEL_SIZE - 1}

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_m", DTYPE, "texture2d", precision="mediump")}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "texture2d")}
${layout_declare_tensor(B, "r", "t_residual", DTYPE, "texture3d", precision="mediump")}
${layout_declare_tensor(B, "r", "t_scale", DTYPE, "texture2d")}
${layout_declare_ubo(B, "ivec4", "out_sizes")}

layout(push_constant) uniform restrict Block {
  // The first tile row of this dispatch, and the tile rows t_m holds.
  ivec4 chunk;
  float activation_a;
  float activation_b;
  float residual_alpha;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "bool", "has_activation", "false")}
${layout_declare_spec_const(C, "int", "activation_op", "0")}
${layout_declare_spec_const(C, "bool", "has_residual", "false")}
${layout_declare_spec_const(C, "bool", "has_scale", "false")}
${layout_declare_spec_const(C, "int", "residual_op", "0")}

#include "conv2d_epilogue.glslh"
#include "conv2d_winograd.glslh"
#include "conv2d_winograd_f${OUT_TILE}x${OUT_TILE}_${KERNEL_SIZE}x${KERNEL_SIZE}.glslh"

// Pixels past the output read its last row or column, and are not stored.
VEC4_T load_residual(const ivec2 p, const int n4) {
  return VEC4_T(texelFetch(t_residual, ivec3(min(p, out_sizes.xy - 1), n4), 0));
}

// Stores row o of the output tile at `origin` in slice n4 from the GEMM output
// in column `tile` of t_m. Pass a constant row, so that only the rows of M with
// nonzero coefficients are loaded.
void transform_row(
    const int o,
    const ivec2 origin,
    const int tile,
    const int n4,
    const int N4) {
  VEC4_T x[IN_TILE];
  [[unroll]] for (int j = 0; j < IN_TILE; ++j) {
    VEC4_T column[IN_TILE];
    [[unroll]] for (int i = 0; i < IN_TILE; ++i) {
      column[i] = VEC4_T(
          texelFetch(t_m, ivec2(tile, (i * IN_TILE + j) * N4 + n4), 0));
    }
    x[j] = at_transform_element(column, o);
  }

  const VEC4_T bias = VEC4_T(texelFetch(t_bias, ivec2(n4, 0), 0));
  const vec4 scale = has_scale ? texelFetch(t_scale, ivec2(n4, 0), 0) : vec4(1);

  // Residual texels are loaded before the first store, since loading each at
  // its store would serialize it behind the stores before it. Loading them
  // ahead of M instead makes pipeline creation fail on Adreno 830.
  VEC4_T residual[OUT_TILE];
  [[unroll]] for (int k = 0; k < OUT_TILE; ++k) {
    residual[k] = VEC4_T(0);
    if (has_residual) {
      residual[k] = load_residual(origin + ivec2(k, o), n4);
    }
  }

  [[unroll]] for (int k = 0; k < OUT_TILE; ++k) {
    const ivec3 p = ivec3(origin + ivec2(k, o), n4);
    if (p.x < out_sizes.x && p.y < out_sizes.y) {
      imageStore(
          t_out,
          p,
          apply_epilogue_with_residual(
              at_transform_element(x, k) + bias, residual[k], scale));
    }
  }
}

/*
 * Output transform of Winograd F(m x m, r x r), with m = OUT_TILE: the m x m
 * output tile A^T M A of the IN_TILE x IN_TILE GEMM output M of each tile in
 * each 4-channel slice, plus the bias and the epilogue. Element
 * e = i * IN_TILE + j of M for tile t of the chunk, in row-major order, is at
 * (t, e * N4 + n4) of t_m.
 *
 * OUT_TILE invocations transform each tile, one per output row. Output rows
 * vary along y, so that the output row of a work group row is uniform.
 */
void main() {
  const int tile_x = int(gl_GlobalInvocationID.x);
  const int row = int(gl_GlobalInvocationID.y / OUT_TILE);
  const int o = int(gl_GlobalInvocationID.y % OUT_TILE);
  const int n4 = int(gl_GlobalInvocationID.z);

  const int N4 = div_up_4(out_sizes.z);
  const int tiles_x = tiles_of(out_sizes.x);
  const int rows = min(chunk.y, tiles_of(out_sizes.y) - chunk.x);
  if (tile_x >= tiles_x || row >= rows || n4 >= N4) {
    return;
  }

  const ivec2 origin = ivec2(tile_x, chunk.x + row) * OUT_TILE;
  const int tile = row * tiles_x + tile_x;
  // An else for the last branch took fewer instructions on Adreno than a test.
  $for i in range(OUT_TILE):
    $branch = ("else " if i > 0 else "") + ("if (o == %d) " % i if i < OUT_TILE - 1 else "")
    ${branch}{
      transform_row(${i}, origin, tile, n4, N4);
    }
}
