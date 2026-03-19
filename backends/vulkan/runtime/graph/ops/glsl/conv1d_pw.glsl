/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}
#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}
#define T ${texel_load_component_type(DTYPE, STORAGE)}

$if STORAGE == "buffer":
  #define OUTPUT_BUFFER
  #define INPUT_BUFFER
$if WEIGHT_STORAGE == "buffer":
  #define WEIGHT_BUFFER
$if HAS_BIAS:
  #define HAS_BIAS
$if STORAGE == "buffer" and HAS_BIAS:
  #define BIAS_BUFFER

#define TILE_M4 ${TILE_M4}
#define TILE_K4 ${TILE_K4}
#define TILE_N4 ${TILE_N4}

#define TILE_M ${TILE_M}
#define TILE_K ${TILE_K4 * 4}
#define TILE_N ${TILE_N4 * 4}

${define_required_extensions(STORAGE, DTYPE)}
$if WEIGHT_STORAGE != STORAGE:
  ${define_required_extensions(WEIGHT_STORAGE, DTYPE)}

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_packed", DTYPE, WEIGHT_STORAGE, is_scalar_array=False)}
$if HAS_BIAS:
  ${layout_declare_tensor(B, "r", "t_bias", DTYPE, STORAGE, is_scalar_array=False)}

// in_sizes: {L, C_in, N, 1} in WHCN order
${layout_declare_ubo(B, "ivec4", "in_sizes")}
// out_sizes: {L, C_out, N, 1} in WHCN order
${layout_declare_ubo(B, "ivec4", "out_sizes")}
$if HAS_BIAS:
  ${layout_declare_ubo(B, "ivec4", "bias_sizes")}

$if HAS_BIAS:
  layout(push_constant) uniform restrict Block {
    int weight_B;
    float alpha;
    float beta;
  };
$else:
  layout(push_constant) uniform restrict Block {
    int weight_B;
  };

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#include "linear_fp_input_tile.glslh"
#include "linear_fp_weight_tile.glslh"
#include "linear_fp_output_tile.glslh"
#include "linear_fp_packed_weight_tile_load.glslh"
#include "linear_fp_output_tile_fp_compute.glslh"

// Conv1d pointwise is matrix multiplication with swapped texture coordinates.
// Linear: input ivec3(k4, m, b), output ivec3(n4, m, b)  [width-packed]
// Conv1d: input ivec3(m, k4, b), output ivec3(m, n4, b)  [height-packed]
// Buffer indexing is identical: (b * M + m) * K4 + k4

VEC4_T load_input_x4(
    const int k4,
    const int m,
    const int b,
    const int K4,
    const int M) {
#ifdef INPUT_BUFFER
  return t_in[(b * M + m) * K4 + k4];
#else
  return texelFetch(t_in, ivec3(m, k4, b), 0);
#endif
}

void load_input_tile_with_checks(
    out FPInputTile tile,
    const int k4_start,
    const int m_start,
    const int b,
    const int K4,
    const int M) {
  [[unroll]] for (int m = 0; m < TILE_M; ++m) {
    [[unroll]] for (int k4 = 0; k4 < TILE_K4; ++k4) {
      if (k4_start + k4 < K4 && m_start + m < M) {
        tile.data[m][k4] =
            load_input_x4(k4_start + k4, m_start + m, b, K4, M);
      } else {
        tile.data[m][k4] = VEC4_T(0.0);
      }
    }
  }
}

void store_output_x4(
    const VEC4_T texel,
    const int n4,
    const int m,
    const int b,
    const int N4,
    const int M) {
#ifdef OUTPUT_BUFFER
  t_out[(b * M + m) * N4 + n4] = texel;
#else
  imageStore(t_out, ivec3(m, n4, b), texel);
#endif
}

void store_output_tile_with_checks(
    const FPOutTile out_tile,
    const int n4_start,
    const int m_start,
    const int b,
    const int N4,
    const int M) {
  [[unroll]] for (int m = 0; m < TILE_M; ++m) {
    [[unroll]] for (int n4 = 0; n4 < TILE_N4; ++n4) {
      if (m_start + m < M && n4_start + n4 < N4) {
        store_output_x4(
            out_tile.data[m][n4], n4_start + n4, m_start + m, b, N4, M);
      }
    }
  }
}

void main() {
  // Thread mapping: X=OC4 (N4), Y=L/tile_m (M tiles), Z=batch
  const int tile_idx_n = int(gl_GlobalInvocationID.x);
  const int tile_idx_m = int(gl_GlobalInvocationID.y);

  const int n4_start = tile_idx_n * TILE_N4;
  const int m_start = tile_idx_m * TILE_M;

  // in_sizes: {L, C_in, N, 1} in WHCN
  const int K = in_sizes.y;  // C_in
  const int M = in_sizes.x;  // L
  const int K4 = div_up_4(K);
  // out_sizes: {L, C_out, N, 1} in WHCN
  const int N_out = out_sizes.y; // C_out
  const int N4 = div_up_4(N_out);

  if (n4_start >= N4 || m_start >= M) {
    return;
  }

  FPOutTile out_tile;
  initialize(out_tile);

  FPInputTile in_tile;
  FPWeightTile w_tile;

  const int b = int(gl_GlobalInvocationID.z);

  for (int k4 = 0; k4 < K4; k4++) {
    load_input_tile_with_checks(in_tile, k4, m_start, b, K4, M);
    load_packed_weight_tile_with_checks(w_tile, n4_start, k4, 0, N4, K4);
    fp_accumulate_with_fp_weight(out_tile, in_tile, w_tile);
  }

#ifdef HAS_BIAS
  // Load bias (per output channel, width-packed) and apply
  [[unroll]] for (int n4 = 0; n4 < TILE_N4; ++n4) {
    VEC4_T bias_val = VEC4_T(0.0);
    if (n4_start + n4 < N4) {
#ifdef BIAS_BUFFER
      bias_val = t_bias[n4_start + n4];
#else
      bias_val = texelFetch(t_bias, ivec3(n4_start + n4, 0, 0), 0);
#endif
    }
    [[unroll]] for (int m = 0; m < TILE_M; ++m) {
      out_tile.data[m][n4] =
          VEC4_T(alpha) * out_tile.data[m][n4] + VEC4_T(beta) * bias_val;
    }
  }
#endif

  store_output_tile_with_checks(out_tile, n4_start, m_start, b, N4, M);
}
