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

${layout_declare_tensor(B, "w", "t_output", DTYPE, STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_mat1", DTYPE, STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_packed", DTYPE, WEIGHT_STORAGE, is_scalar_array=False)}
$if HAS_BIAS:
  ${layout_declare_tensor(B, "r", "t_bias", DTYPE, STORAGE, is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "mat1_sizes")}
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

#include "matmul_fp_mat1_tile_load.glslh"
#include "linear_fp_packed_weight_tile_load.glslh"
#include "linear_fp_output_tile_fp_compute.glslh"
#include "matmul_fp_out_tile_store.glslh"

void main() {
  const int tile_idx_n = int(gl_GlobalInvocationID.x);
  const int tile_idx_m = int(gl_GlobalInvocationID.y);

  const int n4_start = tile_idx_n * TILE_N4;
  const int m_start = tile_idx_m * TILE_M;

  const int K = mat1_sizes.x;
  const int M = mat1_sizes.y; // mat1 [M, K] in WHCN = {K, M, 1, 1}
  const int K4 = div_up_4(K);
  const int N = out_sizes.x;
  const int N4 = div_up_4(N);

  if (n4_start >= N4 || m_start >= M) {
    return;
  }

  FPOutTile out_tile;
  initialize(out_tile);

  FPInputTile in_tile;
  FPWeightTile w_tile;

  const int b = int(gl_GlobalInvocationID.z);

  for (int k4 = 0; k4 < K4; k4++) {
    load_mat1_tile_with_checks(in_tile, k4, m_start, b, K4, M);
    load_packed_weight_tile_with_checks(w_tile, n4_start, k4, b % weight_B, N4, K4);
    fp_accumulate_with_fp_weight(out_tile, in_tile, w_tile);
  }

#ifdef HAS_BIAS
  // bias_sizes in WHCN order: {bias_N, bias_M, ...}
  const int b_N = bias_sizes.x;
  const int b_M = bias_sizes.y;
  const int b_N4 = div_up_4(b_N);
  [[unroll]] for (int m = 0; m < TILE_M; ++m) {
    const int bias_row = min(m_start + m, b_M - 1);
    [[unroll]] for (int n4 = 0; n4 < TILE_N4; ++n4) {
      VEC4_T bias_val = VEC4_T(0.0);
      if (n4_start + n4 < N4) {
        const int bias_n4 = min(n4_start + n4, b_N4 - 1);
#ifdef BIAS_BUFFER
        bias_val = t_bias[bias_row * b_N4 + bias_n4];
#else
        bias_val = texelFetch(t_bias, ivec3(bias_n4, bias_row, 0), 0);
#endif
        // When bias width is 1, broadcast .x to all components
        if (b_N == 1) {
          bias_val = VEC4_T(bias_val.x);
        }
      }
      out_tile.data[m][n4] =
          VEC4_T(alpha) * out_tile.data[m][n4] + VEC4_T(beta) * bias_val;
    }
  }
#endif

  store_matmul_out_tile_with_checks(out_tile, n4_start, m_start, b, N4, M);
}
