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

#define OUTPUT_BUFFER
#define INPUT_BUFFER
#define MAT2_BUFFER
#define SCALAR_BUFFER
$if HAS_BIAS:
  #define HAS_BIAS
  #define BIAS_BUFFER

#define TILE_M4 ${TILE_M4}
#define TILE_K4 ${TILE_K4}
#define TILE_N4 ${TILE_N4}

#define TILE_M ${TILE_M}
#define TILE_K ${TILE_K4 * 4}
#define TILE_N ${TILE_N4 * 4}

${define_required_extensions(STORAGE, DTYPE)}

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_output", DTYPE, STORAGE, is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_mat1", DTYPE, STORAGE, is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_mat2", DTYPE, STORAGE, is_scalar_array=True)}
$if HAS_BIAS:
  ${layout_declare_tensor(B, "r", "t_bias", DTYPE, STORAGE, is_scalar_array=True)}

${layout_declare_ubo(B, "ivec4", "mat1_sizes")}
${layout_declare_ubo(B, "ivec4", "mat2_sizes")}

$if HAS_BIAS:
  ${layout_declare_ubo(B, "ivec4", "bias_sizes")}

$if HAS_BIAS:
  layout(push_constant) uniform restrict Block {
    float alpha;
    float beta;
  };

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#include "matmul_fp_mat1_tile_load.glslh"
#include "matmul_fp_mat2_tile_load.glslh"
#include "linear_fp_output_tile_fp_compute.glslh"
#include "matmul_fp_out_tile_store.glslh"
#include "matmul_fp_bias_apply.glslh"

void main() {
  const int tile_idx_n = int(gl_GlobalInvocationID.x);
  const int tile_idx_m = int(gl_GlobalInvocationID.y);
  const int b = int(gl_GlobalInvocationID.z);

  const int n4_start = tile_idx_n * TILE_N4;
  const int m_start = tile_idx_m * TILE_M;

  // mat1 sizes in WHCN order: {K, M, B, 1}
  const int K = mat1_sizes.x;
  const int M = mat1_sizes.y;
  const int K4 = div_up_4(K);

  // mat2 sizes in WHCN order: {N, K, B, 1}
  const int N = mat2_sizes.x;
  const int N4 = div_up_4(N);

  if (n4_start >= N4 || m_start >= M) {
    return;
  }

  FPOutTile out_tile;
  initialize(out_tile);

  FPInputTile in_tile;
  FPWeightTile w_tile;

  for (int k4 = 0; k4 < K4; k4++) {
    load_mat1_tile_scalar(in_tile, k4, m_start, b, K4, K, M);
    load_mat2_tile_scalar(w_tile, n4_start, mul_4(k4), b, N4, N, K);
    fp_accumulate_with_fp_weight(out_tile, in_tile, w_tile);
  }

#ifdef HAS_BIAS
  apply_bias(out_tile, n4_start, m_start, N4, N);
#endif

  store_matmul_out_tile_scalar(out_tile, n4_start, m_start, b, N4, N, M);
}
