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
  #define SCALAR_BUFFER
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

$if STORAGE == "buffer":
  ${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE, is_scalar_array=True)}
  ${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE, is_scalar_array=True)}
$else:
  ${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE, is_scalar_array=False)}
  ${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_packed", DTYPE, WEIGHT_STORAGE, is_scalar_array=False)}
$if HAS_BIAS:
  $if STORAGE == "buffer":
    ${layout_declare_tensor(B, "r", "t_bias", DTYPE, STORAGE, is_scalar_array=True)}
  $else:
    ${layout_declare_tensor(B, "r", "t_bias", DTYPE, STORAGE, is_scalar_array=False)}

// in_sizes: {L, C_in, N, 1} in WHCN order
${layout_declare_ubo(B, "ivec4", "in_sizes")}
// out_sizes: {L, C_out, N, 1} in WHCN order
${layout_declare_ubo(B, "ivec4", "out_sizes")}
$if HAS_BIAS:
  ${layout_declare_ubo(B, "ivec4", "bias_sizes")}

layout(push_constant) uniform restrict Block {
  int weight_B;
  float output_min;
  float output_max;
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
//
// For buffer storage, height-packed tensors have packed_dim_block_size=1 (no
// vec4 grouping). Data is stored as contiguous scalars with strides based on
// logical sizes, so scalar indexing is required: (b * M + m) * C + c.
// For texture storage, 4 channels are packed per texel as usual.

#ifndef SCALAR_BUFFER
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
#endif // !SCALAR_BUFFER

#ifdef SCALAR_BUFFER
void load_input_tile_scalar(
    out FPInputTile tile,
    const int k4_start,
    const int m_start,
    const int b,
    const int K4,
    const int K,
    const int M) {
  [[unroll]] for (int m = 0; m < TILE_M; ++m) {
    [[unroll]] for (int k4 = 0; k4 < TILE_K4; ++k4) {
      if (k4_start + k4 < K4 && m_start + m < M) {
        const int base = (b * M + m_start + m) * K + mul_4(k4_start + k4);
        T s0 = t_in[base];
        T s1 = (mul_4(k4_start + k4) + 1 < K) ? t_in[base + 1] : T(0);
        T s2 = (mul_4(k4_start + k4) + 2 < K) ? t_in[base + 2] : T(0);
        T s3 = (mul_4(k4_start + k4) + 3 < K) ? t_in[base + 3] : T(0);
        tile.data[m][k4] = VEC4_T(s0, s1, s2, s3);
      } else {
        tile.data[m][k4] = VEC4_T(0.0);
      }
    }
  }
}

void store_output_tile_scalar(
    const FPOutTile out_tile,
    const int n4_start,
    const int m_start,
    const int b,
    const int N4,
    const int N,
    const int M) {
  [[unroll]] for (int m = 0; m < TILE_M; ++m) {
    [[unroll]] for (int n4 = 0; n4 < TILE_N4; ++n4) {
      if (m_start + m < M && n4_start + n4 < N4) {
        const int base = (b * M + m_start + m) * N + mul_4(n4_start + n4);
        const VEC4_T val = out_tile.data[m][n4];
        t_out[base] = val.x;
        if (mul_4(n4_start + n4) + 1 < N) t_out[base + 1] = val.y;
        if (mul_4(n4_start + n4) + 2 < N) t_out[base + 2] = val.z;
        if (mul_4(n4_start + n4) + 3 < N) t_out[base + 3] = val.w;
      }
    }
  }
}
#endif // SCALAR_BUFFER

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
#ifdef SCALAR_BUFFER
    load_input_tile_scalar(in_tile, k4, m_start, b, K4, K, M);
#else
    load_input_tile_with_checks(in_tile, k4, m_start, b, K4, M);
#endif
    load_packed_weight_tile_with_checks(w_tile, n4_start, k4, 0, N4, K4);
    fp_accumulate_with_fp_weight(out_tile, in_tile, w_tile);
  }

#ifdef HAS_BIAS
  // Load bias (per output channel) and apply
  [[unroll]] for (int n4 = 0; n4 < TILE_N4; ++n4) {
    VEC4_T bias_val = VEC4_T(0.0);
    if (n4_start + n4 < N4) {
#ifdef BIAS_BUFFER
      // Bias is a 1D tensor [C_out], width-packed.
      // For buffer storage, width-packed has packed_dim_block_size=1, so data
      // is stored as contiguous scalars. Read 4 with bounds checking.
      const int bias_base = mul_4(n4_start + n4);
      T b0 = t_bias[bias_base];
      T b1 = (bias_base + 1 < N_out) ? t_bias[bias_base + 1] : T(0);
      T b2 = (bias_base + 2 < N_out) ? t_bias[bias_base + 2] : T(0);
      T b3 = (bias_base + 3 < N_out) ? t_bias[bias_base + 3] : T(0);
      bias_val = VEC4_T(b0, b1, b2, b3);
#else
      bias_val = texelFetch(t_bias, ivec3(n4_start + n4, 0, 0), 0);
#endif
    }
    [[unroll]] for (int m = 0; m < TILE_M; ++m) {
      out_tile.data[m][n4] = out_tile.data[m][n4] + bias_val;
    }
  }
#endif

  // Apply activation clamp
  [[unroll]] for (int m = 0; m < TILE_M; ++m) {
    [[unroll]] for (int n4 = 0; n4 < TILE_N4; ++n4) {
      out_tile.data[m][n4] =
          clamp(out_tile.data[m][n4], VEC4_T(output_min), VEC4_T(output_max));
    }
  }

#ifdef SCALAR_BUFFER
  store_output_tile_scalar(out_tile, n4_start, m_start, b, N4, N_out, M);
#else
  store_output_tile_with_checks(out_tile, n4_start, m_start, b, N4, M);
#endif
}
