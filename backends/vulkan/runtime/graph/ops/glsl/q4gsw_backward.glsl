/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define T ${texel_load_component_type(DTYPE, STORAGE)}

#define TILE_M 4
#define TILE_K 4

${define_required_extensions(STORAGE, DTYPE)}
${define_required_extensions("buffer", DTYPE)}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_dx", DTYPE, STORAGE, is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_dout", DTYPE, STORAGE, is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_q4_weights", "int", "buffer", is_scalar_array=False, vec_size=4)}
${layout_declare_tensor(B, "r", "t_scales", DTYPE, "buffer", is_scalar_array=True)}

${layout_declare_ubo(B, "ivec4", "dout_sizes")}
${layout_declare_ubo(B, "ivec4", "dx_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "group_size", "32")}

// d_x[M, K] = d_out[M, N] @ dequant(W)[N, K], contracting over N.
// dequant(W[n, k]) = (code - 8) * scale, with code read from the same W_4X8
// block-packed weight the forward reads (mirrors q4gsw_linear_gemm__w_4x8.glsl).
void main() {
  const int N = dout_sizes.x;
  const int M = dout_sizes.y * dout_sizes.z * dout_sizes.w;
  const int K = dx_sizes.x;

  const int nmt = (M + TILE_M - 1) / TILE_M;
  const int nkt = (K + TILE_K - 1) / TILE_K;
  const int tiles = nmt * nkt;

  const int tile_idx = int(gl_GlobalInvocationID.x);
  if (tile_idx >= tiles) {
    return;
  }

  const int m0 = (tile_idx / nkt) * TILE_M;
  const int k0 = (tile_idx % nkt) * TILE_K;

  // K and N are multiples of 4 (prepack guarantees), so k0 is 4-aligned: the
  // tile's 4 K lanes are byte b = kl of one k4 group and share one scale group.
  const int k4 = k0 >> 2;
  const int N4 = (N + 3) >> 2;
  const int N4_padded = (N4 + 1) & ~1;
  const int N8 = N4_padded >> 1;
  const int group = k0 / group_size;

  float acc[TILE_M * TILE_K];
  for (int i = 0; i < TILE_M * TILE_K; ++i) {
    acc[i] = 0.0;
  }

  for (int n = 0; n < N; ++n) {
    float dout_reg[TILE_M];
    for (int ml = 0; ml < TILE_M; ++ml) {
      const int m_eff = min(m0 + ml, M - 1);
      dout_reg[ml] = float(t_dout[m_eff * N + n]);
    }

    // W_4X8 address for column n: ivec4 at (k4, n8); component by (n4 parity,
    // n-in-tile half); low/high nibble by n parity (even-N low, odd-N high).
    const int n4 = n >> 2;
    const int ni = n & 3;
    const int n8 = n4 >> 1;
    const int comp = (n4 & 1) * 2 + (ni >> 1);
    const int nib_hi = (ni & 1) * 4;
    const ivec4 w_block = t_q4_weights[k4 * N8 + n8];
    const int w_int = w_block[comp];
    const float scale = float(t_scales[group * N + n]);

    for (int kl = 0; kl < TILE_K; ++kl) {
      const int code = int((uint(w_int) >> (8 * kl + nib_hi)) & 0xFu);
      const float dq = float(code - 8) * scale;
      for (int ml = 0; ml < TILE_M; ++ml) {
        acc[ml * TILE_K + kl] += dout_reg[ml] * dq;
      }
    }
  }

  for (int ml = 0; ml < TILE_M; ++ml) {
    const int m = m0 + ml;
    for (int kl = 0; kl < TILE_K; ++kl) {
      const int k = k0 + kl;
      if (m < M && k < K) {
        t_dx[m * K + k] = T(acc[ml * TILE_K + kl]);
      }
    }
  }
}
