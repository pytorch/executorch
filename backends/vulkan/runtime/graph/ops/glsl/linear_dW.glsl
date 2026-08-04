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

#define TILE_N 4
#define TILE_K 4

${define_required_extensions(STORAGE, DTYPE)}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_dw", DTYPE, STORAGE, is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_dout", DTYPE, STORAGE, is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_x", DTYPE, STORAGE, is_scalar_array=True)}

${layout_declare_ubo(B, "ivec4", "dout_sizes")}
${layout_declare_ubo(B, "ivec4", "x_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  // dW[N, K] = sum_m d_out[m, N] * x[m, K]; contraction over the flattened M.
  const int N = dout_sizes.x;
  const int M = dout_sizes.y * dout_sizes.z * dout_sizes.w;
  const int K = x_sizes.x;

  const int nnt = (N + TILE_N - 1) / TILE_N;
  const int nkt = (K + TILE_K - 1) / TILE_K;
  const int tiles = nnt * nkt;

  const int tile_idx = int(gl_GlobalInvocationID.x);
  if (tile_idx >= tiles) {
    return;
  }

  const int n0 = (tile_idx / nkt) * TILE_N;
  const int k0 = (tile_idx % nkt) * TILE_K;

  float acc[TILE_N * TILE_K];
  for (int i = 0; i < TILE_N * TILE_K; ++i) {
    acc[i] = 0.0;
  }

  for (int m = 0; m < M; ++m) {
    float dout_reg[TILE_N];
    for (int nl = 0; nl < TILE_N; ++nl) {
      const int n_eff = min(n0 + nl, N - 1);
      dout_reg[nl] = float(t_dout[m * N + n_eff]);
    }
    for (int kl = 0; kl < TILE_K; ++kl) {
      const int k_eff = min(k0 + kl, K - 1);
      const float xv = float(t_x[m * K + k_eff]);
      for (int nl = 0; nl < TILE_N; ++nl) {
        acc[nl * TILE_K + kl] += dout_reg[nl] * xv;
      }
    }
  }

  for (int nl = 0; nl < TILE_N; ++nl) {
    const int n = n0 + nl;
    for (int kl = 0; kl < TILE_K; ++kl) {
      const int k = k0 + kl;
      if (n < N && k < K) {
        t_dw[n * K + k] = T(acc[nl * TILE_K + kl]);
      }
    }
  }
}
