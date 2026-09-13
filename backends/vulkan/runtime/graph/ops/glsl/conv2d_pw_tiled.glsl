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

#define TILE_M4 ${TILE_M4}
#define TILE_K4 ${TILE_K4}
#define TILE_N4 ${TILE_N4}

#define TILE_M ${TILE_M}
#define TILE_K ${TILE_K4 * 4}
#define TILE_N ${TILE_N4 * 4}

${define_required_extensions(STORAGE, DTYPE)}

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_weight_packed", DTYPE, "texture2d")}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "texture2d")}

${layout_declare_ubo(B, "ivec4", "in_sizes")}
${layout_declare_ubo(B, "ivec4", "out_sizes")}

layout(push_constant) uniform restrict Block {
  int stride_h;
  int stride_w;
  int padding_h;
  int padding_w;
  float out_min;
  float out_max;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "stride_1_padding_0", "0")}
${layout_declare_spec_const(C, "int", "activation_type", "0")}

#include "linear_fp_input_tile.glslh"
#include "linear_fp_packed_weight_tile_load.glslh"
#include "linear_fp_output_tile_fp_compute.glslh"

// Texel coordinates for the TILE_M spatial positions this invocation covers.
//
// Recovering (x, y) from the flattened spatial index costs a division and a
// modulo by W_out, which is a runtime uniform, so the compiler cannot strength
// reduce either into a shift. Those coordinates depend only on m, so computing
// them once per invocation keeps the K loop free of integer division; done
// per (m, k4) instead, a conv with K4 = 4 issues four times as many divides as
// the arithmetic they feed.
struct InCoords {
  ivec2 pos[TILE_M];
  ivec2 out_pos[TILE_M];
  bool valid[TILE_M];
};

void compute_in_coords(
    out InCoords coords,
    const int m_start,
    const int M,
    const int W_out,
    const int W_in,
    const int H_in) {
  [[unroll]] for (int m = 0; m < TILE_M; ++m) {
    const int out_spatial = m_start + m;
    if (out_spatial >= M) {
      coords.pos[m] = ivec2(0);
      coords.out_pos[m] = ivec2(0);
      coords.valid[m] = false;
      continue;
    }
    const int out_x = out_spatial % W_out;
    const int out_y = out_spatial / W_out;
    coords.out_pos[m] = ivec2(out_x, out_y);
    if (stride_1_padding_0 != 0) {
      coords.pos[m] = ivec2(out_x, out_y);
      coords.valid[m] = true;
    } else {
      const int in_x = out_x * stride_w - padding_w;
      const int in_y = out_y * stride_h - padding_h;
      coords.pos[m] = ivec2(in_x, in_y);
      coords.valid[m] = in_x >= 0 && in_x < W_in && in_y >= 0 && in_y < H_in;
    }
  }
}

void load_input_tile_with_checks(
    out FPInputTile tile,
    const InCoords coords,
    const int k4_start,
    const int K4) {
  [[unroll]] for (int m = 0; m < TILE_M; ++m) {
    [[unroll]] for (int k4 = 0; k4 < TILE_K4; ++k4) {
      if (k4_start + k4 < K4 && coords.valid[m]) {
        tile.data[m][k4] =
            texelFetch(t_in, ivec3(coords.pos[m], k4_start + k4), 0);
      } else {
        tile.data[m][k4] = VEC4_T(0.0);
      }
    }
  }
}

void store_output_tile_with_checks(
    const FPOutTile out_tile,
    const InCoords coords,
    const int n4_start,
    const int m_start,
    const int N4,
    const int M) {
  // Reuse the coordinates computed for this invocation rather than dividing by
  // W_out again per output texel.
  [[unroll]] for (int m = 0; m < TILE_M; ++m) {
    [[unroll]] for (int n4 = 0; n4 < TILE_N4; ++n4) {
      if (m_start + m < M && n4_start + n4 < N4) {
        VEC4_T texel = out_tile.data[m][n4];
        if (activation_type == 1) {
          texel = max(texel, VEC4_T(0.0));
        } else if (activation_type == 2) {
          texel = clamp(texel, VEC4_T(out_min), VEC4_T(out_max));
        }
        imageStore(t_out, ivec3(coords.out_pos[m], n4_start + n4), texel);
      }
    }
  }
}

void main() {
  const int tile_idx_n = int(gl_GlobalInvocationID.x);
  const int tile_idx_m = int(gl_GlobalInvocationID.y);

  const int n4_start = tile_idx_n * TILE_N4;
  const int m_start = tile_idx_m * TILE_M;

  const int W_in = in_sizes.x;
  const int H_in = in_sizes.y;
  const int K = in_sizes.z;
  const int K4 = div_up_4(K);

  const int W_out = out_sizes.x;
  const int H_out = out_sizes.y;
  const int M = W_out * H_out;
  const int N = out_sizes.z;
  const int N4 = div_up_4(N);

  if (n4_start >= N4 || m_start >= M) {
    return;
  }

  FPOutTile out_tile;
  initialize(out_tile);

  FPInputTile in_tile;
  FPWeightTile w_tile;

  InCoords in_coords;
  compute_in_coords(in_coords, m_start, M, W_out, W_in, H_in);

  for (int k4 = 0; k4 < K4; k4++) {
    load_input_tile_with_checks(in_tile, in_coords, k4, K4);
    load_packed_weight_tile_with_checks(w_tile, n4_start, k4, 0, N4, K4);
    fp_accumulate_with_fp_weight(out_tile, in_tile, w_tile);
  }

  // Apply bias
  [[unroll]] for (int m = 0; m < TILE_M; ++m) {
    [[unroll]] for (int n4 = 0; n4 < TILE_N4; ++n4) {
      if (n4_start + n4 < N4) {
        out_tile.data[m][n4] +=
            texelFetch(t_bias, ivec2(n4_start + n4, 0), 0);
      }
    }
  }

  store_output_tile_with_checks(out_tile, in_coords, n4_start, m_start, N4, M);
}
