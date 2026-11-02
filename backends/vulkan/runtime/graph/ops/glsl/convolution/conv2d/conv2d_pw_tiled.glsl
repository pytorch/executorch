/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}
#define T ${buffer_scalar_type(DTYPE)}
#define VEC4_T ${buffer_gvec_type(DTYPE, 4)}

#define TILE_M4 ${TILE_M4}
#define TILE_K4 ${TILE_K4}
#define TILE_N4 ${TILE_N4}

#define TILE_M ${TILE_M}
#define TILE_H ${TILE_H}
#define TILE_W (TILE_M / TILE_H)
#define TILE_K ${TILE_K4 * 4}
#define TILE_N ${TILE_N4 * 4}

${define_required_extensions(STORAGE, DTYPE)}
$if DTYPE == "half":
  #extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
$if WEIGHT_STORAGE == "buffer":
  ${define_required_extensions("buffer", DTYPE)}

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_weight_packed", DTYPE, WEIGHT_STORAGE, is_scalar_array=False)}
$if WEIGHT_STORAGE == "buffer":
  #define WEIGHT_BUFFER
$if PIXEL_TILES_ON_X:
  #define PIXEL_TILES_ON_X
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

#include "gemm/tile_utils/linear_fp_input_tile.glslh"
#include "gemm/tile_utils/linear_fp_packed_weight_tile_load.glslh"
#include "gemm/tile_utils/linear_fp_output_tile_fp_compute.glslh"

// Texel coordinates for the TILE_W x TILE_H block of output pixels this
// invocation covers.
//
// A block spanning two rows lets each warp read and write whole cache lines of
// the block-interleaved texture layout used by some GPUs.
//
// Locating the block costs a division and a modulo by a runtime uniform, so the
// coordinates are computed once per invocation to keep the K loop free of
// integer division.
struct InCoords {
  ivec2 pos[TILE_M];
  ivec2 out_pos[TILE_M];
  bool in_bounds[TILE_M];
  // 1 for pixels inside the input, 0 for padding.
  T in_scale[TILE_M];
};

void compute_in_coords(
    out InCoords coords,
    const int tile_idx_m,
    const int tiles_x,
    const int W_out,
    const int H_out,
    const ivec2 in_extents) {
  const int x0 = (tile_idx_m % tiles_x) * TILE_W;
  const int y0 = (tile_idx_m / tiles_x) * TILE_H;
  [[unroll]] for (int m = 0; m < TILE_M; ++m) {
    const int x = x0 + m % TILE_W;
    const int y = y0 + m / TILE_W;
    coords.in_bounds[m] = x < W_out && y < H_out;
    // Pixels outside the output reuse the nearest pixel inside it so the K loop
    // needs no bounds check for them; their results are never stored.
    const int out_x = min(x, W_out - 1);
    const int out_y = min(y, H_out - 1);
    coords.out_pos[m] = ivec2(out_x, out_y);
    if (stride_1_padding_0 != 0) {
      coords.pos[m] = ivec2(out_x, out_y);
      coords.in_scale[m] = T(1);
    } else {
      // Padding pixels read the nearest input pixel and are zeroed by a
      // multiply. Zeroing them with a branch or select instead dropped valid
      // pixels on Adreno 830 (Float) and PowerVR DXT (Half).
      const ivec2 pos =
          ivec2(out_x * stride_w - padding_w, out_y * stride_h - padding_h);
      coords.pos[m] = clamp(pos, ivec2(0), in_extents - 1);
      coords.in_scale[m] = T(all(equal(coords.pos[m], pos)) ? 1.0 : 0.0);
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
      // The k4 check is redundant when TILE_K4 == 1, but with no branch left in
      // the K loop the Adreno compiler reads the weight buffer through a much
      // slower path.
      if (k4_start + k4 < K4) {
        tile.data[m][k4] =
            VEC4_T(texelFetch(t_in, ivec3(coords.pos[m], k4_start + k4), 0));
        if (stride_1_padding_0 == 0) {
          tile.data[m][k4] *= coords.in_scale[m];
        }
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
    const int N4) {
  // Reuse the coordinates computed for this invocation rather than dividing by
  // W_out again per output texel.
  [[unroll]] for (int m = 0; m < TILE_M; ++m) {
    [[unroll]] for (int n4 = 0; n4 < TILE_N4; ++n4) {
      if (coords.in_bounds[m] && n4_start + n4 < N4) {
        VEC4_T texel = out_tile.data[m][n4];
        if (activation_type == 1) {
          texel = max(texel, VEC4_T(0.0));
        } else if (activation_type == 2) {
          texel = clamp(texel, VEC4_T(out_min), VEC4_T(out_max));
        }
        imageStore(t_out, ivec3(coords.out_pos[m], n4_start + n4), vec4(texel));
      }
    }
  }
}

void main() {
#ifdef PIXEL_TILES_ON_X
  const int tile_idx_m = int(gl_GlobalInvocationID.x);
  const int tile_idx_n = int(gl_GlobalInvocationID.y);
#else
  const int tile_idx_n = int(gl_GlobalInvocationID.x);
  const int tile_idx_m = int(gl_GlobalInvocationID.y);
#endif

  const int n4_start = tile_idx_n * TILE_N4;

  const int K = in_sizes.z;
  const int K4 = div_up_4(K);

  const int W_out = out_sizes.x;
  const int H_out = out_sizes.y;
  const int tiles_x = div_up(W_out, TILE_W);
  const int N = out_sizes.z;
  const int N4 = div_up_4(N);

  if (n4_start >= N4 || tile_idx_m >= tiles_x * div_up(H_out, TILE_H)) {
    return;
  }

  FPOutTile out_tile;
  initialize(out_tile);

  FPInputTile in_tile;
  FPWeightTile w_tile;

  InCoords in_coords;
  compute_in_coords(
      in_coords, tile_idx_m, tiles_x, W_out, H_out, in_sizes.xy);

  for (int k4 = 0; k4 < K4; k4++) {
    load_input_tile_with_checks(in_tile, in_coords, k4, K4);
    load_packed_weight_tile_clamped(w_tile, n4_start, k4, 0, N4, K4);
    fp_accumulate_with_fp_weight(out_tile, in_tile, w_tile);
  }

  // Apply bias
  [[unroll]] for (int m = 0; m < TILE_M; ++m) {
    [[unroll]] for (int n4 = 0; n4 < TILE_N4; ++n4) {
      if (n4_start + n4 < N4) {
        out_tile.data[m][n4] +=
            VEC4_T(texelFetch(t_bias, ivec2(n4_start + n4, 0), 0));
      }
    }
  }

  store_output_tile_with_checks(out_tile, in_coords, n4_start, N4);
}
