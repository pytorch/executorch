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

#define TILE_M4 1
#define TILE_K4 1
#define TILE_N4 2
#define TILE_M 4
#define TILE_K 4
#define TILE_N 8

#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#define HAS_FLOAT16_ACCUMULATION
$if WEIGHT_STORAGE == "buffer":
  ${define_required_extensions("buffer", DTYPE)}
  #define WEIGHT_BUFFER

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_m", DTYPE, "texture2d")}
// The texture weight variant reads Half data at mediump, which is exact.
$MEDIUMP = WEIGHT_STORAGE != "buffer"
${layout_declare_tensor(B, "r", "t_v", DTYPE, "texture2d", precision="mediump" if MEDIUMP else "PRECISION")}
${layout_declare_tensor(B, "r", "t_weight_packed", DTYPE, WEIGHT_STORAGE, is_scalar_array=False, precision="mediump" if MEDIUMP else "PRECISION")}
${layout_declare_ubo(B, "ivec4", "in_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "N4", "1")}
// Summing each output texel's four products before accumulating took 1-4% less
// time on Adreno, and halved the rms error at 512 input channels on Mali-G715.
// It is a specialization constant that is never overridden, since the Adreno
// compiler schedules the K loop of an unconditional sum differently, which took
// 1-2% more time on Adreno 740 and 750.
${layout_declare_spec_const(C, "bool", "folded_accumulation", "true")}

#include "gemm/tile_utils/linear_fp_input_tile.glslh"
#include "gemm/tile_utils/linear_fp_packed_weight_tile_load.glslh"
#include "gemm/tile_utils/linear_fp_output_tile_fp_compute.glslh"

/*
 * Batched GEMM of Winograd: M_e = V_e U_e for each element e of a transformed
 * tile, with the tiles as rows, the input channels as the reduction and the
 * output channels as columns. t_v and t_m hold element e of slice c4 of tile t
 * at (t, e * C4 + c4) and the packed weight holds U_e as batch e.
 *
 * Each invocation computes 4 consecutive tiles for 8 output channels of one
 * element. t_v and t_m hold the tiles of the largest chunk, so a shorter chunk
 * computes up to 3 tiles past its last one, from stale texels of t_v, into
 * texels of t_m that the output transform does not read.
 */
void main() {
  const int tile_idx_m = int(gl_GlobalInvocationID.x);
  const int n4_start = int(gl_GlobalInvocationID.y) * TILE_N4;
  const int e = int(gl_GlobalInvocationID.z);

  const int K4 = div_up_4(in_sizes.z);
  const ivec2 m_extents = imageSize(t_m);
  const int tiles = m_extents.x;
  const int tile0 = tile_idx_m * TILE_M;
  // The element bound took 0.5% less time on Mali-G715, though the dispatch
  // spans the elements exactly.
  if (n4_start >= N4 || e * N4 >= m_extents.y || tile0 >= tiles) {
    return;
  }

  // Tiles past the last column reuse it so that the K loop needs no bounds
  // check; their results are never stored.
  int pos[TILE_M];
  [[unroll]] for (int m = 0; m < TILE_M; ++m) {
    pos[m] = min(tile0 + m, tiles - 1);
  }

  FPOutTile out_tile;
  initialize(out_tile);
  FPInputTile in_tile;
  FPWeightTile w_tile;

  for (int k4 = 0; k4 < K4; ++k4) {
    [[unroll]] for (int m = 0; m < TILE_M; ++m) {
      in_tile.data[m][0] =
          VEC4_T(texelFetch(t_v, ivec2(pos[m], e * K4 + k4), 0));
    }
    load_packed_weight_tile_clamped(w_tile, n4_start, k4, e, N4, K4);
    if (folded_accumulation) {
      fp_accumulate_with_fp_weight_folded(out_tile, in_tile, w_tile);
    } else {
      fp_accumulate_with_fp_weight(out_tile, in_tile, w_tile);
    }
  }

  [[unroll]] for (int m = 0; m < TILE_M; ++m) {
    [[unroll]] for (int n4 = 0; n4 < TILE_N4; ++n4) {
      if (tile0 + m < tiles && n4_start + n4 < N4) {
        imageStore(
            t_m,
            ivec2(tile0 + m, e * N4 + n4_start + n4),
            folded_accumulation ? out_tile.data[m][n4].wzyx
                                : out_tile.data[m][n4]);
      }
    }
  }
}
