/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions(OUT_STORAGE, DTYPE)}

$if WEIGHT_STORAGE == "buffer":
  ${define_required_extensions("buffer", "int8")}

#define PRECISION ${PRECISION}

#define T ${buffer_scalar_type(DTYPE)}
#define VEC4_T ${buffer_gvec_type(DTYPE, 4)}

#define TILE_ROWS ${TILE_ROWS}
#define TILE_TXCOLS ${TILE_TXCOLS}

#define NGROUPS 8
#define NWORKERS 8

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, OUT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, IN_STORAGE, is_scalar_array=False)}
$if QUANT_NBITS == 4:
  ${layout_declare_tensor(B, "r", "t_weight", "uint8", WEIGHT_STORAGE, is_scalar_array=False)}
$else:
  ${layout_declare_tensor(B, "r", "t_weight", "int8", WEIGHT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_scales", DTYPE, SCALES_STORAGE, is_scalar_array=False)}

layout(push_constant) uniform restrict Block {
  ivec4 out_sizes;
  ivec4 in_sizes;
  ivec4 weight_sizes;
};

#include "indexing_utils.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

shared VEC4_T partial_sums[NGROUPS][NWORKERS][TILE_ROWS][TILE_TXCOLS];

void main() {
  // txcol stands for "texel column". One txcol corresponds to 4 scalar columns.
  $if TILE_TXCOLS > 1:
    const uint global_wg_x = uint(divup(out_sizes.x, 4 * TILE_TXCOLS));
    const uint out_txcol = uint(
      (gl_GlobalInvocationID.x % global_wg_x) * TILE_TXCOLS);
  $else:
    const uint global_wg_x = uint(divup4(out_sizes.x));
    const uint out_txcol = uint(gl_GlobalInvocationID.x % global_wg_x);

  const uint out_row = uint(
    (gl_GlobalInvocationID.x / global_wg_x) * TILE_ROWS);

  $if QUANT_NBITS == 4:
    const uint weight_txcol = uint(out_txcol / 2);

  const int gid = int(gl_LocalInvocationID.x); // group id
  const int wid = int(gl_LocalInvocationID.z); // worker id

  if (out_row >= out_sizes.y) {
    return;
  }

  VEC4_T mat1[TILE_ROWS];
  VEC4_T qmat2[4][TILE_TXCOLS];
  VEC4_T local_sums[TILE_ROWS][TILE_TXCOLS];

  [[unroll]] for (int r = 0; r < TILE_ROWS; ++r) {
    $for c in range(TILE_TXCOLS):
      local_sums[r][${c}] = VEC4_T(0.0);
  }

  VEC4_T scales[TILE_TXCOLS];
  $for c in range(TILE_TXCOLS):
    $if SCALES_STORAGE == "buffer":
      scales[${c}] = VEC4_T(t_scales[out_txcol + ${c}]);
    $else:
      scales[${c}] = VEC4_T(
        texelFetch(t_scales, ivec2(out_txcol + ${c}, 0), 0));

  for (int pos = (4 * wid), txpos = wid;
       pos < in_sizes.x;
       pos += (4 * NWORKERS), txpos += NWORKERS) {
    $if WEIGHT_STORAGE == "buffer":
      uint qmat2_bufi;
      uint weight_row_txstride = div4(weight_sizes.x);

    // Preload weight tensor
    [[unroll]] for (int r = 0; r < 4; r++) {
      $if QUANT_NBITS == 4:
        $if WEIGHT_STORAGE == "buffer":
          u8vec4 packed_weight_tex;
        $else:
          uvec4 packed_weight_tex;

        $for c in range(0, TILE_TXCOLS, 2):
          $if WEIGHT_STORAGE == "buffer":
            qmat2_bufi = (pos + r) * weight_row_txstride + weight_txcol;
            packed_weight_tex = t_weight[qmat2_bufi + ${c}]
          $else:
            packed_weight_tex = texelFetch(
              t_weight, ivec2(weight_txcol + ${c}, pos + r), 0);

          qmat2[r][${c}] = (VEC4_T((packed_weight_tex & 0xF0) >> 4) - 8.0);
          qmat2[r][${c + 1}] = (VEC4_T(packed_weight_tex & 0x0F) - 8.0);
      $else:
        $for c in range(TILE_TXCOLS):
          $if WEIGHT_STORAGE == "buffer":
            qmat2_bufi = (pos + r) * weight_row_txstride + out_txcol;
            qmat2[r][${c}] = t_weight[qmat2_bufi + ${c}];
          $else:
            qmat2[r][${c}] = VEC4_T(
              texelFetch(t_weight, ivec2(out_txcol + ${c}, pos + r), 0));
    }

    $if IN_STORAGE == "buffer":
      uint in_row_txstride = div4(in_sizes.x);

    // Preload input tensor
    [[unroll]] for (int i = 0; i < TILE_ROWS; i++) {
      $if IN_STORAGE == "buffer":
        mat1[i] = t_in[(out_row + i) * in_row_txstride + txpos];
      $else:
        mat1[i] = VEC4_T(
          texelFetch(t_in, ivec3(txpos, out_row + i, 0), 0));
    }

    // Accumulate partial output
    [[unroll]] for (int r = 0; r < TILE_ROWS; ++r) {
      $for c in range(TILE_TXCOLS):
        local_sums[r][${c}] += mat1[r].x * qmat2[0][${c}] +
                               mat1[r].y * qmat2[1][${c}] +
                               mat1[r].z * qmat2[2][${c}] +
                               mat1[r].w * qmat2[3][${c}];
    }
  }

  [[unroll]] for (int r = 0; r < TILE_ROWS; ++r) {
    $for c in range(TILE_TXCOLS):
      partial_sums[gid][wid][r][${c}] = local_sums[r][${c}];
  }

  memoryBarrierShared();
  barrier();

  if (wid != 0) {
    return;
  }

  VEC4_T sums[TILE_ROWS][TILE_TXCOLS];

  for (int r = 0; r < TILE_ROWS; ++r) {
    $for c in range(TILE_TXCOLS):
      sums[r][${c}] = VEC4_T(0.0);

    [[unroll]] for (int worker = 0; worker < NWORKERS; ++worker) {
      $for c in range(TILE_TXCOLS):
        sums[r][${c}] += partial_sums[gid][worker][r][${c}];
    }
  }

  $if OUT_STORAGE == "buffer":
    uint out_bufi;
    uint out_row_txstride = div4(out_sizes.x);

  [[unroll]] for (int r = 0; r < TILE_ROWS; ++r) {
    $for c in range(TILE_TXCOLS):
      $if OUT_STORAGE == "buffer":
        if (out_row + r < out_sizes.y) {
          out_bufi = (out_row + r) * out_row_txstride + out_txcol;
          t_out[out_bufi + ${c}] = sums[r][${c}] * scales[${c}];
        }
      $else:
        imageStore(
          t_out,
          ivec3(out_txcol + ${c}, out_row + r, 0),
          sums[r][${c}] * scales[${c}]);
  }
}
