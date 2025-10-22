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

#define TILE_ROWS ${TILE_ROWS}
#define TILE_TXCOLS ${TILE_TXCOLS}

${define_required_extensions(DTYPE)}

$if WEIGHT_STORAGE == "buffer":
  ${define_required_extensions("int8")}

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

#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require

void main() {
  // txcol stands for "texel column". One txcol corresponds to 4 scalar columns.
  $if TILE_TXCOLS > 1:
    const int global_wg_x = divup(out_sizes.x, 4 * TILE_TXCOLS);
    const int out_txcol = (int(gl_GlobalInvocationID.x) % global_wg_x) * TILE_TXCOLS;
  $else:
    const int global_wg_x = divup4(out_sizes.x);
    const int out_txcol = int(gl_GlobalInvocationID.x) % global_wg_x;

  const int out_row = (int(gl_GlobalInvocationID.x) / global_wg_x) * TILE_ROWS;

  $if QUANT_NBITS == 4:
    const int weight_txcol = out_txcol / 2;

  if (out_row >= int(out_sizes.y)) {
    return;
  }

  VEC4_T sums[TILE_ROWS][TILE_TXCOLS];

  for (int r = 0; r < TILE_ROWS; ++r) {
    $for c in range(TILE_TXCOLS):
      sums[r][${c}] = VEC4_T(0.0);
  }

  const int in_row_txstride = div4(in_sizes.x);

  for (int pos = 0, txpos = 0;
       txpos < in_row_txstride;
       pos += 4, txpos += 1) {

    T mat1[TILE_ROWS][4];

    // Preload input tensor
    for (int i = 0; i < TILE_ROWS; i++) {
      $if IN_STORAGE == "buffer":
        VEC4_T tmp = t_in[(out_row + i) * in_row_txstride + txpos];
        mat1[i][0] = tmp.x;
        mat1[i][1] = tmp.y;
        mat1[i][2] = tmp.z;
        mat1[i][3] = tmp.w;
      $else:
        VEC4_T tmp = VEC4_T(texelFetch(t_in, ivec3(txpos, out_row + i, 0), 0));
        mat1[i][0] = tmp.x;
        mat1[i][1] = tmp.y;
        mat1[i][2] = tmp.z;
        mat1[i][3] = tmp.w;
    }

    $if WEIGHT_STORAGE == "buffer":
      uint qmat2_bufi;
      uint weight_row_txstride = div4(weight_sizes.x);

    // Preload weight tensor
    for (int r = 0; r < 4; r++) {
      VEC4_T qmat2[TILE_TXCOLS];
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

          qmat2[${c}] = (VEC4_T(packed_weight_tex >> 4) - 8.0);
          qmat2[${c + 1}] = (VEC4_T(packed_weight_tex & 0x0F) - 8.0);
      $else:
        $for c in range(TILE_TXCOLS):
          $if WEIGHT_STORAGE == "buffer":
            qmat2_bufi = (pos + r) * weight_row_txstride + out_txcol;
            qmat2[${c}] = t_weight[qmat2_bufi + ${c}];
          $else:
            qmat2[${c}] = VEC4_T(
              texelFetch(t_weight, ivec2(out_txcol + ${c}, pos + r), 0));

      for (int tr = 0; tr < TILE_ROWS; ++tr) {
        $for c in range(TILE_TXCOLS):
          sums[tr][${c}] += qmat2[${c}] * mat1[tr][r];
      }
    }
  }

  VEC4_T scales[TILE_TXCOLS];
  $for c in range(TILE_TXCOLS):
    $if SCALES_STORAGE == "buffer":
      scales[${c}] = VEC4_T(t_scales[out_txcol + ${c}]);
    $else:
      scales[${c}] = VEC4_T(
        texelFetch(t_scales, ivec2(out_txcol + ${c}, 0), 0));

  // Store to output tensor
  $if OUT_STORAGE == "buffer":
    uint out_bufi;
    uint out_row_txstride = div4(out_sizes.x);

  for (int r = 0; r < TILE_ROWS; ++r) {
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
