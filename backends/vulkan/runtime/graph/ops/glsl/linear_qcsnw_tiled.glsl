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

#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require

void main() {
  // txcol stands for "texel column". One txcol corresponds to 4 scalar columns.
  $if TILE_TXCOLS > 1:
    const uint16_t global_wg_x = uint16_t(divup(out_sizes.x, 4 * TILE_TXCOLS));
    const uint16_t out_txcol = uint16_t(
      (gl_GlobalInvocationID.x % global_wg_x) * TILE_TXCOLS);
  $else:
    const uint16_t global_wg_x = uint16_t(divup4(out_sizes.x));
    const uint16_t out_txcol = uint16_t(gl_GlobalInvocationID.x % global_wg_x);

  const uint16_t out_row = uint16_t(
    (gl_GlobalInvocationID.x / global_wg_x) * TILE_ROWS);

  $if QUANT_NBITS == 4:
    const uint16_t weight_txcol = uint16_t(out_txcol / 2);

  if (out_row >= uint16_t(out_sizes.y)) {
    return;
  }

  VEC4_T mat1[TILE_ROWS];
  VEC4_T qmat2[4][TILE_TXCOLS];
  VEC4_T sums[TILE_ROWS][TILE_TXCOLS];

  VEC4_T scales[TILE_TXCOLS];
  $for c in range(TILE_TXCOLS):
    $if SCALES_STORAGE == "buffer":
      scales[${c}] = VEC4_T(t_scales[out_txcol + ${c}]);
    $else:
      scales[${c}] = VEC4_T(
        texelFetch(t_scales, u16vec2(out_txcol + ${c}, 0), 0));

  [[unroll]] for (int r = 0; r < TILE_ROWS; ++r) {
    $for c in range(TILE_TXCOLS):
      sums[r][${c}] = VEC4_T(0.0);
  }

  for (uint16_t pos = uint16_t(0), txpos = uint16_t(0);
       pos < uint16_t(in_sizes.x);
       pos += uint16_t(4), txpos += uint16_t(1)) {
    $if WEIGHT_STORAGE == "buffer":
      uint qmat2_bufi;
      uint weight_row_txstride = div4(weight_sizes.x);

    // Preload weight tensor
    [[unroll]] for (int r = 0; r < 4; r++) {
      $if QUANT_NBITS == 4:
        $for c in range(0, TILE_TXCOLS, 2):
          $if WEIGHT_STORAGE == "buffer":
            qmat2_bufi = (pos + r) * weight_row_txstride + weight_txcol;
            const u8vec4 packed_weight_tex = t_weight[qmat2_bufi + ${c}]
          $else:
            const uvec4 packed_weight_tex = texelFetch(
              t_weight, u16vec2(weight_txcol + ${c}, pos + r), 0);

          qmat2[r][${c}] = (VEC4_T((packed_weight_tex & 0xF0) >> 4) - 8.0);
          qmat2[r][${c + 1}] = (VEC4_T(packed_weight_tex & 0x0F) - 8.0);
      $else:
        $for c in range(TILE_TXCOLS):
          $if WEIGHT_STORAGE == "buffer":
            qmat2_bufi = (pos + r) * weight_row_txstride + out_txcol;
            qmat2[r][${c}] = t_weight[qmat2_bufi + ${c}];
          $else:
            qmat2[r][${c}] = VEC4_T(
              texelFetch(t_weight, u16vec2(out_txcol + ${c}, pos + r), 0));
    }

    $if IN_STORAGE == "buffer":
      uint in_row_txstride = div4(in_sizes.x);

    // Preload input tensor
    [[unroll]] for (int i = 0; i < TILE_ROWS; i++) {
      $if IN_STORAGE == "buffer":
        mat1[i] = t_in[(out_row + i) * in_row_txstride + txpos];
      $else:
        mat1[i] = VEC4_T(
          texelFetch(t_in, u16vec3(txpos, out_row + i, 0), 0));
    }

    // Accumulate output
    [[unroll]] for (int r = 0; r < TILE_ROWS; ++r) {
      $for c in range(TILE_TXCOLS):
        sums[r][${c}] += mat1[r].x * qmat2[0][${c}] +
                         mat1[r].y * qmat2[1][${c}] +
                         mat1[r].z * qmat2[2][${c}] +
                         mat1[r].w * qmat2[3][${c}];
    }
  }

  // Store to output tensor
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
