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

${define_required_extensions(DTYPE)}

$if WEIGHT_STORAGE == "buffer":
  ${define_required_extensions("int8")}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, OUT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, IN_STORAGE, is_scalar_array=False)}
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
  const uint16_t out_width_ntexels = uint16_t(divup4(out_sizes.x));
  const uint16_t out_col = uint16_t((gl_GlobalInvocationID.x % out_width_ntexels) << 2);
  const uint16_t out_row = uint16_t((gl_GlobalInvocationID.x / out_width_ntexels) * TILE_ROWS);

  if (out_row >= uint16_t(out_sizes.y)) {
    return;
  }

  VEC4_T a[TILE_ROWS];
  VEC4_T b[4];
  VEC4_T c[TILE_ROWS];

  $if SCALES_STORAGE == "buffer":
    const VEC4_T scales = VEC4_T(t_scales[int(out_col >> 2)]);
  $else:
    const VEC4_T scales = VEC4_T(texelFetch(t_scales, u16vec2(out_col >> 2, 0), 0));

  [[unroll]] for (int i = 0; i < TILE_ROWS; ++i) {
    c[i] = VEC4_T(0.0);
  }

  for (uint16_t pos = uint16_t(0); pos < uint16_t(in_sizes.x); pos += uint16_t(4)) {
    // Preload weight tensor
    [[unroll]] for (int i = 0; i < 4; i++) {
      $if WEIGHT_STORAGE == "buffer":
        b[i] = t_weight[((pos + i) * out_sizes.x + out_col) >> 2];
      $else:
        b[i] = VEC4_T(texelFetch(t_weight, u16vec2(out_col >> 2, pos + i), 0));
    }

    // Preload input tensor
    [[unroll]] for (int i = 0; i < TILE_ROWS; i++) {
      $if IN_STORAGE == "buffer":
        a[i] = t_in[((out_row + i) * in_sizes.x + pos) >> 2];
      $else:
        a[i] = VEC4_T(texelFetch(t_in, u16vec3(pos >> 2, out_row + i, 0), 0));
    }

    // Accumulate output
    [[unroll]] for (int i = 0; i < TILE_ROWS; ++i) {
        c[i] += a[i].x * b[0] + a[i].y * b[1] + a[i].z * b[2] + a[i].w * b[3];
    }
  }

  // Store to output tensor
  [[unroll]] for (int i = 0; i < TILE_ROWS; ++i) {
    $if OUT_STORAGE == "buffer":
      if (out_row + i < out_sizes.y) {
        t_out[((out_row + i) * out_sizes.x + out_col) >> 2] = c[i] * scales;
      }
    $else:
      imageStore(t_out, ivec3(out_col >> 2, out_row + i, 0), c[i] * scales);
  }
}
