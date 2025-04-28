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

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint out_row = gl_GlobalInvocationID.y * TILE_ROWS;
  const uint out_col = gl_GlobalInvocationID.x << 2;

  if (out_col >= out_sizes.x || out_row >= out_sizes.y) {
    return;
  }

  VEC4_T a[TILE_ROWS];
  VEC4_T b[4];
  VEC4_T c[TILE_ROWS];

  $if SCALES_STORAGE == "buffer":
    const VEC4_T scales = VEC4_T(t_scales[out_col >> 2]);
  $else:
    const VEC4_T scales = VEC4_T(texelFetch(t_scales, ivec2(out_col >> 2, 0), 0));

  [[unroll]] for (int i = 0; i < TILE_ROWS; ++i) {
    c[i] = VEC4_T(0.0);
  }

  for (int pos = 0; pos < in_sizes.x; pos += 4) {
    // Preload weight tensor
    [[unroll]] for (int i = 0; i < 4; i++) {
      $if WEIGHT_STORAGE == "buffer":
        b[i] = t_weight[((pos + i) * out_sizes.x + out_col) >> 2];
      $else:
        b[i] = VEC4_T(texelFetch(t_weight, ivec2(out_col >> 2, pos + i), 0));
    }

    // Preload input tensor
    [[unroll]] for (int i = 0; i < TILE_ROWS; i++) {
      $if IN_STORAGE == "buffer":
        a[i] = t_in[((out_row + i) * in_sizes.x + pos) >> 2];
      $else:
        a[i] = VEC4_T(texelFetch(t_in, ivec3(pos >> 2, out_row + i, 0), 0));
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
