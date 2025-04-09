/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

$if not NO_INT8_BUFFERS:
  ${define_required_extensions("uint8")}
$if STORAGE == "buffer":
  ${define_required_extensions("int8")}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_qmat2", "uint8", STORAGE, is_scalar_array=False)}
$if NO_INT8_BUFFERS:
  ${layout_declare_tensor(B, "r", "nchw_4x2", "uint", "buffer")}
$else:
  ${layout_declare_tensor(B, "r", "nchw_4x2", "uint8", "buffer")}

layout(push_constant) uniform restrict Block {
  ivec4 qmat2_sizes;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

$if NO_INT8_BUFFERS:
  #define BUF_T uint
$else:
  #define BUF_T uint8_t

$if STORAGE == "buffer":
  #define UVEC4_T u8vec4
$else:
  #define UVEC4_T uvec4

uint get_first(const BUF_T packed) {
  return (packed & 0xF0) >> 4;
}

uint get_second(const BUF_T packed) {
  return packed & 0x0F;
}

uint combine(const uint first, const uint second) {
  return (first << 4 | second);
}

$if NO_INT8_BUFFERS:
  uint extract_comp(const uint packed4, const uint idx) {
    return (packed4 >> (idx * 8)) & 0xFF;
  }

void main() {
  // Each thread writes 2 output texels along the height axis
  ivec2 packed_pos = ivec2(
      gl_GlobalInvocationID.x,
      gl_GlobalInvocationID.y << 1);

  // The packed tensor is width packed
  if ((packed_pos.x << 2) >= qmat2_sizes.x || packed_pos.y >= qmat2_sizes.y) {
    return;
  }

  int out_col = packed_pos.x << 3;
  int out_row = packed_pos.y;

  int in_col = out_row;
  int in_int8_col = in_col >> 1;
  int in_row = out_col;

  int in_numrows = qmat2_sizes.x << 1;
  int in_numcols = qmat2_sizes.y;
  int in_num_int8_cols = qmat2_sizes.y >> 1;

  uint in_vals[8][2];
  for (int r = 0; r < 8; ++r) {
    if (in_row + r < in_numrows) {
      uint scalar_idx = (in_row + r) * in_num_int8_cols + in_int8_col;
      $if NO_INT8_BUFFERS:
        BUF_T in_val_packed_texel = nchw_4x2[scalar_idx >> 2];
        const uint packed_idx = scalar_idx % 4;
        uint in_val_packed = extract_comp(in_val_packed_texel, packed_idx);
      $else:
        BUF_T in_val_packed = nchw_4x2[scalar_idx];

      in_vals[r][0] = get_first(in_val_packed);
      in_vals[r][1] = get_second(in_val_packed);
    } else {
      in_vals[r][0] = uint(0);
      in_vals[r][1] = uint(0);
    }
  }

  UVEC4_T out_tex_1 = UVEC4_T(
      combine(in_vals[0][0], in_vals[4][0]),
      combine(in_vals[1][0], in_vals[5][0]),
      combine(in_vals[2][0], in_vals[6][0]),
      combine(in_vals[3][0], in_vals[7][0]));

  UVEC4_T out_tex_2 = UVEC4_T(
      combine(in_vals[0][1], in_vals[4][1]),
      combine(in_vals[1][1], in_vals[5][1]),
      combine(in_vals[2][1], in_vals[6][1]),
      combine(in_vals[3][1], in_vals[7][1]));

  $if STORAGE == "buffer":
    int stride = qmat2_sizes.x >> 2;
    t_qmat2[packed_pos.y * stride + packed_pos.x] = out_tex_1;
    t_qmat2[(packed_pos.y + 1) * stride + packed_pos.x] = out_tex_2;
  $else:
    imageStore(t_qmat2, packed_pos.xy, out_tex_1);
    imageStore(t_qmat2, ivec2(packed_pos.x, packed_pos.y + 1), out_tex_2);
}
