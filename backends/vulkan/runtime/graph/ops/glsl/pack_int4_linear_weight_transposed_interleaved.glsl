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

layout(std430) buffer;

$if STORAGE == "buffer" and NO_INT8_BUFFERS:
  ${layout_declare_tensor(B, "w", "t_qmat2", "uint", STORAGE, is_scalar_array=True)}
$else:
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
  $if NO_INT8_BUFFERS:
    #define UVEC4_T uvec4
  $else:
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
  return first * 16 + second;
}

$if NO_INT8_BUFFERS:
  uint extract_comp(const uint packed4, const uint idx) {
    return (packed4 >> (idx * 8)) & 0xFF;
  }

/*
 * This shader packs the weight tensor into a texture.
 *
 * The original tensor has a (W, H) shape of (K / 2, N) and each scalar element
 * is a uint8_t, which contains 2 packed 4 bit uint values.
 *
 * The transform performed by this shader is to first transpose the tensor, so
 * the shape of the packed tensor becomes (N / 2, K). Then, the 4 bit integers
 * are re-packed in groups of 8. For each 4 uint8_t values, the "left" 4-bits
 * of each value contain the 0, 1, 2, 3 4-bit values, and the "right" 4-bits of
 * each value contain the 4, 5, 6, 7 4-bit values.
 *
 * As a concrete example, consider the following weight tensor. The | demarks
 * the packing boundary, so 1| 2 represents a single uint8_t value with 1 in the
 * leftmost 4 bits and 2 in the rightmost 4 bits.
 *
 *  1| 2,  3| 4,  5| 6,  7| 8,
 *  9|10, 11|12, 13|14, 15|16,
 * 17|18, 19|20, 21|22, 23|24,
 * 25|26, 27|28, 29|30, 31|32,
 * 33|34, 35|36, 37|38, 39|40,
 * 41|42, 43|44, 45|46, 47|48,
 * 49|50, 51|52, 53|54, 55|56,
 * 57|58, 59|60, 61|62, 63|64,
 *
 * After packing, the packed tensor would contain
 *
 *  1|33,  9|41, 17|49, 25|57,
 *  2|34, 10|42, 18|50, 26|58,
 *  3|35, 11|43, 19|51, 27|59,
 *  4|36, 12|44, 20|52, 28|60,
 *  5|37, 13|45, 21|53, 29|61,
 *  6|38, 14|46, 22|54, 30|62,
 *  7|39, 15|47, 23|55, 31|63,
 *  8|40, 16|48, 24|56, 32|64,
 *
 * The purpose of interleaving is to make it easier to extract the unpacked
 * values in order using the u8vec4 vectorized type. With the packing in place,
 * The 4-bit values can be extracted via
 *
 * u8vec4 packed;
 * u8vec4 vals_0123 = (packed & 0xF0) >> 4;
 * u8vec4 vals_4567 = (packed | 0x0F);
 */
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
    $if NO_INT8_BUFFERS:
      t_qmat2[packed_pos.y * stride + packed_pos.x] = out_tex_1.x | (out_tex_1.y << 8) | (out_tex_1.z << 16) | (out_tex_1.w << 24);
      t_qmat2[(packed_pos.y + 1) * stride + packed_pos.x] = out_tex_2.x | (out_tex_2.y << 8) | (out_tex_2.z << 16) | (out_tex_2.w << 24);
    $else:
      t_qmat2[packed_pos.y * stride + packed_pos.x] = out_tex_1;
      t_qmat2[(packed_pos.y + 1) * stride + packed_pos.x] = out_tex_2;
  $else:
    imageStore(t_qmat2, packed_pos.xy, out_tex_1);
    imageStore(t_qmat2, ivec2(packed_pos.x, packed_pos.y + 1), out_tex_2);
}
