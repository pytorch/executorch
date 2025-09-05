/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

layout(std430) buffer;

#include "indexing_utils.h"

${layout_declare_tensor(B, "w", "t_qmat2", "uint", STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_input", "uint", "buffer")}

layout(push_constant) uniform restrict Block {
  ivec4 qmat2_sizes;
  ivec2 orig_sizes;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

$if STORAGE == "buffer":
  #define BUFFER_WEIGHT

#include "qlinear_weight_pack_utils.glslh"

#define extract_4bit(input_block_data, col, row) \
  (extract_4bit_from_packed_uint_le(input_block_data[row], col))

/*
 * This shader packs the weight tensor into blocks for efficient consumption.
 *
 * The input tensor has shape [K/2, N] where each element is a uint8 containing
 * 2 packed 4-bit values. The logical tensor shape is [K, N] of 4-bit values.
 *
 * The transformation partitions the tensor into blocks of size 4x8 (4-bit values)
 * and transposes each block to 8x4, then packs the result so that each uvec4
 * contains an entire transposed block.
 *
 * Original block (4x8 4-bit values, shown as 2x8 uint8 values):
 * w00|w10, w20|w30,
 * w01|w11, w21|w31,
 * w02|w12, w22|w32,
 * w03|w13, w23|w33,
 * w04|w14, w24|w34,
 * w05|w15, w25|w35,
 * w06|w16, w26|w36,
 * w07|w17, w27|w37,
 *
 * Transposed block (8x4 4-bit values, packed into uvec4):
 * w00|w01, w02|w03, w04|w05, w06|w07
 * w10|w11, w12|w13, w14|w15, w16|w17
 * w20|w21, w22|w23, w24|w25, w26|w27
 * w30|w31, w32|w33, w34|w35, w36|w37
 */
void main() {
  // Each thread writes out 2 adjacent 8 wide x 4 high transposed block. Each
  // block is packed as one uvec4.
  ivec2 block_pos = ivec2(
      MUL_2(gl_GlobalInvocationID.x),
      gl_GlobalInvocationID.y);

  // There are K wide x N high 4-bit values in the original weight tensor
  const int input_width = orig_sizes.x;   // K
  const int input_height = orig_sizes.y;  // N

  const int input_width_uint = DIV_UP_8(input_width);

  // Original block spans 4 wide x 8 high 4-bit values. Since uint is used to
  // read the input tensor, each block spans 0.5 wide x 8 high uint values.
  const ivec2 block_start = ivec2(
      DIV_2(block_pos.x),
      MUL_8(block_pos.y));

  // Check bounds
  if (block_start.x >= input_width_uint || block_start.y >= input_height) {
    return;
  }

  // Read input block. Note that this block will contain the source data for
  // both output blocks, as it contains 1 wide x 8 high uint values, which is
  // equivalent to 8 wide x 8 high 4-bit values.
  uint input_block_data[8];

  // Read in 8 rows along the same column of uints, each uint contains 4 4-bit
  // values. This will be the source data for the transposed block.
  for (int i = 0; i < 8; ++i) {
    uint input_bufi = (block_start.y + i) * input_width_uint + block_start.x;
    input_block_data[i] = t_input[input_bufi];
  }

  for (int col_offset = 0; col_offset <= 4; col_offset+=4) {
    uvec4 output_block;

    output_block.x = pack_8x4bit_into_uint(
        extract_4bit(input_block_data, col_offset, 0),
        extract_4bit(input_block_data, col_offset, 1),
        extract_4bit(input_block_data, col_offset, 2),
        extract_4bit(input_block_data, col_offset, 3),
        extract_4bit(input_block_data, col_offset, 4),
        extract_4bit(input_block_data, col_offset, 5),
        extract_4bit(input_block_data, col_offset, 6),
        extract_4bit(input_block_data, col_offset, 7));

    output_block.y = pack_8x4bit_into_uint(
        extract_4bit(input_block_data, col_offset + 1, 0),
        extract_4bit(input_block_data, col_offset + 1, 1),
        extract_4bit(input_block_data, col_offset + 1, 2),
        extract_4bit(input_block_data, col_offset + 1, 3),
        extract_4bit(input_block_data, col_offset + 1, 4),
        extract_4bit(input_block_data, col_offset + 1, 5),
        extract_4bit(input_block_data, col_offset + 1, 6),
        extract_4bit(input_block_data, col_offset + 1, 7));

    output_block.z = pack_8x4bit_into_uint(
        extract_4bit(input_block_data, col_offset + 2, 0),
        extract_4bit(input_block_data, col_offset + 2, 1),
        extract_4bit(input_block_data, col_offset + 2, 2),
        extract_4bit(input_block_data, col_offset + 2, 3),
        extract_4bit(input_block_data, col_offset + 2, 4),
        extract_4bit(input_block_data, col_offset + 2, 5),
        extract_4bit(input_block_data, col_offset + 2, 6),
        extract_4bit(input_block_data, col_offset + 2, 7));

    output_block.w = pack_8x4bit_into_uint(
        extract_4bit(input_block_data, col_offset + 3, 0),
        extract_4bit(input_block_data, col_offset + 3, 1),
        extract_4bit(input_block_data, col_offset + 3, 2),
        extract_4bit(input_block_data, col_offset + 3, 3),
        extract_4bit(input_block_data, col_offset + 3, 4),
        extract_4bit(input_block_data, col_offset + 3, 5),
        extract_4bit(input_block_data, col_offset + 3, 6),
        extract_4bit(input_block_data, col_offset + 3, 7));

    const uint qmat2_texel_stride_x = DIV_UP_4(qmat2_sizes.x);
    write_transposed_weight_block(
        output_block,
        block_pos.x,
        block_pos.y,
        qmat2_texel_stride_x);

    if (MUL_8(block_start.x) + 4 >= input_width) {
      return;
    }
    // Otherwise, implement the block position to write to the next block in the
    // following iteration.
    block_pos.x += 1;
  }
}
