/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}
#define VEC4_T ${texel_load_type(DTYPE, INPUT_STORAGE)}
#define T ${texel_load_component_type(DTYPE, INPUT_STORAGE)}

// corresponds to the output width dim
#define TILE_M4 1
// corresponds to the output channels dim
#define TILE_K4 1

#define TILE_M 4

$if OUTPUT_STORAGE == "buffer":
  #define OUTPUT_BUFFER
$if INPUT_STORAGE == "buffer":
  #define INPUT_BUFFER

${define_required_extensions(DTYPE)}

layout(std430) buffer;

#define DEBUG_MODE
#include "conv2d_common.glslh"

${layout_declare_tensor(B, "w", "t_fp_output", DTYPE, OUTPUT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_packed_int8_output", "int", INPUT_STORAGE, is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "output_sizes")}

layout(push_constant) uniform restrict Block {
  float scale;
  int zp;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#include "linear_fp_input_tile.glslh"
#include "linear_int8_input_tile.glslh"

void load_packed_int8_tile(
    out Int8InputTile int8_tile,
    const Conv2dBlockIndex block_idx,
    const Conv2dBlockExtents block_extents) {
#ifdef INPUT_BUFFER
  const int buffer_idx = block_idx.data.y * block_extents.data_xz +
      block_idx.data.x * block_extents.data.z + block_idx.data.z;
  int8_tile.data[0][0] = t_packed_int8_output[buffer_idx];
#else
  int8_tile.data[0][0] = texelFetch(t_packed_int8_output, block_idx.data, 0);
#endif
}

VEC4_T
dequantize_8bit(const ivec4 val, const float q_scale, const int q_zero_point) {
  return VEC4_T(val - q_zero_point) * q_scale;
}

void unpack_and_dequantize(
    out FPInputTile fp_tile,
    const Int8InputTile int8_tile,
    const float q_scale,
    const int q_zero_point) {
  [[unroll]] for (int w = 0; w < 4; ++w) {
    int packed = int8_tile.data[0][0][w];
    fp_tile.data[w][0] = dequantize_8bit(
        ivec4(
            extract_8bit_from_packed_int_le(packed, 0),
            extract_8bit_from_packed_int_le(packed, 1),
            extract_8bit_from_packed_int_le(packed, 2),
            extract_8bit_from_packed_int_le(packed, 3)),
        q_scale,
        q_zero_point);
  }
}

void store_fp_output_texel(
    const Conv2dTensorIndex tidx,
    const VEC4_T out_texel) {
  imageStore(t_fp_output, tidx.data, out_texel);
}

void store_fp_tile(
    const FPInputTile block,
    const Conv2dBlockIndex block_idx) {
  Conv2dTensorIndex store_tidx = block_idx_to_tensor_idx(block_idx);
  [[unroll]] for (int w = 0; w < 4; w++) {
    store_fp_output_texel(store_tidx, block.data[w][0]);
    store_tidx.data.x++;
  }
}

void main() {
  Conv2dBlockIndex block_idx;
  block_idx.data = ivec3(gl_GlobalInvocationID);

  Conv2dBlockExtents block_extents = make_block_extents(output_sizes);
  if (block_idx_out_of_bounds(block_idx, block_extents)) {
    return;
  }

  Int8InputTile int8_tile;
  load_packed_int8_tile(int8_tile, block_idx, block_extents);

  FPInputTile fp_tile;
  unpack_and_dequantize(
      fp_tile, int8_tile, scale, zp);

  store_fp_tile(fp_tile, block_idx);
}
