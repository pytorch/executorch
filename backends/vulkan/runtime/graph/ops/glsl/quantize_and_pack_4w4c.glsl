/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions(INPUT_STORAGE, DTYPE)}

#define PRECISION ${PRECISION}
#define VEC4_T ${texel_load_type(DTYPE, INPUT_STORAGE)}
#define T ${texel_load_component_type(DTYPE, INPUT_STORAGE)}

// corresponds to the input width dim
#define TILE_M4 1
// corresponds to the input channels dim
#define TILE_K4 1

#define TILE_M 4

$if OUTPUT_STORAGE == "buffer":
  #define OUTPUT_BUFFER
$if INPUT_STORAGE == "buffer":
  #define INPUT_BUFFER

layout(std430) buffer;

#include "conv2d_common.glslh"

${layout_declare_tensor(B, "w", "t_packed_int8_input", "int", OUTPUT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_fp_input", DTYPE, INPUT_STORAGE)}

${layout_declare_ubo(B, "ivec4", "input_sizes")}

layout(push_constant) uniform restrict Block {
  float inv_scale;
  int zp;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#include "conv2d_fp_input_tile_load.glslh"
#include "linear_int8_input_block.glslh"

void store_packed_int8_block(
    const Conv2dBlockIndex block_idx,
    const Conv2dBlockExtents block_extents,
    const Int8InputBlock packed_int8_block) {
#ifdef OUTPUT_BUFFER
  const int buffer_idx = block_idx.data.y * block_extents.data_xz +
      block_idx.data.x * block_extents.data.z + block_idx.data.z;
  t_packed_int8_input[buffer_idx] = packed_int8_block.data;
#else
  imageStore(t_packed_int8_input, block_idx.data, packed_int8_block.data);
#endif
}

void main() {
  Conv2dBlockIndex block_idx;
  block_idx.data = ivec3(gl_GlobalInvocationID);

  Conv2dBlockExtents block_extents = make_block_extents(input_sizes);
  if (block_idx_out_of_bounds(block_idx, block_extents)) {
    return;
  }

  FPInputTile fp_tile;
  load_fp_input_tile(fp_tile, block_idx);

  Int8InputBlock int8_block;
  quantize_and_pack(int8_block, fp_tile, inv_scale, zp);

  store_packed_int8_block(block_idx, block_extents, int8_block);
}
