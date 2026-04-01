/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

$if STORAGE == "buffer":
  #define PACKED_INT8_OUTPUT_BUFFER
  #define PACKED_INT8_INPUT_BUFFER

#define TILE_M4 1
#define TILE_N4 1
#define TILE_K4 1

#define TILE_M 4
#define TILE_N 4
#define TILE_K 4

layout(std430) buffer;

#include "conv2d_common.glslh"

${layout_declare_tensor(B, "w", "t_packed_int8_output", "int", STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_packed_int8_input", "int", STORAGE, is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "im2col_sizes")}
// Sizes of the output image
${layout_declare_ubo(B, "ivec4", "output_sizes")}
// Sizes of the input image
${layout_declare_ubo(B, "ivec4", "input_sizes")}

${layout_declare_spec_const(C, "int", "apply_bias", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_stride_x", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_stride_y", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_padding_x", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_padding_y", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_dilation_x", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_dilation_y", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_kernel_size_x", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_kernel_size_y", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_in_channels_per_group", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_out_channels_per_group", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_K4_per_group", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_K4", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_K_per_group", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_logical_K", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_logical_K_per_group", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_groups", "1")}

layout(push_constant) uniform restrict Block {
  float inv_scale;
  int zp;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#include "conv2d_int8_output_tile_store.glslh"
#include "im2col_packed_int8_utils.glslh"

void main() {
  const int out_buf_idx = int(gl_GlobalInvocationID.x);

  Conv2dBlockExtents im2col_block_extents = make_block_extents(im2col_sizes);

  Conv2dBlockIndex im2col_block_idx = linear_idx_to_block_idx(
      out_buf_idx, im2col_block_extents);

  if (block_idx_out_of_bounds(im2col_block_idx, im2col_block_extents)) {
    return;
  }

  Im2ColBlockLoadIndices load_ixs = im2col_block_idx_to_load_ixs(
      im2col_block_idx);

  Conv2dBlockExtents input_block_extents = make_block_extents(input_sizes);

  const ivec4 input_zps = ivec4(pack_into_int32(ivec4(zp)));
  Int8OutTile int8_im2col_tile;
  int8_im2col_tile.data[0][0] = load_im2col_block(
      load_ixs, input_block_extents, zp, input_zps);

  store_packed_int8_output_tile(
      int8_im2col_tile, im2col_block_idx, im2col_block_extents);
}
