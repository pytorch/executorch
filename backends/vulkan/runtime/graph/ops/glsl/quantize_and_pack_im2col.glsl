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

$if OUTPUT_STORAGE == "buffer":
  #define OUTPUT_BUFFER
$if INPUT_STORAGE == "buffer":
  #define INPUT_BUFFER

#define TILE_M4 1
#define TILE_N4 1
#define TILE_K4 1

#define TILE_M 4
#define TILE_N 4
#define TILE_K 4

layout(std430) buffer;

#include "conv2d_common.glslh"

${layout_declare_tensor(B, "w", "t_packed_int8_input", "int", OUTPUT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_input", DTYPE, INPUT_STORAGE, is_scalar_array=False)}

// Sizes of the im2col matrix of the convolution input
${layout_declare_ubo(B, "ivec4", "matrix_sizes")}
// Sizes of the input image
${layout_declare_ubo(B, "ivec4", "input_sizes")}
// Sizes of the output image
${layout_declare_ubo(B, "ivec4", "output_sizes")}

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

#include "conv2d_fp_im2col_block_load.glslh"
#include "linear_int8_input_block.glslh"

void main() {
  // The quantized and packed im2col matrix can be conceptualized as a 2D matrix
  // with K/4 columns and M/4 rows. Each element of the matrix is a ivec4 which
  // contains packed data for a 4 wide x 4 high block of the original im2col
  // matrix. Each shader invocation works on writing out one ivec4, i.e. one
  // block of the quantized and packed matrix.

  // Thread id corresponds to the block index
  const int k4 = int(gl_GlobalInvocationID.x);
  const int m4 = int(gl_GlobalInvocationID.y);

  // Convert block idx to tensor idx
  const int k = mul_4(k4);
  const int m = mul_4(m4);

  const int logical_K = conv2d_params_logical_K;
  // Similarly, compute the logical size of the M dim.
  const int logical_M = output_sizes.x * output_sizes.y * output_sizes.w;

  // Check if tensor indices are out of bounds
  if (k >= logical_K || m >= logical_M) {
    return;
  }

  FPInputTile in_tile;
  load_input_im2col_tile(in_tile, k4, m4, logical_K, logical_M);

  Int8InputBlock packed_block;
  quantize_and_pack(packed_block, in_tile, inv_scale, zp);

  // Number of texels in the x dim of the output matrix
  const int K4 = div_4(matrix_sizes.x);
  write_block(packed_block, k4, m4, K4);
}
