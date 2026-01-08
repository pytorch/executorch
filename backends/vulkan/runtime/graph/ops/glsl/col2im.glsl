/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}
#define VEC4_T ${texel_load_type(DTYPE, OUTPUT_STORAGE)}
#define T ${texel_load_component_type(DTYPE, OUTPUT_STORAGE)}

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

${define_required_extensions(DTYPE)}

layout(std430) buffer;

#include "conv2d_common.glslh"

${layout_declare_tensor(B, "w", "t_output", DTYPE, OUTPUT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_input", DTYPE, INPUT_STORAGE, is_scalar_array=False)}

// Sizes of the convolution output image
${layout_declare_ubo(B, "ivec4", "output_sizes")}
// Sizes of the convolution input image
${layout_declare_ubo(B, "ivec4", "input_sizes")}
// Sizes of the im2col matrix of the convolution output
${layout_declare_ubo(B, "ivec4", "matrix_sizes")}

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

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#include "conv2d_fp_im2col_block_store.glslh"

#ifdef INPUT_BUFFER

void load_matrix_tile(
    out FPOutTile tile,
    const int n4,
    const int m_start,
    const int N4) {
  [[unroll]] for (int m = 0; m < TILE_M; m++) {
    tile.data[m][0] = t_input[(m_start + m) * N4 + n4];
  }
}

#else // INPUT_TEXTURE

void load_matrix_tile(
    out FPOutTile tile,
    const int n4,
    const int m_start,
    const int N4) {
  [[unroll]] for (int m = 0; m < TILE_M; m++) {
    tile.data[m][0] = texelFetch(
        t_input, ivec3(n4, m_start + m, 0), 0);
  }
}

#endif // INPUT_BUFFER

void main() {
  // Each thread loads and writes a 4 wide x 4 high block of the matrix
  const int n4 = int(gl_GlobalInvocationID.x);
  const int m4 = int(gl_GlobalInvocationID.y);

  const int n = mul_4(n4);
  const int m = mul_4(m4);

  if (n >= matrix_sizes.x || m >= matrix_sizes.y) {
    return;
  }

  FPOutTile tile;

  const int N4 = div_4(matrix_sizes.x);
  load_matrix_tile(tile, n4, m, N4);
  write_im2col_tile_as_image(tile, n4, m);
}
