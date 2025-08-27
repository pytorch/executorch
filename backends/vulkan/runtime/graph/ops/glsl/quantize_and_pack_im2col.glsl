/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#extension GL_EXT_debug_printf : enable
#define DEBUG_MODE

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

${define_required_extensions(DTYPE)}

#extension GL_EXT_debug_printf : enable

layout(std430) buffer;

#include "conv2d_common.glslh"

${layout_declare_tensor(B, "w", "t_output", "int", OUTPUT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_input", DTYPE, INPUT_STORAGE, is_scalar_array=False)}

// Sizes of the im2col matrix of the convolution input
${layout_declare_ubo(B, "ivec4", "matrix_sizes")}
// Sizes of the input image
${layout_declare_ubo(B, "ivec4", "input_sizes")}
// Sizes of the output image
${layout_declare_ubo(B, "ivec4", "output_sizes")}

${layout_declare_ubo(B, "Conv2DParams", "conv2d_params")}

layout(push_constant) uniform restrict Block {
  float inv_scale;
  int zp;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#define DEBUG_MODE

#include "conv2d_fp_im2col_block_load.glslh"
#include "linear_int8_input_block.glslh"

void main() {
  // Each thread writes out a 4 wide x 4 high block of the output matrix. The
  // thread id corresponds to the block index.
  const int k4 = int(gl_GlobalInvocationID.x);
  const int m4 = int(gl_GlobalInvocationID.y);

  bool should_print = k4 == 0 && m4 == 0;

  // Convert block idx to tensor idx
  const int k = mul_4(k4);
  const int m = mul_4(m4);

  // Compute the "true" im2col matrix dimensions based on input/output image
  // sizes. The sizes of the physical im2col matrix are padded to multiples of 4
  // to accomodate vectorized load/store.

  // flattened convolution window size
  const int K = input_sizes.z * conv2d_params.kernel_size.x *
      conv2d_params.kernel_size.y;
  // flattened output width, height, batches
  const int M = output_sizes.x * output_sizes.y * output_sizes.w;

  if (should_print) {
    debugPrintfEXT("K: %d, M: %d \\n", K, M);
    debugPrintfEXT("matrix_sizes.x: %d, matrix_sizes.y: %d \\n", matrix_sizes.x, matrix_sizes.y);
    debugPrintfEXT("output_sizes: %d, %d, %d, %d\\n", output_sizes.x, output_sizes.y, output_sizes.z, output_sizes.w);

  }

  // Check if tensor indices are out of bounds
  if (k >= K || m >= M) {
    return;
  }

  FPInputTile in_tile;
  load_input_im2col_tile(in_tile, k4, m4, K, M);

  Int8InputBlock packed_block;
  quantize_and_pack(packed_block, in_tile);

  // Number of texels in the x dim of the output matrix
  const int K4 = div_4(matrix_sizes.x);
  write_block(packed_block, k4, m4, K4);
}
