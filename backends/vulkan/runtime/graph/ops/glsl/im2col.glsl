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

// Sizes of the im2col matrix of the convolution input
${layout_declare_ubo(B, "ivec4", "matrix_sizes")}
// Sizes of the input image
${layout_declare_ubo(B, "ivec4", "input_sizes")}
// Sizes of the output image
${layout_declare_ubo(B, "ivec4", "output_sizes")}

${layout_declare_ubo(B, "Conv2DParams", "conv2d_params")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#include "conv2d_fp_im2col_block_load.glslh"

#ifdef OUTPUT_BUFFER

void write_tile(
    const FPInputTile in_tile,
    const int k4,
    const int m_start,
    const int K4) {
  [[unroll]] for (int m = 0; m < TILE_M; m++) {
    t_output[(m_start + m) * K4 + k4] = in_tile.data[m][0];
  }
}

#else // OUTPUT_TEXTURE

void write_tile(
    const FPInputTile in_tile,
    const int k4,
    const int m_start,
    const int K4) {
  [[unroll]] for (int m = 0; m < TILE_M; m++) {
    imageStore(t_output, ivec3(k4, m_start + m, 0), vec4(in_tile.data[m][0]));
  }
}

#endif // OUTPUT_BUFFER

void main() {
  // Each thread writes out a 4 wide x 4 high block of the output matrix. The
  // thread position corresponds to the block index.
  const int k4 = int(gl_GlobalInvocationID.x);
  const int m4 = int(gl_GlobalInvocationID.y);

  // Convert block idx to tensor idx
  const int k = mul_4(k4);
  const int m = mul_4(m4);

  const int in_channels_per_group = input_sizes.z / conv2d_params.groups;

  // Logical K dim size (unpadded)
  const int logical_K = conv2d_params.logical_K;
  // Physical K dim, which contains padding elements
  const int K = matrix_sizes.x;

  // M dim, which represents the number of flattened output width, height,
  // batches. Unlike K, there is no difference between the physical and logical
  // sizes.
  const int M = matrix_sizes.y;

  if (k >= K || m >= M) {
    return;
  }

  FPInputTile in_tile;
  load_input_im2col_tile(in_tile, k4, m4, logical_K, M);

  // Number of texels in the x dim of the output matrix
  const int K4 = div_4(K);
  write_tile(in_tile, k4, m, K4);
}
