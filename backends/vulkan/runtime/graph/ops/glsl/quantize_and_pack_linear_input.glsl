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

${define_required_extensions(DTYPE)}

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_packed_int8_input", "int", OUTPUT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_input", DTYPE, INPUT_STORAGE, is_scalar_array=False)}

$if GRANULARITY == "per_channel":
  ${layout_declare_tensor(B, "r", "t_scale", DTYPE, "buffer")}

${layout_declare_ubo(B, "ivec4", "input_sizes")}

layout(push_constant) uniform restrict Block {
  float inv_scale;
  int zp;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#include "linear_int8_input_block.glslh"
#include "linear_fp_input_tile_load.glslh"

void main() {
  // Each input block contains 4x4 int8 quantized values, which are packed into
  // a ivec4. k4 and m4 represent the "block index" of the current block being
  // processed.
  int k4 = int(gl_GlobalInvocationID.x);
  int m4 = int(gl_GlobalInvocationID.y);

  const int K = input_sizes.x;
  const int M = input_sizes.y;

  // K4 and M4 represent the number of blocks in each dimension.
  const int K4 = div_up_4(K);
  const int M4 = div_up_4(M);

  if (k4 >= K4 || m4 >= M4) {
    return;
  }

  // row of the input tensor to start loading from. Note the input tensor is
  // interpreted as a t
  const int m = mul_4(m4);

  const bool dont_check_bounds = (M - m) >= 4;

  FPInputTile in_tile;
  if (dont_check_bounds) {
    load_input_tile_no_checks(in_tile, k4, m, K4, M);
  } else {
    load_input_tile_with_checks(in_tile, k4, m, K4, M);
  }

  Int8InputBlock packed_block;
  quantize_and_pack(packed_block, in_tile);

  write_block(packed_block, k4, m4, K4);
}
