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
$if WEIGHT_STORAGE == "buffer":
  #define WEIGHT_BUFFER

#define TILE_M4 ${TILE_M4}
#define TILE_K4 ${TILE_K4}
#define TILE_N4 ${TILE_N4}

#define TILE_M ${TILE_M4 * 4}
#define TILE_K ${TILE_K4 * 4}
#define TILE_N ${TILE_N4 * 4}

${define_required_extensions(DTYPE)}

layout(std430) buffer;

#include "conv2d_common.glslh"

${layout_declare_tensor(B, "w", "t_output", DTYPE, OUTPUT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_input", DTYPE, INPUT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_packed_int8_weight", "int", WEIGHT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_scales", DTYPE, "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer", is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "output_sizes")}
${layout_declare_ubo(B, "ivec4", "input_sizes")}
${layout_declare_ubo(B, "Conv2DParams", "conv2d_params")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "apply_bias", "1")}

#include "linear_fp_input_tile_load.glslh"
#include "linear_int8_weight_tile_load.glslh"
#include "linear_fp_weight_scales_load.glslh"
#include "linear_fp_bias_load.glslh"
#include "linear_fp_output_tile_fp_int8_compute.glslh"
#include "linear_fp_output_tile_fp_compute.glslh"
#include "conv2d_fp_im2col_block_store.glslh"

void main() {
  // Each thread writes out a 4 wide x 4 high tile of output values
  const int out_tile_x = int(gl_GlobalInvocationID.x);
  const int out_tile_y = int(gl_GlobalInvocationID.y);

  const int n = int(out_tile_x * TILE_N);
  const int m = int(out_tile_y * TILE_M);

  const int n4 = div_4(n);
  const int m4 = div_4(m);

  // M = flattened output width, height, batches dims
  const int M = output_sizes.x * output_sizes.y * output_sizes.w;
  // N = output channels
  const int N = output_sizes.z;

  if (n >= N || m >= M) {
    return;
  }

  const int group_idx = n / conv2d_params.out_channels_per_group;
  const int input_k4_offset = conv2d_params.K4_per_group * group_idx;

  const int K4 = conv2d_params.K4;
  const int N4 = div_up_4(N);

  FPOutTile out_tile;
  initialize(out_tile);

  FPInputTile in_tile;
  Int8WeightTile int8_weight_tile;

  const bool dont_check_bounds = (M - m) >= TILE_M;

  if (dont_check_bounds) {
    for (int k4 = 0; k4 < conv2d_params.K4_per_group; k4++) {
      load_input_tile_no_checks(in_tile, k4 + input_k4_offset, m, K4, M);
      load_int8_weight_tile(int8_weight_tile, n4, k4, N4);
      fp_accumulate_with_int8_weight(out_tile, in_tile, int8_weight_tile);
    }
  } else {
    for (int k4 = 0; k4 < conv2d_params.K4_per_group; k4++) {
      load_input_tile_with_checks(in_tile, k4 + input_k4_offset, m, K4, M);
      load_int8_weight_tile(int8_weight_tile, n4, k4, N4);
      fp_accumulate_with_int8_weight(out_tile, in_tile, int8_weight_tile);
    }
  }

  FPPerOutChannelParams weight_scales_tile;
  load_weight_scales_tile(weight_scales_tile, n4);

  if (apply_bias > 0) {
    FPPerOutChannelParams bias_tile;
    load_bias_tile(bias_tile, n4);

    apply_weight_scales_and_biases(out_tile, weight_scales_tile, bias_tile);
  }
  else {
    apply_weight_scales(out_tile, weight_scales_tile);
  }

  write_im2col_tile_as_image(out_tile, n4, m);
}
