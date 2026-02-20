/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions(IO_STORAGE, DTYPE)}
${define_required_extensions("buffer", DTYPE)}

#define PRECISION ${PRECISION}
#define VEC4_T ${texel_load_type(DTYPE, IO_STORAGE)}
#define T ${texel_load_component_type(DTYPE, IO_STORAGE)}

$if IO_STORAGE == "buffer":
  #define OUTPUT_BUFFER
  #define INPUT_BUFFER
$if WEIGHT_STORAGE == "buffer":
  #define WEIGHT_BUFFER

#define TILE_M4 ${TILE_M4}
#define TILE_K4 ${TILE_K4}
#define TILE_N4 ${TILE_N4}

#define TILE_M ${TILE_M4 * 4}
#define TILE_K ${TILE_K4 * 4}
#define TILE_N ${TILE_N4 * 4}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_output", DTYPE, IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_input", DTYPE, IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_packed_int8_weight", "int", WEIGHT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_scales", DTYPE, "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer", is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "output_sizes")}
${layout_declare_ubo(B, "ivec4", "input_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "uint", "apply_bias", "0")}

#include "linear_fp_input_tile_load.glslh"
#include "linear_int8_weight_tile_load.glslh"
#include "linear_fp_weight_tile.glslh"
#include "linear_fp_output_tile_fp_compute.glslh"
#include "linear_fp_output_tile_fp_int8_compute.glslh"
#include "linear_fp_output_tile_store.glslh"
#include "linear_fp_weight_scales_load.glslh"
#include "linear_fp_bias_load.glslh"

void main() {
  // Each thread writes out a 4 wide x 4 high tile of output values
  const int out_tile_x = int(gl_GlobalInvocationID.x);
  const int out_tile_y = int(gl_GlobalInvocationID.y);

  const int n = out_tile_x * TILE_N;
  const int m = out_tile_y * TILE_M;

  const int n4 = div_4(n);
  const int m4 = div_4(m);

  if (n >= output_sizes.x || m >= output_sizes.y) {
    return;
  }

  const int M = input_sizes.y;
  const int K4 = div_up_4(input_sizes.x);
  const int N4 = div_up_4(output_sizes.x);

  FPOutTile out_tile;
  initialize(out_tile);

  FPInputTile in_tile;
  Int8WeightTile int8_weight_tile;

  const bool dont_check_bounds = (M - m) >= TILE_M;
  if (dont_check_bounds) {
    for (int k4 = 0; k4 < K4; k4 += TILE_K4) {
      load_input_tile_no_checks(in_tile, k4, m, K4, M);
      load_int8_weight_tile(int8_weight_tile, n4, k4, N4);
      fp_accumulate_with_int8_weight(out_tile, in_tile, int8_weight_tile);
    }
  } else {
    for (int k4 = 0; k4 < K4; k4 += TILE_K4) {
      load_input_tile_with_checks(in_tile, k4, m, K4, M);
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

  if (dont_check_bounds) {
    write_output_tile_no_checks(out_tile, n4, m, N4, M);
  } else {
    write_output_tile_with_checks(out_tile, n4, m, N4, M);
  }
}
