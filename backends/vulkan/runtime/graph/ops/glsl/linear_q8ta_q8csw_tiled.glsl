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
#define T int

$if OUTPUT_STORAGE == "buffer":
  #define OUTPUT_BUFFER
$if INPUT_STORAGE == "buffer":
  #define INPUT_BUFFER
$if WEIGHT_STORAGE == "buffer":
  #define WEIGHT_BUFFER

#define TILE_M4 ${TILE_M4}
#define TILE_N4 ${TILE_N4}
#define TILE_K4 ${TILE_K4}

#define TILE_M ${TILE_M4 * 4}
#define TILE_N ${TILE_N4 * 4}
#define TILE_K ${TILE_K4 * 4}

${define_required_extensions(DTYPE)}

#extension GL_EXT_integer_dot_product : require

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_output", DTYPE, OUTPUT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_input", "int", INPUT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_qmat2", "int", WEIGHT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_sums", "float", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_scales", DTYPE, "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer", is_scalar_array=False)}

${layout_declare_spec_const(C, "uint", "apply_bias", "0")}

${layout_declare_ubo(B, "ivec4", "output_sizes")}
${layout_declare_ubo(B, "ivec4", "input_sizes")}

layout(push_constant) uniform restrict Block {
  float input_scale;
  int input_zp;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#include "linear_int8_input_tile_load.glslh"
#include "linear_int8_weight_tile_load.glslh"
#include "linear_fp_output_tile_int8_compute.glslh"
#include "linear_fp_output_tile_store.glslh"
#include "linear_scales_load.glslh"
#include "linear_weight_sums_load.glslh"
#include "linear_bias_load.glslh"

void main() {
  // Each thread writes out a 4 wide x 4 high tile of output values
  const uint out_tile_x = gl_GlobalInvocationID.x;
  const uint out_tile_y = gl_GlobalInvocationID.y;

  const uint n = out_tile_x * TILE_N;
  const uint m = out_tile_y * TILE_M;

  const uint n4 = div_4(n);
  const uint m4 = div_4(m);

  if (n >= output_sizes.x || m >= output_sizes.y) {
    return;
  }

  const uint M = output_sizes.y;
  const uint K4 = div_up_4(input_sizes.x);
  const uint N4 = div_up_4(output_sizes.x);

  Int8OutAccum out_accum;
  initialize(out_accum);

  Int8InputTile in_tile;
  Int8WeightTile weight_tile;

  for (int k4 = 0; k4 < K4; k4++) {
    load_input_tile(in_tile, k4, m4, K4);
    load_weight_tile(weight_tile, n4, k4, N4);

    accumulate(out_accum, in_tile, weight_tile);
  }

  FPPerOutChannelParams scales_tile;
  load_scales_tile(scales_tile, n4);

  FPPerOutChannelParams sums_tile;
  load_sums_tile(sums_tile, n4);

  FPOutTile out_tile;
  if (apply_bias > 0) {
    FPPerOutChannelParams bias_tile;
    load_bias_tile(bias_tile, uint(n4));

    compute(out_tile, out_accum, sums_tile, scales_tile, bias_tile);
  }
  else {
    compute(out_tile, out_accum, sums_tile, scales_tile);
  }

  if (M - m >= TILE_M) {
    write_output_tile_no_checks(out_tile, n4, m, N4, M);
  } else {
    write_output_tile_with_checks(out_tile, n4, m, N4, M);
  }
}
