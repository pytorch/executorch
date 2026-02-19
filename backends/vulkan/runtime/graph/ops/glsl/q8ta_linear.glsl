/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions("buffer", DTYPE)}

#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_integer_dot_product : require

#define PRECISION ${PRECISION}
#define VEC4_T ${texel_load_type(DTYPE, "buffer")}
#define T int

#define PACKED_INT8_INPUT_BUFFER
$if WEIGHT_STORAGE == "buffer":
  #define WEIGHT_BUFFER

#define TILE_M4 ${TILE_M4}
#define TILE_K4 ${TILE_K4}
#define TILE_N4 ${TILE_N4}

#define TILE_M ${TILE_M4 * 4}
#define TILE_K ${TILE_K4 * 4}
#define TILE_N ${TILE_N4 * 4}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_packed_int8_output", "int", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_packed_int8_input", "int", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_packed_int8_weight", "int", WEIGHT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_sums", "int", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_scales", DTYPE, "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer", is_scalar_array=False)}

${layout_declare_spec_const(C, "int", "apply_bias", "0")}

${layout_declare_ubo(B, "ivec4", "output_sizes")}
${layout_declare_ubo(B, "ivec4", "input_sizes")}

layout(push_constant) uniform restrict Block {
  float input_scale;
  int input_zp;
  float output_inv_scale;
  int output_zp;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#include "common.glslh"
#include "linear_int8_input_tile_load.glslh"
#include "linear_int8_weight_tile_load.glslh"
#include "linear_fp_output_tile_int8_int8_compute.glslh"
#include "linear_fp_weight_scales_load.glslh"
#include "linear_int_weight_sums_load.glslh"
#include "linear_fp_bias_load.glslh"

void main() {
  const int out_tile_x = int(gl_GlobalInvocationID.x);
  const int out_tile_y = int(gl_GlobalInvocationID.y);

  const int n = out_tile_x * TILE_N;
  const int m = out_tile_y * TILE_M;

  const int n4 = div_4(n);
  const int m4 = div_4(m);

  if (n >= output_sizes.x || m >= output_sizes.y) {
    return;
  }

  const int M = output_sizes.y;
  const int K4 = div_up_4(input_sizes.x);
  const int N4 = div_up_4(output_sizes.x);

  Int32Accum out_accum;
  initialize(out_accum);

  Int8InputTile int8_in_tile;
  Int8WeightTile int8_weight_tile;

  for (int k4 = 0; k4 < K4; k4 += TILE_K4) {
    load_int8_input_tile(int8_in_tile, k4, m4, K4);
    load_int8_weight_tile(int8_weight_tile, n4, k4, N4);

    int_accumulate_with_int8_weight(out_accum, int8_in_tile, int8_weight_tile);
  }

  FPPerOutChannelParams weight_scales_tile;
  load_weight_scales_tile(weight_scales_tile, n4);

  IntPerOutChannelParams weight_sums_tile;
  load_weight_sums_tile(weight_sums_tile, n4);

  FPOutTile out_tile;
  initialize(out_tile);

  if (apply_bias > 0) {
    FPPerOutChannelParams bias_tile;
    load_bias_tile(bias_tile, n4);

    accumulate_out_tile_with_int_accum(
        out_tile,
        out_accum,
        input_scale,
        input_zp,
        weight_sums_tile,
        weight_scales_tile,
        bias_tile);
  }
  else {
    accumulate_out_tile_with_int_accum(
        out_tile,
        out_accum,
        input_scale,
        input_zp,
        weight_sums_tile,
        weight_scales_tile);
  }

  // Quantize float output tile to int8 and write in PACKED_INT8_4H4W format
  const int M4 = div_up_4(M);

  [[unroll]] for (int tile_m4 = 0; tile_m4 < TILE_M4; ++tile_m4) {
    if (m4 + tile_m4 >= M4) {
      break;
    }
    [[unroll]] for (int tile_n4 = 0; tile_n4 < TILE_N4; ++tile_n4) {
      if (n4 + tile_n4 >= N4) {
        break;
      }
      ivec4 packed_block;
      [[unroll]] for (int i = 0; i < 4; ++i) {
        const int tile_m = tile_m4 * 4 + i;
        if (m + tile_m < M) {
          packed_block[i] = quantize_and_pack(
              out_tile.data[tile_m][tile_n4], output_inv_scale, output_zp);
        } else {
          packed_block[i] = 0;
        }
      }
      t_packed_int8_output[(m4 + tile_m4) * N4 + n4 + tile_n4] = packed_block;
    }
  }
}
