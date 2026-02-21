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

$if WEIGHT_STORAGE == "buffer":
  #define WEIGHT_BUFFER

#define TILE_K4 ${TILE_K4}
#define TILE_N4 ${TILE_N4}

#define TILE_M4 1
#define TILE_M 1
#define TILE_K ${TILE_K4 * 4}
#define TILE_N ${TILE_N4 * 4}

#define WGS ${WGS}

layout(std430) buffer;

// Scalar int arrays for 4W packed int8 input/output
${layout_declare_tensor(B, "w", "t_packed_int8_output", "int", "buffer")}
${layout_declare_tensor(B, "r", "t_packed_int8_input", "int", "buffer")}
// Weight uses ivec4 (same format as q8ta_linear)
${layout_declare_tensor(B, "r", "t_packed_int8_weight", "int", WEIGHT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_sums", "int", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_scales", DTYPE, "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer", is_scalar_array=False)}

${layout_declare_spec_const(C, "int", "apply_bias", "0")}
${layout_declare_spec_const(C, "int", "activation_type", "0")}

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
#include "linear_int8_weight_tile_load.glslh"
#include "linear_fp_output_tile_int8_int8_compute.glslh"
#include "linear_fp_weight_scales_load.glslh"
#include "linear_int_weight_sums_load.glslh"
#include "linear_fp_bias_load.glslh"

shared Int32Accum partial_accums[WGS];

void main() {
  const int lid = int(gl_LocalInvocationID.z);
  const int n4 = int(gl_GlobalInvocationID.x) * TILE_N4;

  const int n = mul_4(n4);

  const int K4 = div_up_4(input_sizes.x);
  const int N4 = div_up_4(output_sizes.x);

  if (n >= output_sizes.x) {
    return;
  }

  Int32Accum out_accum;
  initialize(out_accum);

  Int8WeightTile int8_weight_tile;

  for (int k4 = lid; k4 < K4; k4 += WGS) {
    // Load one packed int32 from the 4W input buffer. Each int32 contains
    // 4 int8 values at k=k4*4..k4*4+3.
    const int packed_input = t_packed_int8_input[k4];

    load_int8_weight_tile(int8_weight_tile, n4, k4, N4);

    // Accumulate dot products of the input int8x4 with each weight int8x4
    [[unroll]] for (int n = 0; n < TILE_N; ++n) {
      const int tile_n4 = div_4(n);
      const int n4i = mod_4(n);
      out_accum.data[0][tile_n4][n4i] = dotPacked4x8AccSatEXT(
          packed_input,
          int8_weight_tile.data[0][tile_n4][n4i],
          out_accum.data[0][tile_n4][n4i]);
    }
  }

  partial_accums[lid] = out_accum;

  memoryBarrierShared();
  barrier();

  // Only the first thread writes the result
  if (lid == 0) {
    for (int i = 1; i < WGS; ++i) {
      [[unroll]] for (int tile_n4 = 0; tile_n4 < TILE_N4; ++tile_n4) {
        out_accum.data[0][tile_n4] +=
            partial_accums[i].data[0][tile_n4];
      }
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
    } else {
      accumulate_out_tile_with_int_accum(
          out_tile,
          out_accum,
          input_scale,
          input_zp,
          weight_sums_tile,
          weight_scales_tile);
    }

    // Apply ReLU if enabled
    if (activation_type > 0) {
      [[unroll]] for (int tile_n4 = 0; tile_n4 < TILE_N4; ++tile_n4) {
        out_tile.data[0][tile_n4] = max(out_tile.data[0][tile_n4], vec4(0.0));
      }
    }

    // Quantize and write to scalar int[] buffer. Each int32 at position n4
    // contains 4 packed int8 output values for channels n4*4..n4*4+3.
    [[unroll]] for (int tile_n4 = 0; tile_n4 < TILE_N4; ++tile_n4) {
      if (n4 + tile_n4 < N4) {
        t_packed_int8_output[n4 + tile_n4] = quantize_and_pack(
            out_tile.data[0][tile_n4], output_inv_scale, output_zp);
      }
    }
  }
}
