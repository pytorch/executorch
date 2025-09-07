/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}
#define VEC4_T ${texel_load_type(DTYPE, IO_STORAGE)}
#define T ${texel_load_component_type(DTYPE, IO_STORAGE)}

$if IO_STORAGE == "buffer":
  #define OUTPUT_BUFFER
  #define INPUT_BUFFER
$if WEIGHT_STORAGE == "buffer":
  #define WEIGHT_BUFFER

#define TILE_N8 ${TILE_N8}

#define TILE_K4 ${TILE_K4}
#define TILE_N4 ${TILE_N8 * 2}

#define TILE_M ${TILE_M}
#define TILE_K ${TILE_K4 * 4}
#define TILE_N ${TILE_N8 * 8}

#define WGS ${WGS}

${define_required_extensions(DTYPE)}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_output", DTYPE, IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_input", DTYPE, IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_packed_int4_weight", "int", WEIGHT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_scales", DTYPE, "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer", is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "output_sizes")}
${layout_declare_ubo(B, "ivec4", "input_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "apply_bias", "0")}
${layout_declare_spec_const(C, "int", "K4_per_group", "0")}

#include "common.glslh"
#include "linear_fp_input_tile_load.glslh"
#include "linear_int4_weight_tile_load.glslh"
#include "linear_fp_weight_scales_load.glslh"
#include "linear_fp_output_tile_fp_int4_compute.glslh"
#include "linear_fp_output_tile_fp_compute.glslh"
#include "linear_fp_output_tile_store.glslh"
#include "linear_fp_bias_load.glslh"

shared FPOutTile partial_sums[WGS];

void main() {
  const int lid = int(gl_LocalInvocationID.x);
  const int n8 = int(gl_GlobalInvocationID.y);

  // The output tensor will have a shape of [n, 1, 1, 1]. Each thread computes
  // 8 output elements, so each thread will write to 8 elements starting at the
  // tensor index (gid.x * 8, 0, 0, 0).
  const int n = mul_8(n8);
  const int n4 = mul_2(n8);
  const int K4 = div_up_4(input_sizes.x);
  const int N4 = div_up_4(output_sizes.x);

  const int group_size = mul_4(K4_per_group);

  if (n >= output_sizes.x) {
    return;
  }

  FPOutTile out_tile;
  initialize(out_tile);

  FPInputTile in_tile;
  Int4WeightTile int4_weight_tile;

  FPPerOutChannelParams weight_scales_tile;
  FPPerOutChannelParams weight_zeros_tile;
  weight_zeros_tile.data[0] = VEC4_T(0.0);
  weight_zeros_tile.data[1] = VEC4_T(0.0);

  // initialize the group index to a value larger than the largest possible
  int cur_group_idx = input_sizes.x;

  for (int k4 = lid; k4 < div_up_4(input_sizes.x); k4 += WGS) {
    const int group_idx = k4 / K4_per_group;

    // Only update the scales/zeros if the current iteration is now working on a
    // new quantization group.
    if (group_idx != cur_group_idx) {
      load_weight_scales_tile_for_group(weight_scales_tile, n4, group_idx, N4);
      cur_group_idx = group_idx;
    }

    load_input_tile_no_checks(in_tile, k4, 0, K4, 1);
    load_int4_weight_tile(int4_weight_tile, k4, n8, K4);

    fp_accumulate_with_int4_weight(
        out_tile,
        in_tile,
        int4_weight_tile,
        weight_scales_tile,
        weight_zeros_tile);
  }

  partial_sums[lid] = out_tile;

  memoryBarrierShared();
  barrier();

  // Tree reduction to compute the overall result.
  for (int i = WGS / 2; i > 0; i /= 2) {
    if (lid < i) {
      accumulate_out_tile_with_out_tile(
          partial_sums[lid], partial_sums[lid + i]);
    }
    memoryBarrierShared();
    barrier();
  }

  // Only the first thread will write out result
  if (lid == 0) {
    out_tile = partial_sums[0];
    write_output_tile_with_checks(out_tile, n4, 0, N4, 1);
  }
}
