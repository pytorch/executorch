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
${define_required_extensions("int8")}

#extension GL_EXT_integer_dot_product : require

#define NUM_GROUPS_PER_WG ${NUM_GROUPS_PER_WG}
#define NUM_WORKERS_PER_GROUP ${NUM_WORKERS_PER_GROUP}

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_packed_int8_input", "int", OUTPUT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "w", "t_int8_input_sums", "int", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_input", DTYPE, INPUT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_int8_input_scales", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_int8_input_zps", "int8", "texture3d")}

${layout_declare_ubo(B, "ivec4", "input_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "K4_per_group", "0")}

shared ivec4 shared_sums[NUM_GROUPS_PER_WG][NUM_WORKERS_PER_GROUP];

#define TILE_M4 1
#define TILE_K4 1

#define TILE_M 4

#include "linear_int8_input_block.glslh"
#include "linear_int8_input_scales_zps_load.glslh"
#include "linear_fp_input_tile_load.glslh"

void main() {
  const int group_idx = int(gl_GlobalInvocationID.x);
  const int m4 = int(gl_GlobalInvocationID.y);

  const int worker_id = int(gl_LocalInvocationID.z);
  const int group_offset = int(gl_LocalInvocationID.x);

  const int K = input_sizes.x;
  const int M = input_sizes.y;

  // K4 and M4 represent the number of blocks in each dimension.
  const int K4 = div_up_4(K);
  const int M4 = div_up_4(M);

  const int num_groups = K4 / K4_per_group;;

  if (group_idx >= num_groups || m4 >= M4) {
    return;
  }

  const int start_k4 = group_idx * K4_per_group + worker_id;
  const int end_k4 = (group_idx + 1) * K4_per_group;

  Int8InputScales input_scales;
  Int8InputZeroPoints input_zps;
  load_int8_input_scales_and_zps(input_scales, input_zps, m4);

  // row of the input tensor to start loading from
  const int m = mul_4(m4);

  FPInputTile in_tile;
  Int8InputBlock packed_block;

  ivec4 local_sum = ivec4(0, 0, 0, 0);
  const int packed_ones = 0x01010101;

  for (int k4 = start_k4; k4 < end_k4; k4 += NUM_WORKERS_PER_GROUP) {
    load_input_tile_no_checks(in_tile, k4, m, K4, M);
    quantize_and_pack(packed_block, in_tile, input_scales, input_zps);

    // Sum the quantized values in the block
    [[unroll]] for (int m = 0; m < TILE_M; m++) {
      local_sum[m] += dotPacked4x8AccSatEXT(
          packed_block.data[m], packed_ones, local_sum[m]);
    }
    write_block(packed_block, k4, m4, K4);
  }

  shared_sums[group_offset][worker_id] = local_sum;

  memoryBarrierShared();
  barrier();

  // Tree reduction to compute the overall result
  for (int i = NUM_WORKERS_PER_GROUP / 2; i > 0; i >>= 1) {
    if (worker_id < i) {
      shared_sums[group_offset][worker_id] =
          shared_sums[group_offset][worker_id] +
          shared_sums[group_offset][worker_id + i];
    }
    memoryBarrierShared();
    barrier();
  }

  if (worker_id == 0) {
    t_int8_input_sums[group_idx * M4 + m4] = shared_sums[group_offset][0];
  }
}
