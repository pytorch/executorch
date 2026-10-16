/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Row-major (utils::kPackedInt8_4W) counterpart of
// quantize_and_pack_4h4w_with_group_sums.glsl.
//
// Identical quantization and per-group sum reduction; only the output layout
// differs. The 4h4w version writes an ivec4 per (m4, k4) block whose component
// selects one of 4 rows -- as a scalar array that is
// index = m4*(4*K4) + k4*4 + r, which is not affine in the row index and so
// cannot be addressed by a cooperative-matrix load. This version writes each
// of the 4 quantized rows to its own row-major slot,
// t_packed_int8_input[m * K4 + k4], giving plain row-major int8 with 4
// K-contiguous values per int32 and a row stride of K4.
//
// Consumed only by linear_dq8ca_q4gsw_wmma_rowmajor. Buffer output only.

#version 450 core

${define_required_extensions(INPUT_STORAGE, DTYPE)}
${define_required_extensions("texture3d", "int8")}

#define PRECISION ${PRECISION}
#define VEC4_T ${texel_load_type(DTYPE, INPUT_STORAGE)}
#define T ${texel_load_component_type(DTYPE, INPUT_STORAGE)}

$if OUTPUT_STORAGE == "buffer":
  #define OUTPUT_BUFFER
$if INPUT_STORAGE == "buffer":
  #define INPUT_BUFFER

#extension GL_EXT_integer_dot_product : require

#define NUM_GROUPS_PER_WG ${NUM_GROUPS_PER_WG}
#define NUM_WORKERS_PER_GROUP ${NUM_WORKERS_PER_GROUP}

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_packed_int8_input", "int", OUTPUT_STORAGE, is_scalar_array=True)}
${layout_declare_tensor(B, "w", "t_int8_input_sums", "int", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_input", DTYPE, INPUT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_int8_input_scales", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_int8_input_zps", "int8" if ZP_DTYPE_MODE == "zpint8" else DTYPE, "texture3d")}

${layout_declare_ubo(B, "ivec4", "input_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "K4_per_group", "0")}

shared ivec4 shared_sums[NUM_GROUPS_PER_WG][NUM_WORKERS_PER_GROUP];

#define TILE_M4 1
#define TILE_K4 1

#define TILE_M 4

// The shared helper's write_block() assumes an ivec4-typed output resource;
// this shader's output is a scalar int array, so opt out and write our own.
#define SKIP_BLOCK_WRITE_HELPERS
#include "gemm/tile_utils/linear_int8_input_block.glslh"

// Scatters the 4 rows of a quantized block into row-major slots. Unlike the
// 4h4w layout, kPackedInt8_4W has NO row padding (outer_dim_align == 1), so
// the tail rows of a workgroup whose m4 block runs past M must be dropped --
// hence the explicit bound check that write_block() does not need.
void write_block_rowmajor(
    const Int8InputBlock block,
    const int k4,
    const int m,
    const int K4,
    const int M) {
  for (int row = 0; row < 4; ++row) {
    const int m_row = m + row;
    if (m_row < M) {
      t_packed_int8_input[m_row * K4 + k4] = block.data[row];
    }
  }
}
#include "gemm/tile_utils/linear_int8_input_scales_zps_load.glslh"
#include "gemm/tile_utils/linear_fp_input_tile_load.glslh"

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

  const int num_groups = K4 / K4_per_group;

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
    write_block_rowmajor(packed_block, k4, m, K4, M);
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
