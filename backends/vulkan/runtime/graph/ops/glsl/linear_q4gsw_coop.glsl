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

#define TILE_M 1
#define TILE_K4 1
#define TILE_N8 1
#define TILE_N4 2
#define TILE_K 4
#define TILE_N 8

// Inputs are widened and partial sums are kept in fp32 for every IO dtype.
#define LINEAR_FP_INPUT_TILE_VEC4_T vec4
#define LINEAR_FP_OUTPUT_TILE_VEC4_T vec4

#define MAX_WG_SIZE ${MAX_WG_SIZE}

layout(std430) buffer;

$if DYNAMIC_QUANT_VARIANT:
  ${layout_declare_tensor(B, "w", "t_output", DTYPE, IO_STORAGE, is_scalar_array=False)}
  $if NUM_OUTPUTS == 3:
    ${layout_declare_tensor(B, "w", "t_output1", DTYPE, IO_STORAGE, is_scalar_array=False)}
    ${layout_declare_tensor(B, "w", "t_output2", DTYPE, IO_STORAGE, is_scalar_array=False)}
  ${layout_declare_tensor(B, "r", "t_input", DTYPE, IO_STORAGE, is_scalar_array=False)}
  ${layout_declare_tensor(B, "r", "t_packed_int8_input", "int", PACKED_INPUT_STORAGE, is_scalar_array=False)}
  ${layout_declare_tensor(B, "r", "t_int_input_sums", "int", "buffer", is_scalar_array=False)}
  ${layout_declare_tensor(B, "r", "t_input_scale", DTYPE, "texture3d")}
  ${layout_declare_tensor(B, "r", "t_input_zp", "int8" if ZP_DTYPE_MODE == "zpint8" else DTYPE, "texture3d")}
  ${layout_declare_tensor(B, "r", "t_packed_int4_weight", "int", WEIGHT_STORAGE, is_scalar_array=False)}
  ${layout_declare_tensor(B, "r", "t_weight_sums", "int", "buffer", is_scalar_array=False)}
  ${layout_declare_tensor(B, "r", "t_weight_scales", DTYPE, "buffer", is_scalar_array=False)}
  ${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer", is_scalar_array=False)}
$else:
  ${layout_declare_tensor(B, "w", "t_output", DTYPE, IO_STORAGE, is_scalar_array=False)}
  ${layout_declare_tensor(B, "r", "t_input", DTYPE, IO_STORAGE, is_scalar_array=False)}
  ${layout_declare_tensor(B, "r", "t_packed_int4_weight", "int", WEIGHT_STORAGE, is_scalar_array=False)}
  ${layout_declare_tensor(B, "r", "t_weight_scales", DTYPE, "buffer", is_scalar_array=False)}
  ${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer", is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "output_sizes")}
${layout_declare_ubo(B, "ivec4", "input_sizes")}
$if NUM_OUTPUTS == 3:
  ${layout_declare_ubo(B, "ivec4", "output1_sizes")}
  ${layout_declare_ubo(B, "ivec4", "output2_sizes")}
  layout(push_constant) uniform restrict Block {
    ivec4 split_sizes;
  };

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "apply_bias", "0")}
${layout_declare_spec_const(C, "int", "K4_per_group", "0")}
${layout_declare_spec_const(C, "int", "num_groups_arg", "0")}
${layout_declare_spec_const(C, "int", "out_N_arg", "0")}
${layout_declare_spec_const(C, "int", "use_fp16_unpack", "0")}

#include "common.glslh"
#include "linear_fp_input_tile_load.glslh"
#include "linear_int4_weight_tile_load.glslh"
#include "linear_fp_weight_scales_load.glslh"
#include "linear_fp_output_tile_fp_compute.glslh"
#include "linear_fp_output_tile_store.glslh"
$if NUM_OUTPUTS == 3:
  #include "linear_fp_output_tile_split_store.glslh"
#include "linear_fp_bias_load.glslh"

shared FPOutTile partial_sums[MAX_WG_SIZE];

/*
 * Each packed block holds rows n8 * 8 + (0..7) at k4 * 4 + (0..3). Within
 * block[r], byte j holds (row r, k j) in its low nibble and (row r + 4, k j) in
 * its high nibble.
 *
 * Both functions return sum_k x[k] * (q - 8) over one quantization group, for
 * rows 0-3 in lo and rows 4-7 in hi.
 */

void accumulate_group_fp32(
    out vec4 lo,
    out vec4 hi,
    const int group,
    const int n8,
    const int K4,
    const int N8) {
  lo = vec4(0);
  hi = vec4(0);
  float x_sum = 0.0;
  [[unroll]] for (int i = 0; i < K4_per_group; ++i) {
    const int k4 = group * K4_per_group + i;
    const vec4 x = load_input_x4(k4, 0, K4);
    const ivec4 block = load_int4_weight_block(k4, n8, N8);
    x_sum += (x.x + x.y) + (x.z + x.w);
    [[unroll]] for (int j = 0; j < 4; ++j) {
      lo = fma(vec4(x[j]), vec4((block >> (8 * j)) & 0xF), lo);
      hi = fma(vec4(x[j]), vec4((block >> (8 * j + 4)) & 0xF), hi);
    }
  }
  lo -= 8.0 * x_sum;
  hi -= 8.0 * x_sum;
}

$if DTYPE == "half":
  // Mali has far more fp16 than int-to-float conversion throughput. Splicing a
  // nibble q into the mantissa of fp16 1024.0 (0x6400) yields 1024 + q exactly,
  // so two weights are dequantized with one mask, one or and one f16vec2 sub.
  void accumulate_group_fp16(
      out vec4 lo,
      out vec4 hi,
      const int group,
      const int n8,
      const int K4,
      const int N8) {
    f16vec2 acc[8];
    [[unroll]] for (int r = 0; r < 8; ++r) {
      acc[r] = f16vec2(0);
    }
    [[unroll]] for (int i = 0; i < K4_per_group; ++i) {
      const int k4 = group * K4_per_group + i;
      const f16vec4 x = f16vec4(load_input_x4(k4, 0, K4));
      const uvec4 block = uvec4(load_int4_weight_block(k4, n8, N8));
      [[unroll]] for (int r = 0; r < 4; ++r) {
        // (k 0, k 2) and (k 1, k 3) pairs of rows r and r + 4.
        const f16vec2 w_r_02 =
            unpackFloat2x16((block[r] & 0x000F000Fu) | 0x64006400u) -
            f16vec2(1032.0);
        const f16vec2 w_r4_02 =
            unpackFloat2x16(((block[r] >> 4) & 0x000F000Fu) | 0x64006400u) -
            f16vec2(1032.0);
        const f16vec2 w_r_13 =
            unpackFloat2x16(((block[r] >> 8) & 0x000F000Fu) | 0x64006400u) -
            f16vec2(1032.0);
        const f16vec2 w_r4_13 =
            unpackFloat2x16(((block[r] >> 12) & 0x000F000Fu) | 0x64006400u) -
            f16vec2(1032.0);
        acc[r] = fma(w_r_02, x.xz, fma(w_r_13, x.yw, acc[r]));
        acc[r + 4] = fma(w_r4_02, x.xz, fma(w_r4_13, x.yw, acc[r + 4]));
      }
    }
    lo = vec4(
        acc[0].x + acc[0].y,
        acc[1].x + acc[1].y,
        acc[2].x + acc[2].y,
        acc[3].x + acc[3].y);
    hi = vec4(
        acc[4].x + acc[4].y,
        acc[5].x + acc[5].y,
        acc[6].x + acc[6].y,
        acc[7].x + acc[7].y);
  }

/*
 * M = 1 GEMV. Thread (x, y) of a workgroup of size (R, S) computes the 8 output
 * channels of block column n8 = workgroup_x * R + x, over quantization groups
 * y, y + S, y + 2S, ...; the S partial sums are then reduced in shared memory.
 * With [k4][n8]-ordered weight blocks, adjacent threads read adjacent blocks.
 */
void main() {
  const int R = int(gl_WorkGroupSize.x);
  const int S = int(gl_WorkGroupSize.y);
  const int x = int(gl_LocalInvocationID.x);
  const int y = int(gl_LocalInvocationID.y);

  const int n8 = int(gl_WorkGroupID.x) * R + x;
  const int n4 = mul_2(n8);
  const int num_groups = num_groups_arg;
  const int K4 = num_groups * K4_per_group;
  const int N4 = div_up_4(out_N_arg);
  const int N8 = div_up_8(out_N_arg);

  FPOutTile out_tile;
  initialize(out_tile);

  if (n8 < N8) {
    const bool has_hi = n4 + 1 < N4;
    for (int group = y; group < num_groups; group += S) {
      vec4 lo;
      vec4 hi;
      $if DTYPE == "half":
        if (use_fp16_unpack != 0) {
          accumulate_group_fp16(lo, hi, group, n8, K4, N8);
        } else {
          accumulate_group_fp32(lo, hi, group, n8, K4, N8);
        }
      $else:
        accumulate_group_fp32(lo, hi, group, n8, K4, N8);
      const vec4 scales_lo = vec4(load_scale_x4(n4, group, N4));
      const vec4 scales_hi =
          has_hi ? vec4(load_scale_x4(n4 + 1, group, N4)) : vec4(0);
      out_tile.data[0][0] = fma(lo, scales_lo, out_tile.data[0][0]);
      out_tile.data[0][1] = fma(hi, scales_hi, out_tile.data[0][1]);
    }
  }

  if (S > 1) {
    partial_sums[y * R + x] = out_tile;
    memoryBarrierShared();
    barrier();
    if (y > 0) {
      return;
    }
    for (int i = 1; i < S; ++i) {
      accumulate_out_tile_with_out_tile(out_tile, partial_sums[i * R + x]);
    }
  }

  if (n8 < N8) {
    if (apply_bias > 0) {
      FPPerOutChannelParams bias_tile;
      load_bias_tile(bias_tile, n4);
      add_bias_to_out_tile(out_tile, bias_tile);
    }
    $if NUM_OUTPUTS == 3:
      write_output_tile_split_with_checks(out_tile, n4, 0, N4, 1);
    $else:
      write_output_tile_with_checks(out_tile, n4, 0, N4, 1);
  }
}
