/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}
#define VEC4_T ${texel_load_type(DTYPE, "buffer")}
#define T ${texel_load_component_type(DTYPE, "buffer")}

$if IO_STORAGE == "buffer":
  #define PACKED_INT8_OUTPUT_BUFFER
  #define PACKED_INT8_INPUT_BUFFER
$if WEIGHT_STORAGE == "buffer":
  #define WEIGHT_BUFFER

#define MAX_WINDOW_WIDTH 16

// corresponds to input/output width dim
#define TILE_M4 1
// corresponds to input channels dim
#define TILE_K4 1
// corresponds to output channels dim
#define TILE_N4 1

#define TILE_M 4
#define TILE_K 4
#define TILE_N 4

${define_required_extensions(DTYPE)}

layout(std430) buffer;

#include "conv2d_common.glslh"

${layout_declare_tensor(B, "w", "t_packed_int8_output", "int", IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_packed_int8_input", "int", IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_packed_int8_weight", "int", WEIGHT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_sums", "int", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_scales", DTYPE, "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer", is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "output_sizes")}
${layout_declare_ubo(B, "ivec4", "input_sizes")}
${layout_declare_ubo(B, "Conv2DParams", "conv2d_params")}

layout(push_constant) uniform restrict Block {
  float input_scale;
  int input_zp;
  float output_inv_scale;
  int output_zp;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "apply_bias", "1")}

#include "im2col_packed_int8_utils.glslh"
#include "conv2d_int8_input_tile_load.glslh"
#include "linear_int8_weight_tile_load.glslh"
#include "linear_fp_output_tile_int8_int8_compute.glslh"
#include "linear_int_weight_sums_load.glslh"
#include "linear_fp_weight_scales_load.glslh"
#include "linear_fp_bias_load.glslh"
#include "linear_int8_output_tile_compute.glslh"
#include "conv2d_int8_output_tile_store.glslh"

#include "conv2d_q8_utils.glslh"

void main() {
  Conv2dBlockIndex out_block_idx;
  out_block_idx.data.z = int(gl_GlobalInvocationID.x) * TILE_N4;
  out_block_idx.data.x = int(gl_GlobalInvocationID.y) * TILE_M4;
  out_block_idx.data.y = int(gl_GlobalInvocationID.z);

  Conv2dBlockExtents out_block_extents = make_block_extents(output_sizes);
  if (block_idx_out_of_bounds(out_block_idx, out_block_extents)) {
    return;
  }

  const int out_x_start = mul_4(out_block_idx.data.x);

  Conv2dBlockExtents in_block_extents = make_block_extents(input_sizes);

  const ivec4 input_zps = ivec4(pack_into_int32(ivec4(input_zp)));
  const vec4 weight_scales = vec4(t_weight_scales[out_block_idx.data.z]);

  Int32Accum out_accum;
  initialize(out_accum);

  const int IC4_per_group = div_up_4(conv2d_params.in_channels_per_group);

  const int out_z = mul_4(out_block_idx.data.z);
  const int group_idx = out_z / conv2d_params.out_channels_per_group;
  const int group_ic4_offset = group_idx * IC4_per_group;

  for (int ky = 0; ky < conv2d_params.kernel_size.y; ky++) {
    const int in_y = out_block_idx.data.y * conv2d_params.stride.y -
        conv2d_params.padding.y + ky * conv2d_params.dilation.y;

    for (int kx = 0; kx < conv2d_params.kernel_size.x; kx++) {
      int in_x_load_start =
          (out_x_start * conv2d_params.stride.x)
          - conv2d_params.padding.x
          + (kx * conv2d_params.dilation.x);

      int in_x_load_end =
          ((out_x_start + 3) * conv2d_params.stride.x)
          - conv2d_params.padding.x
          + (kx * conv2d_params.dilation.x);

      in_x_load_start = align_down_4(in_x_load_start);
      in_x_load_end = align_down_4(in_x_load_end);

      for (int ic4 = 0; ic4 < IC4_per_group; ic4++) {
        const ivec4 weight_block = load_weight_block(
            ic4,
            kx,
            ky,
            out_block_idx.data.z,
            IC4_per_group,
            conv2d_params.kernel_size.x,
            conv2d_params.kernel_size.y,
            out_block_extents.data.z);

        for (int in_x = in_x_load_start; in_x <= in_x_load_end; in_x+=4) {
          const ivec4 in_block = load_input_block(
              div_4(in_x),
              in_y,
              group_ic4_offset + ic4,
              in_block_extents,
              input_zps);

          conv1d_accumulate(
              out_accum,
              in_block,
              weight_block,
              kx,
              out_x_start,
              in_x);
        }
      }
    }
  }

  FPPerOutChannelParams weight_scales_tile;
  load_weight_scales_tile(weight_scales_tile, out_block_idx.data.z);

  IntPerOutChannelParams weight_sums_tile;
  load_weight_sums_tile(weight_sums_tile, out_block_idx.data.z);

  Int8OutTile int8_out_tile;
  initialize(int8_out_tile);

  if (apply_bias > 0) {
    FPPerOutChannelParams bias_tile;
    load_bias_tile(bias_tile, out_block_idx.data.z);

    compute_int8_out_tile_with_int32_accum(
        int8_out_tile,
        out_accum,
        input_scale,
        input_zp,
        output_inv_scale,
        output_zp,
        weight_sums_tile,
        weight_scales_tile,
        bias_tile);
  }
  else {
    compute_int8_out_tile_with_int32_accum(
        int8_out_tile,
        out_accum,
        input_scale,
        input_zp,
        output_inv_scale,
        output_zp,
        weight_sums_tile,
        weight_scales_tile);
  }

  store_packed_int8_output_tile(
      int8_out_tile, out_block_idx, out_block_extents);
}
