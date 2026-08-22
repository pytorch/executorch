/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions("buffer", DTYPE)}

#define PRECISION ${PRECISION}
#define VEC4_T ${texel_load_type(DTYPE, "buffer")}
#define T ${texel_load_component_type(DTYPE, "buffer")}

$if IO_STORAGE == "buffer":
  #define PACKED_INT8_OUTPUT_BUFFER
  #define PACKED_INT8_INPUT_BUFFER
$if WEIGHT_STORAGE == "buffer":
  #define WEIGHT_BUFFER

// corresponds to input/output width dim
#define TILE_M4 1
// corresponds to input channels dim
#define TILE_K4 1
// corresponds to output channels dim
#define TILE_N4 2

#define TILE_M 4
#define TILE_K 4
#define TILE_N 8

layout(std430) buffer;

#include "conv2d_common.glslh"

${layout_declare_tensor(B, "w", "t_packed_int8_output", "int", IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_packed_int8_input", "int", IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_packed_int8_weight", "int", WEIGHT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_sums", "int", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_scales", DTYPE, "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer", is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "output_sizes")}
${layout_declare_ubo(B, "ivec4", "im2col_sizes")}

layout(push_constant) uniform restrict Block {
  float input_scale;
  int input_zp;
  float output_inv_scale;
  int output_zp;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "apply_bias", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_stride_x", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_stride_y", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_padding_x", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_padding_y", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_dilation_x", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_dilation_y", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_kernel_size_x", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_kernel_size_y", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_in_channels_per_group", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_out_channels_per_group", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_K4_per_group", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_K4", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_K_per_group", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_logical_K", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_logical_K_per_group", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_groups", "1")}


#include "conv2d_int8_input_tile_load.glslh"
#include "linear_int8_weight_tile_load.glslh"
#include "linear_fp_output_tile_int8_int8_compute.glslh"
#include "linear_int_weight_sums_load.glslh"
#include "linear_fp_weight_scales_load.glslh"
#include "linear_fp_bias_load.glslh"
#include "linear_int8_output_tile_compute.glslh"
#include "conv2d_int8_output_tile_store.glslh"

void main() {
  Conv2dBlockIndex output_block_idx;
  output_block_idx.data.z = int(gl_GlobalInvocationID.x) * TILE_N4;
  output_block_idx.data.x = int(gl_GlobalInvocationID.y) * TILE_M4;
  output_block_idx.data.y = int(gl_GlobalInvocationID.z);

  Conv2dBlockExtents output_block_extents = make_block_extents(output_sizes);
  if (block_idx_out_of_bounds(output_block_idx, output_block_extents)) {
    return;
  }

  const int n = mul_4(output_block_idx.data.z);

  const int group_idx = n / conv2d_params_out_channels_per_group;
  const int group_k4_offset = group_idx * conv2d_params_K4_per_group;

  Conv2dBlockExtents input_block_extents = make_block_extents(im2col_sizes);

  Int32Accum out_accum;
  initialize(out_accum);

  Int8InputTile int8_input_tile;
  Int8WeightTile int8_weight_tile;

  Int8InputTileIndex input_idx = make_initial_int8_input_tile_index(
      output_block_idx, input_block_extents, group_k4_offset);

  for (int k4 = 0; k4 < conv2d_params_K4_per_group; k4++) {
    load_packed_int8_input_tile(int8_input_tile, input_idx);

    load_int8_weight_tile(
        int8_weight_tile,
        output_block_idx.data.z,
        k4,
        output_block_extents.data.z);

    int_accumulate_with_int8_weight(
        out_accum, int8_input_tile, int8_weight_tile);

    increment_k4(input_idx);
  }

  FPPerOutChannelParams weight_scales_tile;
  load_weight_scales_tile(weight_scales_tile, output_block_idx.data.z);

  IntPerOutChannelParams weight_sums_tile;
  load_weight_sums_tile(weight_sums_tile, output_block_idx.data.z);

  Int8OutTile int8_out_tile;
  initialize(int8_out_tile);

  if (apply_bias > 0) {
    FPPerOutChannelParams bias_tile;
    load_bias_tile(bias_tile, output_block_idx.data.z);

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
      int8_out_tile, output_block_idx, output_block_extents);
}
