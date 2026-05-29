/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

// W8A8 dynamic: int8 dynamic-per-token activations × int8 per-channel
// symmetric weights. Direct sibling of linear_dq8ca_q4gsw_tiled, but with
// the int4 nibble-unpack stage replaced by a direct int8 weight load and
// the per-group loop collapsed into a single K loop (per-channel weights
// have no groups).

// For input/output tensors
${define_required_extensions(IO_STORAGE, DTYPE)}
// For int8 input scales/zps
${define_required_extensions("texture3d", "int8")}
// For weight scales and bias
${define_required_extensions("buffer", DTYPE)}

#define PRECISION ${PRECISION}
#define VEC4_T ${texel_load_type(DTYPE, IO_STORAGE)}
#define T ${texel_load_component_type(DTYPE, IO_STORAGE)}

$if IO_STORAGE == "buffer":
  #define OUTPUT_BUFFER
  #define INPUT_BUFFER
$if PACKED_INT8_INPUT_STORAGE == "buffer":
  #define PACKED_INT8_INPUT_BUFFER
$if WEIGHT_STORAGE == "buffer":
  #define WEIGHT_BUFFER

#define TILE_N8 ${TILE_N8}

#define TILE_M4 ${TILE_M4}
#define TILE_K4 ${TILE_K4}
#define TILE_N4 ${TILE_N8 * 2}

#define TILE_M ${TILE_M4 * 4}
#define TILE_K ${TILE_K4 * 4}
#define TILE_N ${TILE_N8 * 8}

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_output", DTYPE, IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_input", DTYPE, IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_packed_int8_input", "int", PACKED_INT8_INPUT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_int8_input_sums", "int", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_int8_input_scales", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_int8_input_zps", "int8", "texture3d")}
${layout_declare_tensor(B, "r", "t_packed_int8_weight", "int", WEIGHT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_sums", "int", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_scales", DTYPE, "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer", is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "output_sizes")}
${layout_declare_ubo(B, "ivec4", "input_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "apply_bias", "0")}

#include "linear_fp_input_tile_load.glslh"
#include "linear_int8_input_tile_load.glslh"
#include "linear_int8_input_scales_zps_load.glslh"
#include "linear_int8_weight_tile_load.glslh"
#include "linear_int_weight_sums_load.glslh"
#include "linear_fp_weight_scales_load.glslh"
#include "linear_fp_output_tile_int8_int8_compute.glslh"
#include "linear_fp_output_tile_fp_compute.glslh"
#include "linear_fp_output_tile_store.glslh"
#include "linear_fp_bias_load.glslh"

void main() {
  const int out_tile_x = int(gl_GlobalInvocationID.x);
  const int out_tile_y = int(gl_GlobalInvocationID.y);

  const int n = out_tile_x * TILE_N;
  const int m = out_tile_y * TILE_M;

  const int n8 = div_8(n);
  const int n4 = div_4(n);
  const int m4 = div_4(m);

  if (n >= output_sizes.x || m >= output_sizes.y) {
    return;
  }

  const int M = input_sizes.y;
  const int K4 = div_up_4(input_sizes.x);
  const int M4 = div_up_4(M);
  const int N4 = div_up_4(output_sizes.x);
  const int N8 = div_up_8(output_sizes.x);

  FPOutTile out_tile;
  initialize(out_tile);

  Int32Accum out_accum;
  initialize(out_accum);

  Int8InputTile int8_in_tile;
  Int8WeightTile int8_weight_tile;

  Int8InputScales input_scales;
  Int8InputZeroPoints input_zps;
  load_int8_input_scales_and_zps(input_scales, input_zps, m4);

  FPPerOutChannelParams weight_scales_tile;
  IntPerOutChannelParams weight_sums_tile;

  // Per-channel symmetric: single K loop, no per-group reset of accumulator.
  for (int k4 = 0; k4 < K4; ++k4) {
    load_int8_input_tile(int8_in_tile, k4, m4, K4);
    load_int8_weight_tile(int8_weight_tile, n4, k4, N4);

    int_accumulate_with_int8_weight(
        out_accum, int8_in_tile, int8_weight_tile);
  }

  load_weight_scales_tile(weight_scales_tile, n4);
  load_weight_sums_tile(weight_sums_tile, n4);

  // Per-row dequant: dq8ca uses per-row (per-token) activation quant, so each
  // output row gets its own (input_scale, input_zp). The scales/zps for this
  // tile's TILE_M rows were loaded into the tile-local arrays starting at
  // index 0, so index them tile-locally by m_row (not by absolute row m+m_row,
  // which would run off the end of the TILE_M4-sized arrays for m >= TILE_M).
  [[unroll]] for (int m_row = 0; m_row < TILE_M; ++m_row) {
    const int row_m4 = div_4(m_row);
    const int row_m4i = mod_4(m_row);
    float row_scale = float(input_scales.data[row_m4][row_m4i]);
    int row_zp = int(input_zps.data[row_m4][row_m4i]);

    // Apply per-row scale/zp to this row of the accumulator into out_tile.
    ivec4 input_zp_vec = ivec4(-row_zp);
    [[unroll]] for (int n4_inner = 0; n4_inner < TILE_N4; ++n4_inner) {
      ivec4 accum_adjusted =
          input_zp_vec * weight_sums_tile.data[n4_inner] +
          out_accum.data[m_row][n4_inner];
      out_tile.data[m_row][n4_inner] =
          fma(VEC4_T(accum_adjusted),
              VEC4_T(row_scale * weight_scales_tile.data[n4_inner]),
              out_tile.data[m_row][n4_inner]);
    }
  }

  if (apply_bias > 0) {
    FPPerOutChannelParams bias_tile;
    load_bias_tile(bias_tile, n4);
    add_bias_to_out_tile(out_tile, bias_tile);
  }

  write_output_tile_with_checks(out_tile, n4, m, N4, M);
}
