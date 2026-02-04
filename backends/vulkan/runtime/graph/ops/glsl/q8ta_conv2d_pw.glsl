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
#define T ${texel_load_component_type(DTYPE, "buffer")}

${define_active_storage_type("buffer")}

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

#include "indexing.glslh"
#include "common.glslh"
#include "block_indexing.glslh"

${layout_declare_tensor(B, "w", "t_packed_int8_output", "int", "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_packed_int8_input", "int", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_packed_int8_weight", "int", "texture2d", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_sums", "int", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_scales", DTYPE, "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer", is_scalar_array=False)}

// Metadata for input/output tensors
${layout_declare_ubo(B, "BufferMetadata", "outp")}
${layout_declare_ubo(B, "BufferMetadata", "inp")}

layout(push_constant) uniform restrict Block {
  float input_scale;
  int input_zp;
  float output_inv_scale;
  int output_zp;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "apply_bias", "1")}
${layout_declare_spec_const(C, "int", "conv2d_params_K4_per_group", "1")}

// Layout specialization constants
${layout_declare_spec_const(C, "int", "outp_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "inp_layout", "CONTIG_LAYOUT_INT")}

int compute_outp_buffer_idx(
    const int w_block_idx,
    const int h_idx,
    const int c_block_idx) {
  if (get_outer_packed_dim_block_size(outp_layout) == 1) {
    return h_idx * int(outp.strides[0][1])
           + mul_4(w_block_idx) * int(outp.strides[0][0])
           + c_block_idx * int(outp.strides[0][2]);
  } else {
    return mul_4(
      h_idx * int(outp.strides[0][1])
      + w_block_idx * int(outp.strides[0][0])
      + c_block_idx * int(outp.strides[0][2]));
  }

}

void main() {
  // Thread mapping: each thread handles TILE_M (4) widths × TILE_N (8) output channels
  // gl_GlobalInvocationID.x → output channel blocks (TILE_N4 = 2 blocks of 4 channels)
  // gl_GlobalInvocationID.y → width blocks (TILE_M4 = 1 block of 4 widths)
  // gl_GlobalInvocationID.z → batch (or height * batch combined)
  const int oc_block_idx = int(gl_GlobalInvocationID.x) * TILE_N4;
  const int ow_block_idx = int(gl_GlobalInvocationID.y) * TILE_M4;
  const int oh = int(gl_GlobalInvocationID.z);

  // Get output extents in block space (div_up_4 for packed dimensions)
  const int W = int(outp.sizes[0][0]);
  const int W4 = div_up_4(int(outp.sizes[0][0]));
  const int H = int(outp.sizes[0][1]);
  const int OC4 = div_up_4(int(outp.sizes[0][2]));

  // Bounds check in block space
  if (ow_block_idx >= W4 ||
      oh >= H ||
      oc_block_idx >= OC4) {
    return;
  }

  // Get input extents in block space
  const int inp_W4 = div_up_4(int(inp.sizes[0][0]));
  const int inp_IC4 = div_up_4(int(inp.sizes[0][2]));

  // Precompute stride products for indexing
  // For 4W4C layout: buffer_idx = batch * (W4 * C4) + w4 * C4 + c4
  const int inp_w_stride = int(inp.strides[0][0]);
  const int inp_h_stride = int(inp.strides[0][1]);
  const int inp_c_stride = int(inp.strides[0][2]);

  // Initialize int32 accumulator
  ivec4 out_accum[TILE_M][TILE_N4];
  [[unroll]] for (int m = 0; m < TILE_M; ++m) {
    [[unroll]] for (int n4 = 0; n4 < TILE_N4; ++n4) {
      out_accum[m][n4] = ivec4(0);
    }
  }

  // Compute initial input tile index
  // Input has same spatial layout, channel dimension iterates from 0
  int input_idx = oh * inp_h_stride + ow_block_idx * inp_w_stride;

  // Main accumulation loop over K dimension
  for (int k4 = 0; k4 < conv2d_params_K4_per_group; k4++) {
    // Load packed int8 input tile (TILE_M4=1, TILE_K4=1)
    // Each int contains 4 packed int8s (one per width position in the tile)
    ivec4 int8_input_tile = t_packed_int8_input[input_idx];

    // Load int8 weight tile (TILE_K4=1, TILE_N4=2)
    ivec4 int8_weight_tile[TILE_N4];
    [[unroll]] for (int n4 = 0; n4 < TILE_N4; ++n4) {
      int8_weight_tile[n4] = texelFetch(
          t_packed_int8_weight,
          ivec2(oc_block_idx + n4, k4),
          0);
    }

    // Accumulate using int8 dot product
    // Input tile indexed as input[m] where m is the width index within tile
    // Weight tile indexed as weight[n4][n4i] where n4i is the channel index within block
    [[unroll]] for (int m = 0; m < TILE_M; ++m) {
      [[unroll]] for (int n4 = 0; n4 < TILE_N4; ++n4) {
        [[unroll]] for (int n4i = 0; n4i < 4; ++n4i) {
          out_accum[m][n4][n4i] = dotPacked4x8AccSatEXT(
              int8_input_tile[m],
              int8_weight_tile[n4][n4i],
              out_accum[m][n4][n4i]);
        }
      }
    }

    input_idx++;
  }

  // Load weight scales tile
  VEC4_T weight_scales[TILE_N4];
  [[unroll]] for (int n4 = 0; n4 < TILE_N4; ++n4) {
    weight_scales[n4] = t_weight_scales[oc_block_idx + n4];
  }

  // Load weight sums tile
  ivec4 weight_sums[TILE_N4];
  [[unroll]] for (int n4 = 0; n4 < TILE_N4; ++n4) {
    weight_sums[n4] = ivec4(t_weight_sums[oc_block_idx + n4]);
  }

  // Initialize int8 output tile
  ivec4 int8_out_tile[TILE_M4][TILE_N4];
  [[unroll]] for (int m4 = 0; m4 < TILE_M4; ++m4) {
    [[unroll]] for (int n4 = 0; n4 < TILE_N4; ++n4) {
      int8_out_tile[m4][n4] = ivec4(0);
    }
  }

  // Compute int8 output tile from int32 accumulator
  ivec4 input_zp_vec = ivec4(-input_zp);

  if (apply_bias > 0) {
    // Load bias tile
    VEC4_T bias[TILE_N4];
    [[unroll]] for (int n4 = 0; n4 < TILE_N4; ++n4) {
      bias[n4] = t_bias[oc_block_idx + n4];
    }

    [[unroll]] for (int m4 = 0; m4 < TILE_M4; ++m4) {
      [[unroll]] for (int m4i = 0; m4i < 4; ++m4i) {
        [[unroll]] for (int n4 = 0; n4 < TILE_N4; ++n4) {
          const int m = mul_4(m4) + m4i;
          // Compute floating point output values
          ivec4 accum_adjusted =
              input_zp_vec * weight_sums[n4] + out_accum[m][n4];
          vec4 float_out_texel =
              fma(vec4(accum_adjusted),
                  vec4(weight_scales[n4]) * input_scale,
                  vec4(bias[n4]));
          // Requantize to int8
          float_out_texel =
              round(float_out_texel * output_inv_scale) + output_zp;
          ivec4 quantized_out_texel = clamp(ivec4(float_out_texel), -128, 127);

          int8_out_tile[m4][n4][m4i] = pack_into_int32(quantized_out_texel);
        }
      }
    }
  } else {
    [[unroll]] for (int m4 = 0; m4 < TILE_M4; ++m4) {
      [[unroll]] for (int m4i = 0; m4i < 4; ++m4i) {
        [[unroll]] for (int n4 = 0; n4 < TILE_N4; ++n4) {
          const int m = mul_4(m4) + m4i;
          // Compute floating point output values
          ivec4 accum_adjusted =
              input_zp_vec * weight_sums[n4] + out_accum[m][n4];
          vec4 float_out_texel =
              vec4(accum_adjusted) * vec4(weight_scales[n4] * input_scale);
          // Requantize to int8
          float_out_texel =
              round(float_out_texel * output_inv_scale) + output_zp;
          ivec4 quantized_out_texel = clamp(ivec4(float_out_texel), -128, 127);

          int8_out_tile[m4][n4][m4i] = pack_into_int32(quantized_out_texel);
        }
      }
    }
  }

  const int outp_w_stride = int(outp.strides[0][0]);

  // Store packed int8 output tile
  [[unroll]] for (int m4 = 0; m4 < TILE_M4; m4++) {
    [[unroll]] for (int n4 = 0; n4 < TILE_N4; n4++) {
      const int base_outp_buffer_idx = compute_outp_buffer_idx(
          ow_block_idx + m4,
          oh,
          oc_block_idx + n4);
      if (oc_block_idx + n4 < OC4) {
        // Store individual ints from the ivec4
        const int subtile_w_limit = min(4, W - mul_4(ow_block_idx + m4));
        [[unroll]] for (int subtile_w = 0; subtile_w < subtile_w_limit; ++subtile_w) {
          if (get_outer_packed_dim_block_size(outp_layout) == 1) {
            t_packed_int8_output[base_outp_buffer_idx + subtile_w * outp_w_stride] = int8_out_tile[m4][n4][subtile_w];
          } else {
            t_packed_int8_output[base_outp_buffer_idx + subtile_w] = int8_out_tile[m4][n4][subtile_w];
          }
        }
      }
    }
  }
}
