/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define NAME ${VARIANT_NAME}

#define VEC4_T ${texel_load_type(DTYPE, "buffer")}
#define T ${texel_load_component_type(DTYPE, "buffer")}

$if IO_STORAGE == "buffer":
  #define PACKED_INT8_OUTPUT_BUFFER
  #define PACKED_INT8_INPUT_BUFFER

#define op(X, Y) ${OPERATOR}

${define_required_extensions(DTYPE)}

layout(std430) buffer;

#extension GL_EXT_debug_printf : enable
#define DEBUG_MODE
#include "indexing.glslh"
#include "common.glslh"

${layout_declare_tensor(B, "w", "t_packed_int8_out", "int", IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_packed_int8_in_a", "int", IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_packed_int8_in_b", "int", IO_STORAGE, is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "out_sizes")}

layout(push_constant) uniform restrict Block {
  float input_a_scale;
  int input_a_zp;
  float input_b_scale;
  int input_b_zp;
  float output_inv_scale;
  int output_zp;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const int tid = int(gl_GlobalInvocationID.x);

  const int W4 = div_up_4(out_sizes.x);
  const int H = out_sizes.y;
  const int C4 = div_up_4(out_sizes.z);
  const int N = out_sizes.w;

  if (tid >= W4 * H * C4 * N) {
    return;
  }

  const ivec4 in_block_1 = t_packed_int8_in_a[tid];
  const ivec4 in_block_2 = t_packed_int8_in_b[tid];

  ivec4 out_block = ivec4(pack_into_int32(ivec4(output_zp)));

  for (int row = 0; row < 4; row++) {
    vec4 in_texel_1 = unpack_and_dequantize(
        in_block_1[row], input_a_scale, input_a_zp);
    vec4 in_texel_2 = unpack_and_dequantize(
        in_block_2[row], input_b_scale, input_b_zp);

    vec4 out_texel = op(in_texel_1, in_texel_2);
    out_block[row] = quantize_and_pack(out_texel, output_inv_scale, output_zp);
  }

  t_packed_int8_out[tid] = out_block;
}
