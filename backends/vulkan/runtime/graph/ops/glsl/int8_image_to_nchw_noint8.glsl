/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#include "indexing_utils.h"

layout(std430) buffer;

#extension GL_EXT_control_flow_attributes : require

${layout_declare_buffer(B, "w", "nchw_out", "int")}
${layout_declare_tensor(B, "r", "t_in", "int8", "texture3d")}
${layout_declare_ubo(B, "ivec4", "tensor_sizes")}
${layout_declare_ubo(B, "ivec4", "axis_map")}
${layout_declare_ubo(B, "int", "out_numel")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int packed_dim = C_DIM;

void main() {
  const int out_buf_idx = int(gl_GlobalInvocationID.x);
  // On the CPU, the number of elements is determined based on a buffer of int8
  // elements. However, on the GPU, since the int8 data type is not supported
  // each group of 4 elements is interepreted as 1 int32 element. Thus each
  // thread is actually writing to 4 output elements from the perspective of the
  // CPU.
  if (out_buf_idx * 4 >= out_numel) {
    return;
  }

  ivec4 values;
  int in_buf_idx = 4 * out_buf_idx;

  [[unroll]] for (int i = 0; i < 4; ++i) {
    const ivec4 tensor_idx = from_nchw_buffer_i(in_buf_idx, tensor_sizes);
    const ivec4 texture_pos = to_texture_elem_pos(
        tensor_idx, tensor_sizes, packed_dim);
    values[i] = load_texel(t_in, texture_pos.xyz)[texture_pos.w];
    in_buf_idx++;
  }

  // Manually pack 4x 8-bit integers into a 32 bit integer. Note that little
  // endian is assumed, since most processors use little endian. Thus the
  // "later" values are placed in most significant bytes.
  int packed = ((values[3] & 0xFF) << 24)
             | ((values[2] & 0xFF) << 16)
             | ((values[1] & 0xFF) << 8)
             | ((values[0] & 0xFF));

  nchw_out[out_buf_idx] = packed;
}
