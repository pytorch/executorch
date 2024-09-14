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

${layout_declare_tensor(B, "w", "t_out", "int8", "texture3d")}
${layout_declare_buffer(B, "r", "nchw_in", "int")}
${layout_declare_ubo(B, "ivec4", "sizes")}
${layout_declare_ubo(B, "ivec4", "axis_map")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int packed_dim = C_DIM;

/*
 * Extends sign of int8
 */
int extend_sign(int x) {
  if (x >> 7 == 1) {
    return x | 0xFFFFFF00;
  }
  return x;
}

ivec4 read_texel(ivec4 tensor_idx) {
  const ivec4 buf_indices = get_texel_nchw_buffer_ixs(
      tensor_idx, sizes, packed_dim);

  int shift = (1 << 8) - 1;
  ivec4 masks;
  // Masks used to unpack 4x 8-bit values from a 32 bit integer. Note that
  // little endian is assumed, as most processors use little endian. Thus the
  // most significant bytes correspond to the "latter" packed values.
  masks.x = shift << (8 * (buf_indices.x % 4));
  masks.y = shift << (8 * (buf_indices.y % 4));
  masks.z = shift << (8 * (buf_indices.z % 4));
  masks.w = shift << (8 * (buf_indices.w % 4));

  ivec4 out_tex = ivec4(0);

  [[unroll]] for (int i = 0; i < 4; ++i) {
    if (tensor_idx[packed_dim] + i < sizes[packed_dim]) {
      int in_texel = nchw_in[buf_indices[i] / 4];
      int extracted_val = (in_texel & masks[i]) >> (8 * (buf_indices[i] % 4));
      extracted_val = extend_sign(extracted_val);
      out_tex[i] = extracted_val;
    }
  }

  return out_tex;
}

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const ivec4 tensor_idx = to_tensor_idx(pos, sizes, axis_map, packed_dim);

  if (any(greaterThanEqual(tensor_idx, sizes))) {
    return;
  }

  write_texel(t_out, pos, read_texel(tensor_idx));
}
