/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_type(DTYPE)}

layout(std430) buffer;

#include "indexing_utils.h"

${layout_declare_tensor(0, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(1, "r", "t_in", DTYPE, STORAGE)}
${layout_declare_tensor(2, "r", "t_idx", "int", STORAGE)}
${layout_declare_ubo(3, "ivec4", "out_sizes")}
${layout_declare_ubo(4, "ivec4", "in_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int packed_dim = C_DIM;

void main() {
  const ivec3 out_pos = ivec3(gl_GlobalInvocationID);

  if (pos_out_of_bounds(out_pos, out_sizes, packed_dim)) {
    return;
  }

  const ivec4 idx = to_tensor_idx(out_pos, out_sizes, packed_dim);
  const ivec4 buffer_ixs = get_texel_nchw_buffer_ixs(idx, out_sizes, packed_dim);

  VEC4_T out_texel;
  for (int i = 0; i < 4; ++i) {
      const ivec4 out_idx = from_nchw_buffer_i(buffer_ixs[i], out_sizes);
      int out_channel = out_idx.z;
      int in_channel = texelFetch(t_idx, ivec3(out_channel, 0, 0), 0).x;

      ivec4 in_idx = out_idx;
      in_idx.z = in_channel;

      ivec4 in_elem_pos = to_texture_elem_pos(in_idx, in_sizes, packed_dim);

      VEC4_T in_texel = texelFetch(t_in, in_elem_pos.xyz, 0);

      out_texel[i] = in_texel[in_elem_pos.w];
  }
  imageStore(t_out, out_pos, out_texel);
}
