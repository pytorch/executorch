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
${layout_declare_ubo(2, "ivec4", "out_sizes")}
${layout_declare_ubo(3, "ivec4", "in_sizes")}

layout(set = 0, binding = 4) uniform PRECISION restrict SliceArg {
  int offset;
  int step;
}
slice_arg;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int packed_dim = C_DIM;

void main() {
  const ivec3 out_pos = ivec3(gl_GlobalInvocationID);
  const ivec4 idx = to_tensor_idx(out_pos, out_sizes, packed_dim);

  if (any(greaterThanEqual(idx, out_sizes))) {
    return;
  }

  // We map the output pos using the buffer index.  For each index in the texel,
  // we calculate the source whcn-coordinate amended with offset-ed channel
  // value.  Then we calculate the actual texture position from the
  // whcn-coordinate.
  const ivec4 buf_indices = get_texel_nchw_buffer_ixs(idx, out_sizes, packed_dim);

  vec4 outex;
  for (int i=0;i<4;i++) {
      ivec4 user_coor = from_nchw_buffer_i(buf_indices[i], out_sizes);

      int in_channel = user_coor.z;

      ivec4 in_user_coor = user_coor;
      in_user_coor.z = slice_arg.offset + in_channel * slice_arg.step;

      ivec4 in_pow_elem = to_texture_elem_pos(
        in_user_coor,
        in_sizes,
        packed_dim);

      vec4 v = texelFetch(t_in, in_pow_elem.xyz, 0);

      outex[i] = v[in_pow_elem.w];
  }
  imageStore(t_out, out_pos, outex);
}
