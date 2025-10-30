/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

layout(std430) buffer;

${layout_declare_tensor(0, "w", "t_out", "float", "texture3d")}
${layout_declare_tensor(1, "r", "t_in1", "float", "texture3d")}
${layout_declare_tensor(2, "r", "t_in2", "float", "texture3d")}

layout(push_constant) uniform restrict Block {
  ivec4 out_sizes;
  ivec4 in1_sizes;
  ivec4 in2_sizes;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, out_sizes.xyz))) {
    return;
  }


  vec4 out_texel = vec4(0.0);
  for (int row = 0; row < in1_sizes.y; ++row) {
    ivec3 in_pos = ivec3(pos.x, row, pos.z);
    vec4 in1_texel = texelFetch(t_in1, in_pos, 0);
    vec4 in2_texel = texelFetch(t_in2, in_pos, 0);

    out_texel += in1_texel * in2_texel;
  }

  imageStore(t_out, pos, out_texel + ${OFFSET});
}
