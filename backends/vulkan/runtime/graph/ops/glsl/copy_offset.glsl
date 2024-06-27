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

${layout_declare_tensor(0, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(1, "r", "t_in", DTYPE, STORAGE)}

layout(set = 0, binding = 2) uniform PRECISION restrict CopyArgs {
  ivec3 range;
  int unused0;
  ivec3 src_offset;
  int unused1;
  ivec3 dst_offset;
  int unused2;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  const ivec3 out_pos = pos + dst_offset;
  const ivec3 in_pos = pos + src_offset;

  if (any(greaterThanEqual(pos, range))) {
    return;
  }

  imageStore(t_out, out_pos, texelFetch(t_in, in_pos, 0));
}
