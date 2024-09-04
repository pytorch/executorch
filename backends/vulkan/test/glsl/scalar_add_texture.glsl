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

${layout_declare_tensor(0, "rw", "t_in", "float", "texture3d")}
${layout_declare_ubo(1, "uvec3", "extents")}
${layout_declare_ubo(2, "int", "scalar")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(pos, extents))) {
    return;
  }

  vec4 in_tex = imageLoad(t_in, pos);
  imageStore(t_in, pos, imageLoad(t_in, pos) + float(scalar));
}
