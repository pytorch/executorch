/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

${define_active_storage_type("texture3d")}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", "float", "texture3d")}
${layout_declare_tensor(B, "r", "nchw_in", "uint", "buffer")}

${layout_declare_ubo(B, "ivec3", "out_limits")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 lpos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(lpos, out_limits))) {
    return;
  }

  // Placeholder: just copy input to output
  vec4 in_texel = vec4(1.0f);
  imageStore(t_out, lpos, in_texel);
}
