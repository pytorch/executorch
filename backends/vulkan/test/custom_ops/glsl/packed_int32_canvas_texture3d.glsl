/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

${define_active_storage_type("texture2d")}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", "int", "texture3d")}
${layout_declare_tensor(B, "r", "nchw_in", "uint", "buffer")}

${layout_declare_ubo(B, "ivec3", "out_limits")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 lpos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(lpos, out_limits))) {
    return;
  }

  // Pack four 8-bit values equal to 1 into a single uint
  int packed = (1 << 0) | (1 << 8) | (1 << 16) | (1 << 24);

  debugPrintfEXT(
      "t_out[%i, %i] = %i\\n",
      lpos.x, lpos.y,
      packed);


  // Placeholder: just copy input to output
  ivec4 in_texel = ivec4(packed);
  imageStore(t_out, lpos, in_texel);
}
