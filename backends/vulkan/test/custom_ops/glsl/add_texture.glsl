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

${define_active_storage_type("texture3d")}
${define_required_extensions(DTYPE)}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_in", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_other", DTYPE, "texture3d")}

layout(push_constant) uniform restrict Block {
  ivec4 out_sizes;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "out_layout", "0")}

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  // Simple addition without broadcasting - same position for all tensors
  VEC4_T in_texel = texelFetch(t_in, pos, 0);
  VEC4_T other_texel = texelFetch(t_other, pos, 0);

  imageStore(t_out, pos, in_texel + other_texel);
}
