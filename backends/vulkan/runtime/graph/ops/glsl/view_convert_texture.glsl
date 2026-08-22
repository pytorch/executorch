/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define IN_VEC4_T ${texel_type(IN_DTYPE)}
#define OUT_VEC4_T ${texel_type(OUT_DTYPE)}

${define_required_extensions("texture3d", IN_DTYPE)}
${define_required_extensions("texture3d", OUT_DTYPE)}

#include "indexing_utils.h"

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", OUT_DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_in", IN_DTYPE, "texture3d")}

layout(push_constant) uniform restrict Block {
  ivec4 out_sizes;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int packed_dim = C_DIM;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const ivec4 idx = to_tensor_idx(pos, out_sizes, packed_dim);

  if (any(greaterThanEqual(idx, out_sizes))) {
    return;
  }

  IN_VEC4_T in_texel = IN_VEC4_T(texelFetch(t_in, pos, 0));
  imageStore(t_out, pos, OUT_VEC4_T(in_texel));
}
