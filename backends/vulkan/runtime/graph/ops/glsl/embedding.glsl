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
${layout_declare_tensor(1, "r", "t_in", "int", STORAGE)}
${layout_declare_tensor(2, "r", "t_weight", DTYPE, STORAGE)}
${layout_declare_ubo(3, "ivec4", "sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int packed_dim = C_DIM;

void main() {
  const ivec3 out_pos = ivec3(gl_GlobalInvocationID);

  if (pos_out_of_bounds(out_pos, sizes, packed_dim)) {
    return;
  }

  const ivec4 out_idx = to_tensor_idx(out_pos, sizes, packed_dim);
  VEC4_T out_texel;

  // Consider optimizing via W-packing format for t_in and t_weight.
  for (int i = 0; i < 4; ++i) {
    // Read input tensor for embedding index.
    const ivec3 in_pos = ivec3(out_pos.y, out_idx.z * 4 + i, out_idx.w / 4);
    const int in_texel_elem = texelFetch(t_in, in_pos, 0)[out_idx.w % 4];

    // Read weight tensor for embedding.
    out_texel[i] = texelFetch(t_weight, ivec3(out_pos.x, in_texel_elem, 0), 0).x;
  }

  imageStore(t_out, out_pos, out_texel);
}
