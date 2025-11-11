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
${layout_declare_ubo(1, "ivec4", "sizes")}
${layout_declare_ubo(2, "float", "start")}
${layout_declare_ubo(3, "float", "step")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int packed_dim = C_DIM;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const ivec4 idx = to_tensor_idx(pos, sizes, packed_dim);

  if (pos_out_of_bounds(pos, sizes, packed_dim)) {
    return;
  }

  VEC4_T outtex = VEC4_T(start + pos.x * step, 0, 0, 0);

  imageStore(t_out, pos, outtex);
}
