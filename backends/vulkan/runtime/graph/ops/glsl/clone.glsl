/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#include "indexing_utils.h"

#define PRECISION ${PRECISION}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE)}

layout(push_constant) uniform restrict Block {
  ivec3 out_limits;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(pos, out_limits))) {
    return;
  }
  imageStore(t_out, pos, load_texel(t_in, pos));
}
