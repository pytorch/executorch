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

/* Qualifiers: layout - storage - precision - memory */

// clang-format off
layout(set = 0, binding = 0, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly ${IMAGE_T[NDIM][DTYPE]} uOutput;
// clang-format on
layout(set = 0, binding = 1) uniform PRECISION restrict Block {
  ivec3 size;
  int fill;
  vec4 vals;
} params;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, params.size))) {
    return;
  }

  imageStore(uOutput, pos, params.vals);
}
