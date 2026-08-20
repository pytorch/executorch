/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

$PRECISION = "highp"
$DTYPE = "float"

#define PRECISION ${PRECISION}

#define VEC4_T ${buffer_gvec_type(DTYPE, 4)}

#include "indexing_utils.h"

layout(std430) buffer;

layout(set = 0, binding = 0) buffer  PRECISION restrict writeonly Buffer {
  VEC4_T data[];
}
buffer_in;

layout(set = 0, binding = 1) uniform PRECISION restrict Params {
  int len;
}
params;



layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const float scale = 1;
layout(constant_id = 4) const float offset = 0;

void main() {
  const int i = ivec3(gl_GlobalInvocationID).x;

  const int base = 4 * i;
  if (base < params.len) {
    buffer_in.data[i] = scale * (VEC4_T(base) + VEC4_T(0, 1, 2, 3)) + offset;
  }
}
