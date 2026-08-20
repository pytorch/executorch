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


${layout_declare_buffer(0, "r", "source", DTYPE)}
${layout_declare_buffer(1, "w", "destination", DTYPE)}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int niter = 1;
layout(constant_id = 4) const int stride = 1;
layout(constant_id = 5) const int pitch = 1;

void main() {
  float c = 0;
  for (int i = 0; i < niter; ++i) {
    const int zero = i >> 31;
    c += source[zero + pitch * gl_GlobalInvocationID[0]];
    c += source[zero + stride + pitch * gl_GlobalInvocationID[0]];
  }
  destination[0] = c;
}
