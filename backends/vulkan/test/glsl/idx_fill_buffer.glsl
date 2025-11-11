/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define T ${buffer_scalar_type(DTYPE)}

#include "indexing_utils.h"

${define_required_extensions(DTYPE)}

layout(std430) buffer;

${layout_declare_buffer(0, "w", "out_buf", DTYPE, PRECISION, True)}
${layout_declare_ubo(1, "int", "numel")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const int t_id = ivec3(gl_GlobalInvocationID).x;
  if (t_id >= numel) {
    return;
  }

  out_buf[t_id] = T(t_id);
}
