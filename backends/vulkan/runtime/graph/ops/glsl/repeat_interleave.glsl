/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}

${define_required_extensions(DTYPE)}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "tout", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "tin", DTYPE, STORAGE)}
${layout_declare_ubo(B, "ivec3", "tin_limits")}
${layout_declare_ubo(B, "ivec4", "tin_axis_map")}
${layout_declare_ubo(B, "ivec4", "tout_axis_map")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int nrepeats = 1;
layout(constant_id = 4) const int repeat_dim = 1;

#include "indexing_utils.h"

void main() {
  const ivec3 tin_lpos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(tin_lpos, tin_limits))) {
    return;
  }

  const VEC4_T intex = load_texel_lpos(tin, tin_lpos, tin_axis_map);

  ivec3 tout_lpos = tin_lpos;
  tout_lpos[repeat_dim] *= nrepeats;

  for (int i = 0; i < nrepeats; ++i, tout_lpos[repeat_dim]++) {
    write_texel_lpos(tout, tout_lpos, intex, tout_axis_map);
  }
}
