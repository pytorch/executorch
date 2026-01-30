/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions(STORAGE, DTYPE)}

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "tout", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "tin", DTYPE, STORAGE)}
${layout_declare_ubo(B, "ivec3", "tin_limits")}

#include "indexing_utils.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "tout_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 tout_axis_map = unhash_axis_map(tout_layout);

${layout_declare_spec_const(C, "int", "tin_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 tin_axis_map = unhash_axis_map(tin_layout);

${layout_declare_spec_const(C, "int", "nrepeats", "1")}
${layout_declare_spec_const(C, "int", "repeat_dim", "1")}

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
