/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#include "broadcasting_utils.h"
#include "indexing_utils.h"

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}

layout(std430) buffer;

${layout_declare_tensor(0, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(1, "r", "t_in", DTYPE, STORAGE)}
${layout_declare_ubo(2, "ivec3", "out_limits")}
${layout_declare_ubo(3, "ivec2", "input_size")}
${layout_declare_ubo(4, "vec2", "rev_scales")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, out_limits))) {
    return;
  }

  const ivec2 ipos = clamp(ivec2(pos.xy * rev_scales), ivec2(0), input_size);

  VEC4_T in_texel = texelFetch(t_in, ivec3(ipos, pos.z), 0);
  imageStore(t_out, pos, in_texel);
}
