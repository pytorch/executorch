/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#include "indexing_utils.h"

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE)}

${layout_declare_ubo(B, "ivec3", "range", "ivec3", "src_offset", "ivec3", "dst_offset")}
${layout_declare_ubo(B, "ivec4", "out_axis_map")}
${layout_declare_ubo(B, "ivec4", "in_axis_map")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  const ivec3 out_pos = pos + dst_offset;
  const ivec3 in_pos = pos + src_offset;

  if (any(greaterThanEqual(pos, range))) {
    return;
  }

  write_texel_lpos(
    t_out,
    out_pos,
    load_texel_lpos(t_in, in_pos, in_axis_map),
    out_axis_map);
}
