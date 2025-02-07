/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

${define_active_storage_type(STORAGE)}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE)}

layout(push_constant) uniform restrict Block {
  ivec3 range;
  ivec3 src_offset;
  ivec3 dst_offset;
};

#include "indexing_utils.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "out_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 out_axis_map = unhash_axis_map(out_layout);

${layout_declare_spec_const(C, "int", "in_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 in_axis_map = unhash_axis_map(in_layout);

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
