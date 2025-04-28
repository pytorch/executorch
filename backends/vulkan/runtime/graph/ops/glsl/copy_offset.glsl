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
  // xyz is source offset w is channel size
  ivec4 src_offset;
  // xyz is destination offset w is channel size
  ivec4 dst_offset;
};

#include "indexing_utils.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "out_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 out_axis_map = unhash_axis_map(out_layout);

${layout_declare_spec_const(C, "int", "in_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 in_axis_map = unhash_axis_map(in_layout);

${layout_declare_spec_const(C, "int", "batch_index_function", "0")}

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, range))) {
    return;
  }

  ivec3 in_pos = pos + src_offset.xyz;
  ivec3 out_pos = pos + dst_offset.xyz;
  if (src_offset.w > 0) {
    if (batch_index_function == 1) {
      // batch index is calculated using source channel size
      const int channel_index = pos.z % src_offset.w;
      const int batch_index = pos.z / src_offset.w;
      out_pos.z = channel_index + dst_offset.z + batch_index * dst_offset.w;
    } else if (batch_index_function == 2) {
      // batch index is calculated using destination channel size
      const int channel_index = pos.z % dst_offset.w;
      const int batch_index = pos.z / dst_offset.w;
      in_pos.z = channel_index + src_offset.z + batch_index * src_offset.w;
    }
  }

  write_texel_lpos(
    t_out,
    out_pos,
    load_texel_lpos(t_in, in_pos, in_axis_map),
    out_axis_map);
}
