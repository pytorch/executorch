/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_type(DTYPE)}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "existing_out", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE)}

layout(push_constant) uniform restrict Block {
  ivec4 out_sizes;
  ivec4 in_sizes;
  // Operates on (x, y, z) logical extents.
  // channel_range is stored in range.w
  ivec4 range;
  // Analogus to range variable in copy. It defines the # of channel being
  // copied.
  // dst channel offset is stored in dst_offset.w
  ivec4 dst_offset;
  int src_channel_offset;
};

#include "indexing_utils.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "out_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 out_axis_map = unhash_axis_map(out_layout);
const lowp int packed_dim = unhash_packed_dim(out_layout);

${layout_declare_spec_const(C, "int", "in_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 in_axis_map = unhash_axis_map(in_layout);

void main() {
  // Note: Unlike other shaders, the range is often not equal to the destination
  // texture extent.
  const ivec3 lpos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(lpos, range.xyz))) {
    return;
  }

  const ivec3 out_lpos = lpos + dst_offset.xyz;

  const ivec4 out_tidx = lpos_to_tidx(out_lpos, out_sizes, out_axis_map.w, packed_dim);

  // First read the existing values to make sure the boundary values stay.
  VEC4_T v = load_texel_lpos(existing_out, out_lpos, out_axis_map);

  ivec4 in_tidx = out_tidx;
  for (int i=0; i<4; i++) {

    in_tidx[packed_dim] = out_tidx[packed_dim] - dst_offset.w + i;

    // Handle the partial update for begining of channel in an existing tensor.
    // If the source channel index is below zero or exceeds the range, we skip
    // updating the element to avoid overwriting existing data.
    if ((in_tidx[packed_dim] < 0) || (in_tidx[packed_dim] >= range.w)) {
      continue;
    }

    // Readjust for the source offset.
    in_tidx[packed_dim] += src_channel_offset;

    ivec4 in_posi = tidx_to_posi(in_tidx, in_sizes, in_axis_map, packed_dim);
    v[i] = load_texel(t_in, in_posi.xyz)[in_posi.w];
  }

  write_texel_lpos(t_out, out_lpos, v, out_axis_map);
}
