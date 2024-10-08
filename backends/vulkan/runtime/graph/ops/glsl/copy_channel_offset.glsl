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

#include "indexing_utils.h"

${layout_declare_tensor(0, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(1, "r", "existing_out", DTYPE, STORAGE)}
${layout_declare_tensor(2, "r", "t_in", DTYPE, STORAGE)}

${layout_declare_ubo(3, "ivec4", "out_sizes")}
${layout_declare_ubo(4, "ivec4", "out_axis_map")}
${layout_declare_ubo(5, "ivec4", "in_sizes")}
${layout_declare_ubo(6, "ivec4", "in_axis_map")}
layout(set = 0, binding = 7) uniform PRECISION restrict CopyArgs {
  // Operates on (x, y, z) logical extents.
  ivec3 range;
  // Analogus to range variable in copy. It defines the # of channel being
  // copied.
  int channel_range;
  ivec3 dst_offset;
  int dst_channel_offset;
  int src_channel_offset;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int packed_dim = C_DIM;

void main() {
  // Note: Unlike other shaders, the range is often not equal to the destination
  // texture extent.
  const ivec3 lpos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(lpos, range))) {
    return;
  }

  const ivec3 out_lpos = lpos + dst_offset;

  const ivec4 out_tidx = lpos_to_tidx(out_lpos, out_sizes, out_axis_map.w, packed_dim);

  // First read the existing values to make sure the boundary values stay.
  VEC4_T v = load_texel_lpos(existing_out, out_lpos, out_axis_map);

  ivec4 in_tidx = out_tidx;
  for (int i=0; i<4; i++) {

    in_tidx[packed_dim] = out_tidx[packed_dim] - dst_channel_offset + i;

    // Handle the partial update for begining of channel in an existing tensor.
    // If the source channel index is below zero or exceeds the range, we skip
    // updating the element to avoid overwriting existing data.
    if ((in_tidx[packed_dim] < 0) || (in_tidx[packed_dim] >= channel_range)) {
      continue;
    }

    // Readjust for the source offset.
    in_tidx[packed_dim] += src_channel_offset;

    ivec4 in_posi = tidx_to_posi(in_tidx, in_sizes, in_axis_map, packed_dim);
    v[i] = load_texel(t_in, in_posi.xyz)[in_posi.w];
  }

  write_texel_lpos(t_out, out_lpos, v, out_axis_map);
}
