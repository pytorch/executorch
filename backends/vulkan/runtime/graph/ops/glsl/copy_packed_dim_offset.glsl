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
  ivec4 range;

  // xyz is source offset w is channel size
  ivec4 src_offset;

  // xyz is destination offset w is channel size
  ivec4 dst_offset;
};

#include "indexing_utils.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "out_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 out_axis_map = unhash_axis_map(out_layout);
const lowp int packed_dim = unhash_packed_dim(out_layout);

${layout_declare_spec_const(C, "int", "in_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 in_axis_map = unhash_axis_map(in_layout);

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, range.xyz))) {
    return;
  }

  // Position in input tensor
  ivec3 in_pos = pos + src_offset.xyz;
  in_pos[packed_dim] = pos[packed_dim] + (src_offset[packed_dim] >> 2);

  // Read input value mapping to this output texel
  VEC4_T in_value = load_texel_lpos(t_in, in_pos, in_axis_map);

  // Starting offset to read from a texel
  const int src_lane_offset = src_offset[packed_dim] & 0x3;
  const bool has_src_lane_offset = src_lane_offset != 0;

  // If input lane offset is non zero i.e packed texel is composed from multiple sources
  if (has_src_lane_offset) {
    // Boundary values will come from next input texel in the packed dim.
    ivec3 next_in_pos = in_pos;
    next_in_pos[packed_dim] = in_pos[packed_dim] + 1;
    VEC4_T next_value = load_texel_lpos(t_in, next_in_pos, in_axis_map);

    // Keep input values from the end of current input pixel based on src_lane_offset
    // offset 1 means the first lane of current input texel is not a part of the output texel
    // offset 2 means first 2 lanes are not and so on
    // Copy next texel's values towards the end of input texel, based on lane offset
    // offset 1 means the first lane from next texel is part of the input texel
    // offset 2 means first 2 lanes from next texel is part of the input texel and so on
    if (src_lane_offset == 1) {
      in_value = ivec4(in_value.yzw, next_value.x);
    } else if (src_lane_offset == 2) {
      in_value = ivec4(in_value.zw, next_value.xy);
    } else {
      in_value = ivec4(in_value.w, next_value.xyz);
    }
  }

  // Starting offset to write at within a texel
  const int out_lane_offset = dst_offset[packed_dim] & 0x3;
  const bool has_dst_lane_offset = out_lane_offset != 0;

  ivec3 out_pos = pos + dst_offset.xyz;
  out_pos[packed_dim] = pos[packed_dim] + (dst_offset[packed_dim] >> 2);

  VEC4_T out_value;

  // If lane offset is non zero i.e packed texel is composed from multiple sources
  if (has_dst_lane_offset) {
    // When position in packed dim is > 0
    if (pos[packed_dim] > 0) {
      // Boundary values will come from previous input texel in the packed dim.
      ivec3 prev_in_pos = in_pos;
      prev_in_pos[packed_dim] = in_pos[packed_dim] - 1;
      VEC4_T prev_value = load_texel_lpos(t_in, prev_in_pos, in_axis_map);

      // Shift values toward the beginning based on out_lane_offset
      // offset 1 means the last lane from the previous texel is a part of the output texel
      // offset 2 means last 2 lanes and so on
      if (out_lane_offset == 1) {
        out_value.x = prev_value.w;
      } else if (out_lane_offset == 2) {
        out_value.xy = prev_value.zw;
      } else {
        out_value.xyz = prev_value.yzw;
      }
    } else {
      // When position in packed dim is == 0
      // Boundary values will be the previous texel values.
      out_value = load_texel_lpos(existing_out, out_pos, out_axis_map);
    }

    // Copy input values towards the end of output array, based on lane offset
    // offset 1 means the first lane from previous texel is part of the output texel starting at offset
    // offset 2 means first 2 lanes from the previous texel is part of the output texel and so on
    if (out_lane_offset == 1) {
      out_value.yzw = in_value.xyz;
    } else if (out_lane_offset == 2) {
      out_value.zw = in_value.xy;
    } else {
      out_value.w = in_value.x;
    }
  } else {
    out_value = in_value;
  }

  write_texel_lpos(
    t_out,
    out_pos,
    out_value,
    out_axis_map);
}
