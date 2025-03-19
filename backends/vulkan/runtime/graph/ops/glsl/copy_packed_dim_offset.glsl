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

  // if not repeating
  // xyz is source offset w is channel size
  // if repeating
  // xyzw is source tensor sizes in WHCB dims respectively
  ivec4 src_offset;

  // if not repeating
  // xyz is destination offset w is channel size
  // if repeating
  // xyzw is destination tensor sizes in WHCB dims respectively
  ivec4 dst_offset;
};

#include "indexing_utils.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "out_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 out_axis_map = unhash_axis_map(out_layout);
const lowp int packed_dim = unhash_packed_dim(out_layout);

${layout_declare_spec_const(C, "int", "in_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 in_axis_map = unhash_axis_map(in_layout);

${layout_declare_spec_const(C, "bool", "repeat", "false")}

void no_repeat_copy(ivec3 pos) {
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

void repeat_copy(ivec3 pos) {
  // expand position in packed dim
  pos[packed_dim] <<= 2;

  // channel size aligned by 4 when tensors are channel packed raw value otherwise
  const int channel_size = (packed_dim == C_DIM ? alignup4(src_offset.z) : src_offset.z);

  // find input texel's WHCB index
  const int width_index = pos.x % src_offset.x;
  const int height_index = pos.y % src_offset.y;
  int channel_index;
  int batch_index;

  // if tensors are channel packed
  if (packed_dim == C_DIM) {
    // the output channels in a batch will be channel size * channel repetitions aligned by 4
    const int out_channel_size = alignup4(src_offset.z * dst_offset.z);

    // batch index in the output
    const int out_pos_batch_index = pos.z / out_channel_size;

    // source batch index for based on current output pos
    batch_index = out_pos_batch_index % src_offset.w;

    // batch repetition count for current output pos
    const int batch_repetition_index = out_pos_batch_index / src_offset.w;

    // calculate input channel index based on current output pos and batch index
    // its done this way because we want source channel to restart from zero when a batch index increments
    // also batch_index will reset to zero after hitting batch repetition count
    // so track the current repetition in batch_repetition_index so it can be used for determining current_index
    channel_index = (pos.z - (batch_index + batch_repetition_index * src_offset.w) * out_channel_size) % src_offset.z;
  } else {
    // the output channels in a batch will be channel size * channel repetitions
    const int out_channel_size = src_offset.z * dst_offset.z;

    // source batch index for based on current output pos
    batch_index = (pos.z / out_channel_size) % src_offset.w;

    // source channel index is current output pos wrapped based on channel count
    channel_index = pos.z % src_offset.z;
  }

  // input texel's WCB position
  const ivec3 in_pos = ivec3(width_index, height_index, channel_index);

  // squeeze position in packed dim
  pos[packed_dim] >>= 2;

  // packed dim index of texel last fetched
  int fetched_in_pos_packed_dim = -1;

  // fetched input texel
  VEC4_T in_value;

  // output texel value
  VEC4_T out_value = VEC4_T(0);

  int src_lane_offset = in_pos[packed_dim];

  for (int i=0; i<4; i++) {
    if ((src_lane_offset >> 2) != fetched_in_pos_packed_dim) {
      fetched_in_pos_packed_dim = (src_lane_offset >> 2);

      ivec3 curr_in_pos = in_pos;
      curr_in_pos[packed_dim] = src_lane_offset;
      curr_in_pos.z = curr_in_pos.z + batch_index * channel_size;
      curr_in_pos[packed_dim] >>= 2;

      in_value = load_texel_lpos(t_in, curr_in_pos, in_axis_map);
    }

    out_value[i] = in_value[src_lane_offset & 0x3];

    src_lane_offset++;
    // if packed index exceeded source packed dim round to zero
    src_lane_offset = mix(src_lane_offset, 0, src_lane_offset >= src_offset[packed_dim]);
  }

  write_texel_lpos(
    t_out,
    pos,
    out_value,
    out_axis_map);
}

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, range.xyz))) {
    return;
  }

  if (repeat) {
    repeat_copy(pos);
  } else {
    no_repeat_copy(pos);
  }
}
