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
${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE)}

layout(push_constant) uniform restrict Block {
  ivec4 range;
  // source tensor sizes in WHCB dims respectively
  ivec4 src_dims;
  // destination tensor sizes in WHCB dims respectively
  ivec4 dst_dims;
};

#include "indexing_utils.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "out_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 out_axis_map = unhash_axis_map(out_layout);
const lowp int packed_dim = unhash_packed_dim(out_layout);

${layout_declare_spec_const(C, "int", "in_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 in_axis_map = unhash_axis_map(in_layout);

void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, range.xyz))) {
    return;
  }

  // expand position in packed dim
  pos[packed_dim] <<= 2;

  // channel size aligned by 4 when tensors are channel packed raw value otherwise
  const int channel_size = (packed_dim == C_DIM ? alignup4(src_dims.z) : src_dims.z);

  // find input texel's WHCB index
  const int width_index = pos.x % src_dims.x;
  const int height_index = pos.y % src_dims.y;
  int channel_index;
  int batch_index;

  // if tensors are channel packed
  if (packed_dim == C_DIM) {
    // the output channels in a batch will be channel size * channel repetitions aligned by 4
    const int out_channel_size = alignup4(src_dims.z * dst_dims.z);

    // batch index in the output
    const int out_pos_batch_index = pos.z / out_channel_size;

    // source batch index for based on current output pos
    batch_index = out_pos_batch_index % src_dims.w;

    // batch repetition count for current output pos
    const int batch_repetition_index = out_pos_batch_index / src_dims.w;

    // calculate input channel index based on current output pos and batch index
    // its done this way because we want source channel to restart from zero when a batch index increments
    // also batch_index will reset to zero after hitting batch repetition count
    // so track the current repetition in batch_repetition_index so it can be used for determining current_index
    channel_index = (pos.z - (batch_index + batch_repetition_index * src_dims.w) * out_channel_size) % src_dims.z;
  } else {
    // the output channels in a batch will be channel size * channel repetitions
    const int out_channel_size = src_dims.z * dst_dims.z;

    // source batch index for based on current output pos
    batch_index = (pos.z / out_channel_size) % src_dims.w;

    // source channel index is current output pos wrapped based on channel count
    channel_index = pos.z % src_dims.z;
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
    src_lane_offset = mix(src_lane_offset, 0, src_lane_offset >= src_dims[packed_dim]);
  }

  write_texel_lpos(
    t_out,
    pos,
    out_value,
    out_axis_map);
}
