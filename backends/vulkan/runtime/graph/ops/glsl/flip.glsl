/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}

#include "indexing_utils.h"

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE)}
${layout_declare_ubo(B, "ivec3", "out_limits")}
${layout_declare_ubo(B, "ivec4", "out_sizes")}
${layout_declare_ubo(B, "ivec4", "dims")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, out_limits))) {
    return;
  }

  VEC4_T out_texel = VEC4_T(0);
  uint src_x = pos.x;
  uint src_y = pos.y;
  uint src_z = pos.z;

  int flattened_channels = int(ceil(out_sizes.z / 4.0));

  // Width
  if (dims.x == 1) {
    src_x = out_sizes.x - 1 - pos.x;
  }
  // Height
  if (dims.y == 1) {
    src_y = out_sizes.y - 1 - pos.y;
  }
  // Batch
  if (dims.w == 1) {
    uint n = pos.z / flattened_channels;
    uint src_n = out_sizes.w - 1 - n;
    uint c4 = pos.z - n * flattened_channels;
    src_z = src_n * flattened_channels + c4;
  }

  uint prev_src_z = src_z;
  for (int p = 0; p < 4; ++p) {
    uint src_p = p;

    // Channel
    if (dims.z == 1) {
      uint nc = (pos.z / flattened_channels) * flattened_channels;
      uint c4 = pos.z - nc;
      uint c = c4 * 4 + p;
      uint src_c = out_sizes.z - 1 - c;

      src_z = (dims.w == 1)
          ? prev_src_z - c4 + src_c / 4 // Batch and Channel
          : nc + src_c / 4; // Channel only
      src_p = src_c % 4;
    }

    VEC4_T in_texel = VEC4_T(texelFetch(t_in, ivec3(src_x, src_y, src_z), 0));
    out_texel[p] = in_texel[src_p];
  }
  imageStore(t_out, pos, out_texel);
}
