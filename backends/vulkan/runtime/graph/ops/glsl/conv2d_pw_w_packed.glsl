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

#define TILE_SIZE_X ${TILE_SIZE_X}
#define TILE_SIZE_Y ${TILE_SIZE_Y}
#define LOCAL_WG_SIZE 64

#define op(X, A, B) ${OPERATOR}

#include "indexing_utils.h"

layout(std430) buffer;

${layout_declare_tensor(0, "w", "t_out", DTYPE, "texture3d")}
${layout_declare_tensor(1, "r", "t_in", DTYPE, "texture3d")}
${layout_declare_tensor(2, "r", "t_kernel", DTYPE, "texture2d")}
${layout_declare_tensor(3, "r", "t_bias", DTYPE, "texture2d")}

layout(push_constant) uniform restrict Block {
  ivec4 out_limits;
  ivec2 stride;
  ivec2 padding;
  int in_group_size;
  int in_channels;
  float out_min;
  float out_max;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#extension GL_EXT_control_flow_attributes : require

// For performance improvement, reduce register usage by caching positions in shared memory.
// Offset index by 1 every 16 points to avoid bank access conflict.
#define offset_pos_index(index) ((index) + ((index) >> 4))
shared ivec3 pos_shared[offset_pos_index(LOCAL_WG_SIZE * TILE_SIZE_X * TILE_SIZE_Y)];

/*
 * Computes a 2D pointwise convolution of an NxN output tile. Calculating an
 * output tile for pointwise convolution is more efficient because the kernel
 * size is only 1x1, making it easier to re-use loaded texels from t_kernel.
 */
void main() {
  const ivec2 out_limits_scaled = (out_limits.xy + ivec2(TILE_SIZE_X - 1, TILE_SIZE_Y - 1)) / ivec2(TILE_SIZE_X, TILE_SIZE_Y);
  const uint shared_mem_stride = LOCAL_WG_SIZE;

  const uint div_by_x = gl_GlobalInvocationID.x / out_limits_scaled.x;
  const ivec3 gpos = ivec3(
    gl_GlobalInvocationID.x % out_limits_scaled.x,
    div_by_x % out_limits_scaled.y,
    div_by_x / out_limits_scaled.y);

  // Output position for TILE_SIZE_X = 2, TILE_SIZE_Y = 2
  // +--------+--------+
  // | pos[0] | pos[1] |
  // +--------+--------+
  // | pos[2] | pos[3] |
  // +--------+--------+
  ivec2 pos[TILE_SIZE_X * TILE_SIZE_Y];
  [[unroll]] for (int y = 0; y < TILE_SIZE_Y; ++y) {
    [[unroll]] for (int x = 0; x < TILE_SIZE_X; ++x) {
      const int index = y * TILE_SIZE_X + x;
      pos[index] = ivec2(gpos.x * TILE_SIZE_X + x, gpos.y * TILE_SIZE_Y + y);
      pos_shared[offset_pos_index((shared_mem_stride * index) + gl_LocalInvocationIndex)] = ivec3(pos[index], gpos.z);
    }
  }

  // If the channel position is out of bounds, then this invocation has no work to do.
  if (gpos.z >= out_limits.z) {
    return;
  }

  // check if input access in packed axis is 4 element aligned
  const bool x_access_4_aligned = (padding.x == 0 && stride.x == 1);

  // channel index for this invocation
  const int channel_index = gpos.z >> 2;

  // channel lane for this invocation
  const int channel_lane = gpos.z & 3;

  // Compute the index of the input texture that needs to be loaded for each
  // output position. Note that negative indices can be produced indicating that
  // the top-left element is in a region added by padding.
  ivec2 ipos[TILE_SIZE_X * TILE_SIZE_Y];

  if (x_access_4_aligned) {
    [[unroll]] for (int i = 0; i < TILE_SIZE_X * TILE_SIZE_Y; ++i) {
      // In case of 4 element aligned access, input to output texel mapping will be 1:1
      // Thus we can use the output position as the input position
      ipos[i] = pos[i];
    }
  } else {
    [[unroll]] for (int i = 0; i < TILE_SIZE_X * TILE_SIZE_Y; ++i) {
      ipos[i] = ivec2(pos[i].x << 2, pos[i].y) * stride - padding;
    }
  }

  vec4 sum[TILE_SIZE_X * TILE_SIZE_Y];
  sum[0] = vec4(texelFetch(t_bias, ivec2(channel_index, 0), 0)[channel_lane]);
  [[unroll]] for (int i = 1; i < TILE_SIZE_X * TILE_SIZE_Y; ++i) {
    sum[i] = sum[0];
  }

  if (x_access_4_aligned) {
    // Since the kernel is 1x1, we only have to loop over the depth dimension.
    for (int z = 0; z < in_group_size; z++) {
      // During prepacking, the weight tensor has been permuted so that the
      // channel (IC) dim is along the x-axis, and the batch (OC) dim is along
      // the z-axis.
      const vec4 k_tex = texelFetchOffset(t_kernel, ivec2(z, channel_index), 0, ivec2(0, 0));

      [[unroll]] for (int i = 0; i < TILE_SIZE_X * TILE_SIZE_Y; ++i) {
        const vec4 in_tex = texelFetch(t_in, ivec3(ipos[i], z), 0);
        sum[i] = fma(in_tex, vec4(k_tex[channel_lane]), sum[i]);
      }
    }
  } else {
    // Since the kernel is 1x1, we only have to loop over the depth dimension.
    for (int z = 0; z < in_group_size; z++) {
      // During prepacking, the weight tensor has been permuted so that the
      // channel (IC) dim is along the x-axis, and the batch (OC) dim is along
      // the z-axis.
      const vec4 k_tex = texelFetchOffset(t_kernel, ivec2(z, channel_index), 0, ivec2(0, 0));
      [[unroll]] for (int i = 0; i < TILE_SIZE_X * TILE_SIZE_Y; ++i) {
        int prev_pos_x = 0x7FFFFFFF;
        vec4 prev_in_tex;
        [[unroll]] for (int j = 0; j < 4; ++j) {
          const int ipos_x = ipos[i].x + j * stride.x;
          const int fetch_ipos_x = ipos_x >> 2;
          if (fetch_ipos_x != prev_pos_x) {
            prev_in_tex = texelFetch(t_in, ivec3(fetch_ipos_x, ipos[i].y, z), 0);
            prev_pos_x = fetch_ipos_x;
          }
          float val = prev_in_tex[ipos_x & 3];
          val = mix(val, 0, ipos_x < 0);
          sum[i][j] = fma(val, k_tex[channel_lane], sum[i][j]);
        }
      }
    }
  }

  [[unroll]] for (int i = 0; i < TILE_SIZE_X * TILE_SIZE_Y; ++i) {
    const uint index = (shared_mem_stride * i) + gl_LocalInvocationIndex;
    const ivec3 pos = pos_shared[offset_pos_index(index)];
    if (all(lessThan(pos, out_limits.xyz))) {
      imageStore(t_out, pos, op(sum[i], out_min, out_max));
    }
  }
}
