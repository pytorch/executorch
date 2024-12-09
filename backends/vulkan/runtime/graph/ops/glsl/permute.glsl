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

layout(set = 0, binding = 0, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly ${IMAGE_T[NDIM][DTYPE]} image_out;
layout(set = 0, binding = 1) uniform PRECISION ${SAMPLER_T[NDIM][DTYPE]} image_in;

layout(push_constant) uniform PRECISION restrict Block {
  ivec4 out_limits;
  ivec4 sizes;
  // output dims
  ivec4 out_ndims;
  // x = output channels aligned to 4, y = input channels aligned to 4
  ivec2 ch_info;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require

void main() {
  const u16vec3 pos = u16vec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, out_limits.xyz))) {
    return;
  }

  const int out_channel_4up = int(ch_info.x);
  const int in_channel_4up = int(ch_info.y);
  const int out_batch = int(sizes[3]);
  VEC4_T outval = VEC4_T(0.0);
  ivec4 v = ivec4(0); // holds b,c,h,w

  v[out_ndims[2]] = pos.y;
  v[out_ndims[3]] = pos.x;

  const int dst_index = pos.z << 2;
  int dst_out_index = dst_index / out_channel_4up;
  int dst_out_lane = dst_index % out_channel_4up;

  for (int j = 0; j < 4; ++j, ++dst_out_lane) {
    if (dst_out_index >= out_batch) {
      // out of range
      break;
    }

    if (dst_out_lane == out_channel_4up) {
      dst_out_lane = 0;
      dst_out_index++;
    }

    v[out_ndims[0]] = dst_out_index;
    v[out_ndims[1]] = dst_out_lane;

    int src_index = v[0] * in_channel_4up + v[1];

    VEC4_T inval = VEC4_T(texelFetch(image_in, u16vec3(v[3], v[2], src_index >> 2), 0));
    outval[j] = inval[src_index & 0x3];
  }

  imageStore(image_out, pos, outval);
}
