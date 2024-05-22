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

layout(set = 0, binding = 2) uniform PRECISION restrict OutLimits {
  ivec3 out_limits;
};

layout(set = 0, binding = 3) uniform PRECISION restrict Sizes {
  ivec4 sizes;
};

layout(set = 0, binding = 4) uniform PRECISION restrict Block {
  // output dims
  ivec4 out_ndims;
  // x = output channels aligned to 4, y = input channels aligned to 4
  ivec2 ch_info;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, out_limits))) {
    return;
  }

  const int out_channel_4up = int(ch_info.x);
  const int in_channel_4up = int(ch_info.y);
  const int out_batch = int(sizes[3]);
  const int max_dst_index = out_batch * out_channel_4up;
  VEC4_T outval = VEC4_T(0.0);

  for (int j = 0; j < 4; ++j) {
    int dst_index = pos.z * 4 + j;
    if (dst_index >= max_dst_index) {
      // out of range
      break;
    }

    ivec4 v = ivec4(0); // holds b,c,h,w
    v[out_ndims[0]] = dst_index / out_channel_4up;
    v[out_ndims[1]] = dst_index % out_channel_4up;
    v[out_ndims[2]] = pos.y;
    v[out_ndims[3]] = pos.x;

    int src_index = v[0] * in_channel_4up + v[1];
    int w = v[3];
    int h = v[2];

    VEC4_T inval = VEC4_T(texelFetch(image_in, ivec3(w, h, src_index / 4), 0));
    outval[j] = inval[src_index % 4];
  }

  imageStore(image_out, pos, outval);
}
