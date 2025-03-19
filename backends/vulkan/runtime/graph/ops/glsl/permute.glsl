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
  ivec4 in_sizes;
  // output dims
  ivec4 out_ndims;
  // x = output channels aligned to 4, y = input channels aligned to 4
  ivec2 channel_info;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;
layout(constant_id = 3) const int packed_dim = C_DIM;

void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, out_limits.xyz))) {
    return;
  }

  VEC4_T outval = VEC4_T(0.0);

  // scale up output position's packed dim
  pos[packed_dim] <<= 2;

  // index of packed dim in bchw format
  const int in_packed_dim_bchw_index = 3 - packed_dim;

  // determine input position based on output position and permute map
  // out_ndims is in BCHW format
  ivec4 in_bchw_pos = ivec4(0); // holds b,c,h,w
  in_bchw_pos[out_ndims[0]] = (pos.z / channel_info.x);
  in_bchw_pos[out_ndims[1]] = (pos.z % channel_info.x);
  in_bchw_pos[out_ndims[2]] = pos.y;
  in_bchw_pos[out_ndims[3]] = pos.x;

  for (int j = 0; j < 4; ++j) {
    // terminate the loop if trying to access input texture out of bounds
    if (any(greaterThanEqual(in_bchw_pos.wzyx, in_sizes.xyzw))) {
      break;
    }
    ivec3 fetch_pos;

    fetch_pos.xy = in_bchw_pos.wz;
    // calculate input position in z axis using batch and channel index which is in_bchw_pos.x and in_bchw_pos.y respectively
    fetch_pos.z = in_bchw_pos.y + in_bchw_pos.x * channel_info.y;

    // input tensor's packed dim lane corresponding to output tensor's pos
    const int in_packed_dim_lane_index = fetch_pos[packed_dim] & 0x3;

    // scale down input tensor's packed dim pos to perform fetch
    fetch_pos[packed_dim] >>= 2;

    // fetch input texel
    VEC4_T inval = VEC4_T(texelFetch(image_in, fetch_pos, 0));
    outval[j] = inval[in_packed_dim_lane_index];

    // go to next position in the input, that is mapped to the packed dim in the output
    in_bchw_pos[out_ndims[in_packed_dim_bchw_index]]++;
  }

  pos[packed_dim] = int(gl_GlobalInvocationID[packed_dim]);

  imageStore(image_out, pos, outval);
}
