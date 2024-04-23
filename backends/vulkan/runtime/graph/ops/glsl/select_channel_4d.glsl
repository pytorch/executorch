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
layout(set = 0, binding = 1) uniform PRECISION sampler3D image_in;

layout(set = 0, binding = 2) uniform PRECISION restrict OutLimits {
  ivec3 out_limits;
};

layout(set = 0, binding = 3) uniform PRECISION restrict Sizes {
  ivec4 sizes;
};

layout(set = 0, binding = 4) uniform PRECISION restrict SelectVal {
  // data.x: index along channel dim to select
  // data.y: number of batches
  // data.z: number of texels per batch
  // data.w: unused
  ivec4 select_info;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, out_limits))) {
    return;
  }

  const int num_batches = select_info.y;
  const int num_texel_per_batch = select_info.z;
  const int index = select_info.x;

  // read in the same channel from 4 separate batches
  VEC4_T out_texel = VEC4_T(0, 0, 0, 0);
  for (int k = 0; k < 4; k++) {
    if ((k + pos.z * 4) >=
        num_batches) {
      break;
    }
    const uint src_pos_z = (4 * num_texel_per_batch * pos.z) +
        (k * num_texel_per_batch) + (index / 4);
    const uint src_pos_t = index % 4;
    out_texel[k] =
        VEC4_T(texelFetch(image_in, ivec3(pos.x, pos.y, src_pos_z), 0))[src_pos_t];
  }

  imageStore(image_out, pos, out_texel);
}
