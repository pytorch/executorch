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

layout(set = 0, binding = 2) uniform PRECISION restrict OutSizes {
  uvec4 data;
}
out_sizes;

// index to select
layout(set = 0, binding = 3) uniform PRECISION restrict IndexVal {
  int data;
}
index;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  const ivec4 idx = to_tensor_idx_C_packed(pos, out_sizes.data);

  if (any(greaterThanEqual(idx, out_sizes.data))) {
    return;
  }

  // w
  const int src_x = pos.x;
  // h
  const int src_y = index.data;
  // c
  const int src_z = pos.y;

  const VEC4_T v = VEC4_T(texelFetch(image_in, ivec3(src_x, src_y, src_z), 0));

  for (int i = 0; i < 4; i++) {
    ivec3 new_pos = ivec3(pos.x, pos.y * 4 + i, 0);

    // When the C-channel exceeds original block size, exit early
    if (new_pos.y >= out_sizes.data.y) {
      return;
    }

    imageStore(image_out, new_pos, VEC4_T(v[i], 0, 0, 0));
  }
}
