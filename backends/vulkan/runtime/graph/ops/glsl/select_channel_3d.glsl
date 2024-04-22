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
#define T ${texel_component_type(DTYPE)}

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

// index to select
layout(set = 0, binding = 4) uniform PRECISION restrict IndexVal {
  int index;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, out_limits))) {
    return;
  }

  const int tex = index / 4;
  const int ind = index % 4;
  const T v = VEC4_T(texelFetch(image_in, ivec3(pos.x, pos.y, tex), 0))[ind];

  imageStore(image_out, ivec3(pos.x, pos.y, 0), VEC4_T(v, 0, 0, 0));
}
