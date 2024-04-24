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

layout(set = 0, binding = 3) uniform PRECISION restrict InLimits {
  ivec3 in_limits;
};



layout(set = 0, binding = 4) uniform PRECISION restrict CopyArgs {
  ivec3 range;
  int unused0;
  ivec3 src_offset;
  int unused1;
  ivec3 dst_offset;
  int unused2;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  const ivec3 out_pos = pos + dst_offset;
  const ivec3 in_pos = pos + src_offset;

  if (any(greaterThanEqual(pos, range))) {
    return;
  }

  imageStore(image_out, out_pos, texelFetch(image_in, in_pos, 0));
}
