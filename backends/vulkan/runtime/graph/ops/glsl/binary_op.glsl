/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#include "broadcasting_utils.h"
#include "indexing_utils.h"

#define PRECISION ${PRECISION}

#define OP(X, Y, A) ${OPERATOR}

layout(std430) buffer;

layout(set = 0, binding = 0, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly ${IMAGE_T[NDIM][DTYPE]} image_out;
layout(set = 0, binding = 1) uniform PRECISION sampler3D image_in;
layout(set = 0, binding = 2) uniform PRECISION sampler3D image_other;

layout(set = 0, binding = 3) uniform PRECISION restrict OutSizes {
  ivec4 data;
}
out_sizes;

layout(set = 0, binding = 4) uniform PRECISION restrict InSizes {
  ivec4 data;
}
in_sizes;

layout(set = 0, binding = 5) uniform PRECISION restrict OtherSizes {
  ivec4 data;
}
other_sizes;

layout(set = 0, binding = 6) uniform PRECISION restrict Alpha {
  float data;
}
alpha;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const ivec4 coord = POS_TO_COORD_${PACKING}(pos, out_sizes.data);

  if (any(greaterThanEqual(coord, out_sizes.data))) {
    return;
  }

  ivec4 in_coord = out_coord_to_in_coord(coord, in_sizes.data);
  vec4 in_texel = texelFetch(
    image_in,
    COORD_TO_POS_${PACKING}(in_coord, in_sizes.data),
    0);

  ivec4 other_coord = out_coord_to_in_coord(coord, other_sizes.data);
  vec4 other_texel = texelFetch(
    image_other,
    COORD_TO_POS_${PACKING}(other_coord, other_sizes.data),
    0);

  // Detect broadcasting
  if (PACKED_DIM_${PACKING}(other_sizes.data) < PACKED_DIM_${PACKING}(in_sizes.data)) {
    other_texel = other_texel.xxxx;
  }

  imageStore(image_out, pos, OP(in_texel, other_texel, alpha.data));
}
