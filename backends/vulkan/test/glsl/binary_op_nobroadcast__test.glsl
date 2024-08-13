/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define op(X, Y) ${OPERATOR}

layout(std430) buffer;

// clang-format off
layout(set = 0, binding = 0, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly image3D image_out;
// clang-format on
layout(set = 0, binding = 1) uniform PRECISION sampler3D image_in;
layout(set = 0, binding = 2) uniform PRECISION sampler3D image_other;

layout(set = 0, binding = 3) uniform PRECISION restrict OutExtents {
  uvec4 data;
}
out_extents;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, out_extents.data.xyz))) {
    return;
  }

  vec4 in_texel = texelFetch(image_in, pos, 0);
  vec4 other_texel = texelFetch(image_other, pos, 0);

  imageStore(image_out, pos, op(in_texel, other_texel));
}
