/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION highp

#include "indexing_utils.h"

layout(std430) buffer;

layout(set = 0, binding = 0, rgba8) uniform PRECISION restrict writeonly image2D rgba_out;
layout(set = 0, binding = 1) uniform PRECISION sampler3D t_in;

layout(set = 0, binding = 2) uniform PRECISION restrict readonly limits_UBO {
  ivec3 limits;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, limits))) {
    return;
  }

  imageStore(rgba_out, pos.xy, texelFetch(t_in, pos, 0));
}
