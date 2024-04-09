/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#include "indexing_utils.h"

layout(std430) buffer;

layout(set = 0, binding = 0, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly ${IMAGE_T[NDIM][DTYPE]} image_out;
layout(set = 0, binding = 1) uniform PRECISION sampler3D image_in;
layout(set = 0, binding = 2) uniform PRECISION sampler2D kernel_in;
layout(set = 0, binding = 3) uniform PRECISION sampler2D bias_in;

layout(set = 0, binding = 4) uniform PRECISION restrict OutExtents {
  uvec4 data;
}
out_extents;

layout(set = 0, binding = 5) uniform PRECISION restrict InExtents {
  uvec4 data;
}
in_extents;

layout(set = 0, binding = 6) uniform PRECISION restrict Params {
  ivec2 kernel_size;
  ivec2 stride;
  ivec2 padding;
  ivec2 dilation;
}
params;

// If fields are separated, SwiftShader cannot identify in_group_size.
layout(set = 0, binding = 7) uniform PRECISION restrict ExtraParams {
  ivec2 overlay_region;
  int in_group_size;
}
extra_params;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Computes a depthwise convolution. Each shader invocation calculates the
 * output at a single output location.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, out_extents.data.xyz))) {
    return;
  }

  // Compute the index of the top-left element of the overlay region. Negative
  // indices indicate that the top-left element is in a region added by padding.
  const ivec2 ipos = pos.xy * params.stride - params.padding;

  // Compute the start and end of the input indices to load. Padding is assumed
  // to be constant 0 padding, so any reads from the padding region is skipped.
  const ivec2 start = ipos;
  const ivec2 end = ipos + extra_params.overlay_region.xy;

  ${VEC4_T[DTYPE]} sum = texelFetch(bias_in, ivec2(pos.z, 0), 0);
  int kx = 0;
  for (int y = start.y, i = 0; i < ${TILE_SIZE}; y += params.dilation.y, i++) {
    for (int x = start.x, j = 0; j < ${TILE_SIZE}; x += params.dilation.x, j++) {
      // The weight kernel was rearranged such that every NxN filter is
      // flattened to fit in one row. Each filter was then stacked on top of
      // each other vertically.
      const vec4 in_texel = texelFetch(image_in, ivec3(x, y, pos.z), 0);
      sum = fma(in_texel, texelFetch(kernel_in, ivec2(kx, pos.z), 0), sum);
      kx++;
    }
  }

  imageStore(image_out, pos, sum);
}
