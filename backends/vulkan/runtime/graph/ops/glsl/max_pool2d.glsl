/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}
#define FLT_MIN -3.402823466e+38

#include "indexing_utils.h"

layout(std430) buffer;

layout(set = 0, binding = 0, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly ${IMAGE_T[NDIM][DTYPE]} image_out;
layout(set = 0, binding = 1, ${IMAGE_FORMAT["int"]}) uniform PRECISION restrict writeonly ${IMAGE_T[NDIM]["int"]} image_idx;
layout(set = 0, binding = 2) uniform PRECISION sampler3D image_in;

layout(set = 0, binding = 3) uniform PRECISION restrict OutLimits {
  ivec3 out_limits;
};

layout(set = 0, binding = 4) uniform PRECISION restrict InSizes {
  ivec4 in_sizes;
};

layout(set = 0, binding = 5) uniform PRECISION restrict Params {
  ivec2 kernel_size;
  ivec2 stride;
  ivec2 padding;
  ivec2 dilation;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, out_limits))) {
    return;
  }

  const ivec2 ipos = pos.xy * stride - padding;

  const ivec2 start = ipos;
  const ivec2 end = ipos + kernel_size * dilation;

  vec4 out_texel = vec4(FLT_MIN);
  ivec4 idx_texel = ivec4(0);

  for (int y = start.y; y < end.y; y += dilation.y) {
    for (int x = start.x; x < end.x; x += dilation.x) {
      if ((x >= 0 && x < in_sizes.x) && (y >= 0 && y < in_sizes.y)) {
        const vec4 cur_texel = texelFetch(image_in, ivec3(x, y, pos.z), 0);

        // Set idx if value is greatest in the pool; else, keep the existing idx.
        ivec4 cur_idx = ivec4(x + int(in_sizes.x) * y);
        ivec4 mask = ivec4(greaterThan(cur_texel, out_texel));
        idx_texel = ivec4(mix(idx_texel, cur_idx, mask));

        out_texel = max(cur_texel, out_texel);
      }
    }
  }

  imageStore(image_out, pos, out_texel);
  imageStore(image_idx, pos, idx_texel);
}
