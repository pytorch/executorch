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

#include "indexing_utils.h"

layout(std430) buffer;

layout(set = 0, binding = 0, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly ${IMAGE_T[NDIM][DTYPE]} image_out;
layout(set = 0, binding = 1) uniform PRECISION sampler3D image_in;
layout(set = 0, binding = 2) uniform PRECISION sampler3D kernel_in;
layout(set = 0, binding = 3) uniform PRECISION sampler3D bias_in;

layout(set = 0, binding = 4) uniform PRECISION restrict Out_channels {
  int data;
}
out_channels;

layout(set = 0, binding = 5) uniform PRECISION restrict In_length {
  int data;
}
in_length;

layout(set = 0, binding = 6) uniform PRECISION restrict Kernel_size {
  int data;
}
kernel_size;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * This implementation optimize for simplicity (and partially performance) for a
 * (1, C, L) where C == groups. Hence we only focus on calculating the rolling
 * kernel of the L dimension.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  // The global workgroup should have taken care of it. We only perform one
  // work item for each 1d tensor on lengths
  if (pos.x >= 1) {
    return;
  }

  int c = pos.y;
  if (c >= out_channels.data) {
    return;
  }

  // Assume n = 1, do not handle n > 1 case for now.
  int n = pos.z;
  if (n >= 1) {
    return;
  }

  vec4 bias = texelFetch(bias_in, ivec3(c, 0, 0), 0);

  for (int i = 0; i < in_length.data - kernel_size.data + 1; ++i) {
    vec4 v = vec4(0);
    for (int k = 0; k < kernel_size.data; ++k) {
      const ivec3 in_pos = ivec3(i+k, c, 0);
      const vec4 input_value = texelFetch(image_in, in_pos, 0);

      // Note that we are reading weight in the inner loop, this could be
      // improved by moving it before the outer loop. Since the weight vector is
      // contant for the entire call.

      // weight in input-space: (c, 0, k);
      // notice that c is 4-packed. We need to mod 4 to get the actual weight.
      const ivec3 w_pos = ivec3(k, 0, c / 4);
      const vec4 weight = texelFetch(kernel_in, w_pos, 0);

      float w = weight.x;
      if (c % 4 == 1) {
        w = weight.y;
      } else if (c % 4 == 2) {
        w = weight.z;
      } else if (c % 4 == 3) {
        w = weight.w;
      }

      v += w * input_value.x;
    }

    ivec3 out_pos = ivec3(i, c, 0);
    imageStore(image_out, out_pos, vec4(v.x + bias.x, 0, 0, 0));
  }
}
