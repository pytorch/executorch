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

layout(std430) buffer;

layout(set = 0, binding = 0, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly ${IMAGE_T[NDIM][DTYPE]} image_out;
layout(set = 0, binding = 1, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly ${IMAGE_T[NDIM][DTYPE]} image_mean;
layout(set = 0, binding = 2, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly ${IMAGE_T[NDIM][DTYPE]} image_rstd;

layout(set = 0, binding = 3) uniform PRECISION sampler3D image_in;
layout(set = 0, binding = 4) uniform PRECISION sampler3D weight_in;
layout(set = 0, binding = 5) uniform PRECISION sampler3D bias_in;

layout(set = 0, binding = 6) uniform PRECISION restrict OutExtents {
  ivec4 data;
}
out_sizes;

layout(set = 0, binding = 7) uniform PRECISION restrict Epsilon {
  float data;
}
epsilon;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const ivec4 coord = POS_TO_COORD_${PACKING}(pos, out_sizes.data);

  if (any(greaterThanEqual(coord, out_sizes.data))) {
    return;
  }

  const int width = out_sizes.data.x;

  vec4 mean = vec4(0);
  vec4 delta = vec4(0);
  vec4 delta2 = vec4(0);
  vec4 M2 = vec4(0);

  // Use Welford's online algorithm to compute mean and variance in one pass
  // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
  for (int w = 0; w < width; ++w) {
    vec4 v = texelFetch(image_in, ivec3(w, pos.y, pos.z), 0);
    delta = v - mean;
    mean += delta / (w + 1);
    delta2 = v - mean;
    M2 += delta * delta2;
  }

  vec4 var = M2 / width;
  vec4 rstd = pow(var + epsilon.data, vec4(-0.5));
  vec4 offset = -rstd * mean;

  for (int w = 0; w < width; ++w) {
    vec4 v = texelFetch(image_in, ivec3(w, pos.y, pos.z), 0);
    // broadcasting
    vec4 weight = texelFetch(weight_in, ivec3(w, 0, 0), 0).xxxx;
    vec4 bias = texelFetch(bias_in, ivec3(w, 0, 0), 0).xxxx;
    vec4 ot = (v * rstd + offset) * weight + bias;
    imageStore(image_out, ivec3(w, pos.y, pos.z), ot);
  }

  imageStore(image_mean, pos, mean);
  imageStore(image_rstd, pos, rstd);
}
