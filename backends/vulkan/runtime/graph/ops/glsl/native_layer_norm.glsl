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

#define VEC4_T ${texel_type(DTYPE)}

layout(std430) buffer;

layout(set = 0, binding = 0, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly ${IMAGE_T[NDIM][DTYPE]} image_out;
layout(set = 0, binding = 1, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly ${IMAGE_T[NDIM][DTYPE]} image_mean;
layout(set = 0, binding = 2, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly ${IMAGE_T[NDIM][DTYPE]} image_rstd;

layout(set = 0, binding = 3) uniform PRECISION sampler3D image_in;
layout(set = 0, binding = 4) uniform PRECISION sampler3D weight_in;
layout(set = 0, binding = 5) uniform PRECISION sampler3D bias_in;

layout(set = 0, binding = 6) uniform PRECISION restrict OutLimits {
  ivec3 out_limits;
};

layout(set = 0, binding = 7) uniform PRECISION restrict Sizes {
  ivec4 sizes;
};

layout(set = 0, binding = 8) uniform PRECISION restrict Epsilon {
  float epsilon;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, out_limits))) {
    return;
  }

  const int width = int(sizes.x);

  VEC4_T mean = VEC4_T(0);
  VEC4_T delta = VEC4_T(0);
  VEC4_T delta2 = VEC4_T(0);
  VEC4_T M2 = VEC4_T(0);

  // Use Welford's online algorithm to compute mean and variance in one pass
  // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
  for (int w = 0; w < width; ++w) {
    VEC4_T v = texelFetch(image_in, ivec3(w, pos.y, pos.z), 0);
    delta = v - mean;
    mean += delta / (w + 1);
    delta2 = v - mean;
    M2 += delta * delta2;
  }

  VEC4_T var = M2 / width;
  VEC4_T rstd = pow(var + epsilon, VEC4_T(-0.5));
  VEC4_T offset = -rstd * mean;

  for (int w = 0; w < width; ++w) {
    VEC4_T v = texelFetch(image_in, ivec3(w, pos.y, pos.z), 0);
    // broadcasting
    VEC4_T weight = texelFetch(weight_in, ivec3(w, 0, 0), 0).xxxx;
    VEC4_T bias = texelFetch(bias_in, ivec3(w, 0, 0), 0).xxxx;
    VEC4_T outtex = (v * rstd + offset) * weight + bias;
    imageStore(image_out, ivec3(w, pos.y, pos.z), outtex);
  }

  imageStore(image_mean, pos, mean);
  imageStore(image_rstd, pos, rstd);
}
