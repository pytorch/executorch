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

layout(set = 0, binding = 0, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly ${IMAGE_T[NDIM][DTYPE]} image_out;
layout(set = 0, binding = 1) uniform PRECISION sampler3D image_in;
layout(set = 0, binding = 2) uniform PRECISION sampler3D weight_in;
layout(set = 0, binding = 3) uniform PRECISION sampler3D bias_in;
layout(set = 0, binding = 4) uniform PRECISION sampler3D mean_in;
layout(set = 0, binding = 5) uniform PRECISION sampler3D var_in;

layout(set = 0, binding = 6) uniform PRECISION restrict OutLimits {
  ivec3 out_limits;
};

layout(set = 0, binding = 7) uniform PRECISION restrict Params {
  float eps;
};

layout(set = 0, binding = 8) uniform PRECISION restrict Params2 {
  int num_texel_per_batch;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(pos, out_limits))) {
    return;
  }

  VEC4_T v = VEC4_T(texelFetch(image_in, pos, 0));

  ivec3 param_pos = ivec3(pos.z % num_texel_per_batch, 0, 0);

  VEC4_T weight = VEC4_T(texelFetch(weight_in, param_pos, 0));
  VEC4_T bias = VEC4_T(texelFetch(bias_in, param_pos, 0));
  VEC4_T mean = VEC4_T(texelFetch(mean_in, param_pos, 0));
  VEC4_T var = VEC4_T(texelFetch(var_in, param_pos, 0));

  v = ((v - mean) / sqrt(var + eps)) * weight + bias;

  imageStore(image_out, pos, v);
}
