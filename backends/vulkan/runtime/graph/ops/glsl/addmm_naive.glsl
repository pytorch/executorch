/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

$if MAT2_IS_TRANSPOSED:
  #define MAT2_IS_TRANSPOSED

#include "indexing_utils.h"
#include "matmul.h"

layout(set = 0, binding = 0, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly image3D im_out;
layout(set = 0, binding = 1) uniform PRECISION ${SAMPLER_T[NDIM][DTYPE]} im_mat1;
layout(set = 0, binding = 2) uniform PRECISION ${SAMPLER_T[NDIM][DTYPE]} im_mat2;
layout(set = 0, binding = 3) uniform PRECISION ${SAMPLER_T[NDIM][DTYPE]} im_self;

layout(set = 0, binding = 4) uniform PRECISION restrict OutLimits {
  ivec3 out_limits;
};

layout(set = 0, binding = 5) uniform PRECISION restrict InSizes {
  ivec4 in_sizes;
};

layout(set = 0, binding = 6) uniform PRECISION restrict SelfSizes {
  ivec3 self_sizes;
};

layout(set = 0, binding = 7) uniform PRECISION restrict AddmmParams {
  float alpha;
  float beta;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, out_limits))) {
    return;
  }

  vec4 texel = vec4(0);

  $if MAT1_PACKING == "W_packed":
    $if MAT2_PACKING == "H_packed":
      ivec3 mat2_pos = ivec3(pos.x * 4, 0, pos.z);
      texel = matmul_naive_W_packed_H_packed(
          im_mat1,
          im_mat2,
          pos,
          in_sizes[0]);
    $elif MAT2_PACKING == "W_packed":
      texel = matmul_naive_W_packed_W_packed(
          im_mat1,
          im_mat2,
          pos,
          in_sizes[0]);
    $else:
      $raise Exception("Unsupported value for MAT2_PACKING")
  $else:
    $raise Exception("Unsupported value combo for MAT1_PACKING and MAT2_PACKING")

  vec4 self_texel = get_texel_W_packed(
      im_self,
      pos,
      self_sizes.x == 1,
      self_sizes.y == 1);

  texel = beta * self_texel + alpha * texel;
  imageStore(im_out, pos, texel);
}
