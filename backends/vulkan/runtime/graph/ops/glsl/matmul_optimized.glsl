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

layout(set = 0, binding = 3) uniform PRECISION restrict OutLimits {
  ivec3 out_limits;
};

layout(set = 0, binding = 4) uniform PRECISION restrict OutSizes {
  ivec4 out_sizes;
};

layout(set = 0, binding = 5) uniform PRECISION restrict InLimits {
  ivec3 in_limits;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, out_limits))) {
    return;
  }

  FloatMatrix results = matmul_partial_4x4(
      im_mat1,
      im_mat2,
      pos,
      out_sizes[2],
      in_limits[0]);

  for (int idx_c = 0; idx_c < FOUR; idx_c++) {
    for (int idx_r = 0; idx_r < FOUR; idx_r++) {
      const ivec3 out_pos =
          ivec3(idx_r + FOUR * pos.x, idx_c + FOUR * pos.y, pos.z);

      // results is in transposed order w.r.t. the desired output
      imageStore(
          im_out,
          out_pos,
          vec4(
              results.data[idx_c][idx_r][0],
              results.data[idx_c][idx_r][1],
              results.data[idx_c][idx_r][2],
              results.data[idx_c][idx_r][3]));
    }
  }
}
