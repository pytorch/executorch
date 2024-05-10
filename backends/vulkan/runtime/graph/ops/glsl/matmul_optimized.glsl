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
#include "matmul.h"

$if IS_ADDMM:
  // addmm will have additional arguments compared to regular mm
  layout(set = 0, binding = 0, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly image3D im_out;
  layout(set = 0, binding = 1) uniform PRECISION ${SAMPLER_T[NDIM][DTYPE]} im_mat1;
  layout(set = 0, binding = 2) uniform PRECISION ${SAMPLER_T[NDIM][DTYPE]} im_mat2;
  layout(set = 0, binding = 3) uniform PRECISION ${SAMPLER_T[NDIM][DTYPE]} im_self;

  layout(set = 0, binding = 4) uniform PRECISION restrict OutLimits {
    ivec3 out_limits;
  };

  layout(set = 0, binding = 5) uniform PRECISION restrict StepSize {
    int step_size;
  };

  layout(set = 0, binding = 6) uniform PRECISION restrict Reminder {
    int reminder;
  };

  layout(set = 0, binding = 7) uniform PRECISION restrict BatchSize {
    int batch_size;
  };

  layout(set = 0, binding = 8) uniform PRECISION restrict AddmmParams {
    int broadcast_at_width;
    int broadcast_at_height;
    float alpha;
    float beta;
  };
$else:
  // define original matmul_optimized arguments
  layout(set = 0, binding = 0, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly image3D im_out;
  layout(set = 0, binding = 1) uniform PRECISION ${SAMPLER_T[NDIM][DTYPE]} im_mat1;
  layout(set = 0, binding = 2) uniform PRECISION ${SAMPLER_T[NDIM][DTYPE]} im_mat2;

  layout(set = 0, binding = 3) uniform PRECISION restrict OutLimits {
    ivec3 out_limits;
  };

  layout(set = 0, binding = 4) uniform PRECISION restrict StepSize {
    int step_size;
  };

  layout(set = 0, binding = 5) uniform PRECISION restrict Reminder {
    int reminder;
  };

  layout(set = 0, binding = 6) uniform PRECISION restrict BatchSize {
    int batch_size;
  };

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, out_limits))) {
    return;
  }

  FloatMatrix results = matmul_partial_4x4(im_mat1, im_mat2, pos, batch_size, step_size, reminder);

  for (int idx_c = 0; idx_c < FOUR; idx_c++) {
    for (int idx_r = 0; idx_r < FOUR; idx_r++) {
      const ivec3 out_pos =
          ivec3(idx_r + FOUR * pos.x, idx_c + FOUR * pos.y, pos.z);
      $if IS_ADDMM:
        vec4 self_texel = get_texel_C_packed(im_self, out_pos, broadcast_at_width, broadcast_at_height);
        results.data[idx_c][idx_r][0] = beta * self_texel.x + alpha * results.data[idx_c][idx_r][0];

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
