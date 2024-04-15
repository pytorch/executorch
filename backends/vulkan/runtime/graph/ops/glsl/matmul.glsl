/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#include "indexing_utils.h"

#define PRECISION ${PRECISION}

layout(set = 0, binding = 0, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly image3D im_out;
layout(set = 0, binding = 1) uniform PRECISION ${SAMPLER_T[NDIM][DTYPE]} im_mat1;
layout(set = 0, binding = 2) uniform PRECISION ${SAMPLER_T[NDIM][DTYPE]} im_mat2;

layout(set = 0, binding = 3) uniform PRECISION restrict OutExtents {
  uvec4 data;
}
out_extents;

layout(set = 0, binding = 4) uniform PRECISION restrict InSizes {
  ivec4 data;
}
in_sizes;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, out_extents.data.xyz))) {
    return;
  }

  vec4 texel = vec4(0);

  ivec3 mat1_pos = ivec3(0, pos.y, pos.z);

  $if MAT2_PACKING == "H_packed":
    ivec3 mat2_pos = ivec3(pos.x * 4, 0, pos.z);
  $else:
    ivec3 mat2_pos = ivec3(pos.x, 0, pos.z);

  $if MAT1_PACKING == "W_packed":
    int K = divup4(in_sizes.data[0]);
    for (int i = 0; i < K; ++i) {
      $if MAT2_PACKING == "H_packed":
        vec4 mat1_tex = texelFetch(im_mat1, mat1_pos, 0);
        vec4 sums = vec4(
            dot(mat1_tex, texelFetch(im_mat2, mat2_pos, 0)),
            dot(mat1_tex, texelFetch(im_mat2, mat2_pos + ivec3(1, 0, 0), 0)),
            dot(mat1_tex, texelFetch(im_mat2, mat2_pos + ivec3(2, 0, 0), 0)),
            dot(mat1_tex, texelFetch(im_mat2, mat2_pos + ivec3(3, 0, 0), 0)));

        texel += sums;

        mat1_pos.x++;
        mat2_pos.y++;
      $elif MAT2_PACKING == "W_packed":
        vec4 mat1_tex = texelFetch(im_mat1, mat1_pos, 0);
        texel = fma(mat1_tex.xxxx, texelFetch(im_mat2, mat2_pos, 0), texel);
        mat2_pos.y++;
        texel = fma(mat1_tex.yyyy, texelFetch(im_mat2, mat2_pos, 0), texel);
        mat2_pos.y++;
        texel = fma(mat1_tex.zzzz, texelFetch(im_mat2, mat2_pos, 0), texel);
        mat2_pos.y++;
        texel = fma(mat1_tex.wwww, texelFetch(im_mat2, mat2_pos, 0), texel);
        mat2_pos.y++;

        mat1_pos.x++;
      $else:
        $raise Exception("Unsupported value for MAT2_PACKING")
    }
  $elif MAT1_PACKING == "C_packed" and MAT2_PACKING == "C_packed":
    int K = in_sizes.data[0];
    for (int i = 0; i < K; ++i) {
      texel = fma(
          texelFetch(im_mat1, mat1_pos, 0),
          texelFetch(im_mat2, mat2_pos, 0),
          texel);

      mat1_pos.x++;
      mat2_pos.y++;
    }
  $else:
    $raise Exception("Unsupported value combo for MAT1_PACKING and MAT2_PACKING")

  imageStore(im_out, pos, texel);
}
