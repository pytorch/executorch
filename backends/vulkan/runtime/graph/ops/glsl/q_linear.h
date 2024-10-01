/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef Q_LINEAR_H
#define Q_LINEAR_H

#include "indexing_utils.h"

// The functions in this file assume that some variables have been defined as
// descriptors, such as t_mat1, t_qmat2, t_scales, etc.

#ifdef USING_BUFFER

#ifndef FLOAT_T
#define FLOAT_T float
#endif

FLOAT_T q_8w_linear(const ivec4 out_idx, const int K) {
  const FLOAT_T scale = t_scales[out_idx.x];

  FLOAT_T outval = FLOAT_T(0.0);

  // Initial mat1 tensor idx will be (0, out_idx.y, out_idx.z, 0)
  int mat1_offset = out_idx.y * mat1_strides.y + out_idx.z * qmat2_strides.z;
  // Initial qmat2 tensor idx wil be (0, out_idx.x, 0, 0); note that the qmat2
  // tensor is transposed
  int qmat2_offset = out_idx.x * qmat2_strides.y;

  // TODO(ssjia): optimize memory access pattern by traversing K in inner loop
  for (int i = 0; i < K; i++) {
    const FLOAT_T mat1_val = t_mat1[mat1_offset];
    const FLOAT_T mat2_val = t_qmat2[qmat2_offset] * scale;

    outval += mat1_val * mat2_val;

    mat1_offset++;
    qmat2_offset++;
  }

  return outval;
}

#else // USING_TEXTURE

vec4 q_8w_linear(const ivec3 out_lpos) {
  ivec3 mat1_pos = lpos_to_pos(ivec3(0, out_lpos.yz), mat1_axis_map);

  vec4 texel = vec4(0);
  const int K = divup4(mat1_sizes.x);

  // Scales is a 1D texture so we don't need to apply axis mapping
  const ivec3 scales_pos = ivec3(out_lpos.x, 0, 0);
  const vec4 scales = load_texel(t_scales, scales_pos);

  for (int i = 0; i < K; ++i) {
    const vec4 mat1_tex = texelFetch(t_mat1, mat1_pos, 0);

    vec4 sums;
    for (int r = 0; r < 4; ++r) {
      ivec3 qmat2_pos =
          lpos_to_pos(ivec3(i, out_lpos.x * 4 + r, 0), qmat2_axis_map);

      sums[r] = dot(mat1_tex, texelFetch(t_qmat2, qmat2_pos, 0) * scales[r]);
    }

    texel += sums;
    mat1_pos[mat1_axis_map.x]++;
  }

  return texel;
}

#endif // USING_BUFFER

#endif // Q_LINEAR_H
