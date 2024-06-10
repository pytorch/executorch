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

VEC4_T q_8w_linear(const ivec4 out_pos, const int K) {
  const VEC4_T scales = load_texel(t_scales, out_pos.x);

  VEC4_T outtex = VEC4_T(0);

  // Initial mat1 pos will be (0, out_pos.y, out_pos.z, 0)
  int mat1_tid = out_pos.y * mat1_strides.y + out_pos.z * qmat2_strides.z;
  // Initial qmat2 pos wil be (0, out_pos.x * 4, 0, 0)
  int qmat2_tid = out_pos.x * 4 * qmat2_strides.y;

  // TODO(ssjia): optimize memory access pattern by traversing K in inner loop
  for (int i = 0; i < K; i += 4) {
    const VEC4_T mat1_tex = load_texel(t_mat1, mat1_tid);

    const VEC4_T sums = VEC4_T(
        dot(mat1_tex, load_texel(t_qmat2, qmat2_tid) * scales.x),
        dot(mat1_tex,
            load_texel(t_qmat2, qmat2_tid + qmat2_strides.y) * scales.y),
        dot(mat1_tex,
            load_texel(t_qmat2, qmat2_tid + qmat2_strides.y * 2) * scales.z),
        dot(mat1_tex,
            load_texel(t_qmat2, qmat2_tid + qmat2_strides.y * 3) * scales.w));

    outtex += sums;

    mat1_tid++;
    qmat2_tid++;
  }

  return outtex;
}

#else // USING_TEXTURE

VEC4_T q_8w_linear(const ivec3 out_pos, const int K) {
  ivec3 mat1_pos = ivec3(0, out_pos.yz);
  ivec3 qmat2_pos = ivec3(0, out_pos.x * 4, 0);

  VEC4_T outtex = VEC4_T(0);

  const ivec3 scales_pos = ivec3(out_pos.x, 0, 0);
  const VEC4_T scales = load_texel(t_scales, scales_pos);

  for (int i = 0; i < K; i += 4) {
    const VEC4_T mat1_tex = load_texel(t_mat1, mat1_pos);

    const VEC4_T sums = VEC4_T(
        dot(mat1_tex, load_texel(t_qmat2, qmat2_pos) * scales.x),
        dot(mat1_tex,
            load_texel(t_qmat2, qmat2_pos + ivec3(0, 1, 0)) * scales.y),
        dot(mat1_tex,
            load_texel(t_qmat2, qmat2_pos + ivec3(0, 2, 0)) * scales.z),
        dot(mat1_tex,
            load_texel(t_qmat2, qmat2_pos + ivec3(0, 3, 0)) * scales.w));

    outtex += sums;

    mat1_pos.x++;
    qmat2_pos.x++;
  }

  return outtex;
}

#endif // USING_BUFFER

#endif // Q_LINEAR_H
