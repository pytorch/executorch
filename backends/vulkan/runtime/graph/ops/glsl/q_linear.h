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

// To convince the SPIR-V compiler to unroll the loops optimally, need this
// macro
#define FOUR 4

#ifdef TILE_ROW_2
#define TILE_ROWS 2
#else
#define TILE_ROWS 4
#endif

struct FloatMatrix_2d {
  float data[TILE_ROWS][FOUR];
};

struct FloatMatrix_3d {
  float data[TILE_ROWS][FOUR][FOUR];
};

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

FloatMatrix_2d q_8w_linear_optimized_2d(const ivec3 out_pos, const int K) {
  FloatMatrix_2d results;

  VEC4_T im_mat1_partial_load[TILE_ROWS];
  ivec4 im_mat2_partial_load[FOUR];

  [[unroll]] for (int i = 0; i < TILE_ROWS; i++) {
    [[unroll]] for (int j = 0; j < FOUR; j++) { results.data[i][j] = 0.0f; }
  }

  for (int mat1_x = 0; mat1_x < K; mat1_x++) {
    [[unroll]] for (int offset = 0; offset < TILE_ROWS; offset++) {
      const int mat1_y = out_pos.y * TILE_ROWS + offset;
      const ivec3 mat1_pos = ivec3(mat1_x, mat1_y, 0);
      im_mat1_partial_load[offset] = load_texel(t_mat1, mat1_pos);
    }
    [[unroll]] for (int offset = 0; offset < FOUR; offset++) {
      const int mat2_y = (FOUR * out_pos.x) + offset;
      const ivec3 mat2_pos = ivec3(mat1_x, mat2_y, 0);
      im_mat2_partial_load[offset] = load_texel(t_qmat2, mat2_pos);
    }

    [[unroll]] for (int out_row = 0; out_row < TILE_ROWS; out_row++) {
      [[unroll]] for (int out_col = 0; out_col < FOUR; out_col++) {
        results.data[out_row][out_col] +=
            dot(im_mat1_partial_load[out_row], im_mat2_partial_load[out_col]);
      }
    }
  }

  const VEC4_T scales = load_texel(t_scales, ivec3(out_pos.x, 0, 0));
  [[unroll]] for (int i = 0; i < TILE_ROWS; i++) {
    [[unroll]] for (int j = 0; j < FOUR; j++) {
      results.data[i][j] *= scales[j];
    }
  }
  return results;
}

FloatMatrix_3d q_8w_linear_optimized_3d(
    const ivec3 out_pos,
    const int K,
    const int batch_size) {
  FloatMatrix_3d results;

  [[unroll]] for (int i = 0; i < TILE_ROWS; i++) {
    [[unroll]] for (int j = 0; j < FOUR; j++) {
      [[unroll]] for (int k = 0; k < FOUR; k++) {
        results.data[i][j][k] = 0.0f;
      }
    }
  }

  VEC4_T im_mat1_partial_load[TILE_ROWS];
  ivec4 im_mat2_partial_load[FOUR];

  const VEC4_T scales = load_texel(t_scales, ivec3(out_pos.x, 0, 0));

  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    if (FOUR * out_pos.z + batch_idx >= batch_size) {
      break;
    }
    int mat_z = FOUR * out_pos.z + batch_idx;
    for (int mat1_x = 0; mat1_x < K; mat1_x++) {
      [[unroll]] for (int offset = 0; offset < TILE_ROWS; offset++) {
        // read and cache 2x4 (or 4x4) tile of im_mat1
        const int mat1_y = (TILE_ROWS * out_pos.y) + offset;
        const ivec3 mat1_pos = ivec3(mat1_x, mat1_y, mat_z);
        im_mat1_partial_load[offset] = load_texel(t_mat1, mat1_pos);
      }

      [[unroll]] for (int offset = 0; offset < FOUR; offset++) {
        // read and cache 4x4 tile of im_mat2
        const int mat2_y = (FOUR * out_pos.x) + offset;
        const ivec3 mat2_pos = ivec3(mat1_x, mat2_y, 0);
        im_mat2_partial_load[offset] = load_texel(t_qmat2, mat2_pos);
      }

      [[unroll]] for (int out_row = 0; out_row < TILE_ROWS; out_row++) {
        [[unroll]] for (int out_col = 0; out_col < FOUR; out_col++) {
          results.data[out_row][out_col][batch_idx] +=
              dot(im_mat1_partial_load[out_row], im_mat2_partial_load[out_col]);
        }
      }
    }

    [[unroll]] for (int i = 0; i < TILE_ROWS; i++) {
      [[unroll]] for (int j = 0; j < FOUR; j++) {
        results.data[i][j][batch_idx] *= scales[j];
      }
    }
  }
  return results;
}

#endif // USING_BUFFER

#endif // Q_LINEAR_H
