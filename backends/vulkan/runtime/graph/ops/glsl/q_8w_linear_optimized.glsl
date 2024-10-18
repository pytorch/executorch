/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}
#define FLOAT_T ${buffer_scalar_type(DTYPE)}

${define_active_storage_type(STORAGE)}

${define_required_extensions(DTYPE)}
${define_required_extensions("int8")}


$if BATCH_MODE:
  #define BATCH_MODE

#define TILE_ROWS ${TILE_ROWS}
#define FOUR 4

// we avoid mat4 and vec4 usage here as they compile to much less efficient
// SPIR-V
struct FloatMatrix_2d {
  float data[TILE_ROWS][FOUR];
};

struct FloatMatrix_3d {
  float data[TILE_ROWS][FOUR][FOUR];
};

#ifdef BATCH_MODE
  #define FloatMatrix FloatMatrix_3d
#else
  #define FloatMatrix FloatMatrix_2d
#endif

#include "indexing_utils.h"

layout(std430) buffer;

${layout_declare_tensor(0, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(1, "r", "t_mat1", DTYPE, STORAGE)}
${layout_declare_tensor(2, "r", "t_qmat2", "int8", STORAGE)}
${layout_declare_tensor(3, "r", "t_scales", DTYPE, STORAGE)}

$if STORAGE == "buffer":
  ${layout_declare_ubo(4, "ivec4", "out_sizes")}
  ${layout_declare_ubo(5, "ivec4", "out_strides")}
  ${layout_declare_ubo(6, "int", "out_numel")}
  ${layout_declare_ubo(7, "ivec4", "mat1_sizes")}
  ${layout_declare_ubo(8, "ivec4", "mat1_strides")}
  ${layout_declare_ubo(9, "ivec4", "qmat2_strides")}
  ${layout_declare_ubo(10, "ivec4", "scales_strides")}
$else:
  ${layout_declare_ubo(4, "ivec3", "out_limits")}
  ${layout_declare_ubo(5, "ivec4", "mat1_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// This header file must be defined after the layout descriptors have been
// declared because the functions in the header assume some variables have been
// declared as layout descriptors.

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

void main() {
  const int out_bufi = int(gl_GlobalInvocationID.x);
  if (out_bufi >= out_numel) {
    return;
  }

  const ivec4 out_tidx = bufi_to_tidx(out_bufi, out_strides, 0);

  t_out[out_bufi] = q_8w_linear(out_tidx, mat1_sizes.x);
}

#else // USING_TEXTURE
FloatMatrix q_8w_linear_optimized(const ivec3 out_idx_tl) {
  FloatMatrix results;
  for (int i = 0; i < TILE_ROWS; i++) {
    for (int j = 0; j < FOUR; j++) {
#ifdef BATCH_MODE
      for (int k = 0; k < FOUR; k++) {
        results.data[i][j][k] = 0.0f;
      }
#else
      results.data[i][j] = 0.0f;
#endif // BATCH_MODE
    }
  }

  VEC4_T im_mat1_partial_load[TILE_ROWS];
  VEC4_T im_mat2_partial_load[FOUR];

#ifdef BATCH_MODE
  for (int batch_idx = 0; batch_idx < FOUR; batch_idx++) {
    if (out_idx_tl.z + batch_idx >= out_limits.z) {
      break;
    }
#endif
    for (int k = 0; k < mat1_sizes.x; k++) {
      for (int r = 0; r < TILE_ROWS; r++) {
        ivec3 mat1_pos = ivec3(k, out_idx_tl.y * TILE_ROWS + r, 0);
#ifdef BATCH_MODE
        mat1_pos[2] = out_idx_tl.z + batch_idx;
#endif

        im_mat1_partial_load[r] = texelFetch(t_mat1, mat1_pos, 0);
      }

      for (int r = 0; r < FOUR; ++r) {
        ivec3 qmat2_pos = ivec3(k, FOUR * out_idx_tl.x + r, 0);

        im_mat2_partial_load[r] = texelFetch(t_qmat2, qmat2_pos, 0);
      }

      vec4 scales = texelFetch(t_scales, ivec3(out_idx_tl.x, 0, 0), 0);

      // perform partial dot products and add partial result to results
      for (int out_row = 0; out_row < TILE_ROWS; out_row++) {
        for (int out_col = 0; out_col < FOUR; out_col++) {
#ifdef BATCH_MODE
          results.data[out_row][out_col][batch_idx] +=
#else
        results.data[out_row][out_col] +=
#endif
              dot(im_mat1_partial_load[out_row],
                  im_mat2_partial_load[out_col] * scales[out_col]);
        }
      }
    }
#ifdef BATCH_MODE
  }
#endif
  return results;
}

void main() {
  const ivec3 out_idx = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(out_idx, out_limits))) {
    return;
  }

  FloatMatrix results = q_8w_linear_optimized(out_idx);

  ivec3 out_pos = ivec3(
      out_idx.x,
      out_idx.y * TILE_ROWS,
#ifdef BATCH_MODE
      out_idx.z * 4
#else
      out_idx.z
#endif
);

  for (int idx_c = 0; idx_c < TILE_ROWS; idx_c++, out_pos[1]++) {
    out_pos.x = out_idx.x;
    $if BATCH_MODE:
      for (int idx_r = 0; idx_r < FOUR; idx_r++, out_pos[0]++) {
        write_texel(t_out, out_pos, VEC4_T(
              results.data[idx_c][idx_r][0],
              results.data[idx_c][idx_r][1],
              results.data[idx_c][idx_r][2],
              results.data[idx_c][idx_r][3]));
      }
    $else:
      write_texel(t_out, out_pos, VEC4_T(
              results.data[idx_c][0],
              results.data[idx_c][1],
              results.data[idx_c][2],
              results.data[idx_c][3]));
  }
}

#endif
