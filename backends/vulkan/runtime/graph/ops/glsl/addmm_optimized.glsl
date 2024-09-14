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

$if BATCH_MODE:
  #define BATCH_MODE

$if HAS_BIAS:
  #define HAS_BIAS

#include "indexing_utils.h"

${layout_declare_tensor(B, "w", "out_tensor", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "mat1_tensor", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "mat2_tensor", DTYPE, "texture3d")}
$if HAS_BIAS:
  ${layout_declare_tensor(B, "r", "bias_tensor", DTYPE, "texture3d")}
${layout_declare_ubo(B, "ivec4", "out_sizes")}
${layout_declare_ubo(B, "ivec4", "out_axis_map")}
${layout_declare_ubo(B, "ivec4", "mat1_sizes")}
${layout_declare_ubo(B, "ivec4", "mat1_axis_map")}
${layout_declare_ubo(B, "ivec4", "mat2_sizes")}
${layout_declare_ubo(B, "ivec4", "mat2_axis_map")}
$if HAS_BIAS:
  ${layout_declare_ubo(B, "ivec4", "bias_sizes")}
  ${layout_declare_ubo(B, "ivec4", "bias_axis_map")}
  ${layout_declare_ubo(B, "float", "alpha", "float", "beta")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int out_packed_dim = C_DIM;

// To convince the SPIR-V compiler to unroll the loops optimally, need this
// macro
#define FOUR 4

#define TILE_ROWS ${TILE_ROWS}

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
#endif // BATCH_MODE

#ifdef HAS_BIAS
// get texel from self tensor (channel_packed) in addmm
vec4 get_texel_C_packed(const ivec2 idx) {
  ivec3 bias_pos = ivec3(0);
  if (bias_sizes.x > 1) {
    bias_pos[bias_axis_map.x] = idx.x;
  }
  if (bias_sizes.y > 1) {
    bias_pos[bias_axis_map.y] = idx.y;
  }

  return texelFetch(bias_tensor, bias_pos, 0);
}
#endif // HAS_BIAS

FloatMatrix matmul_partial(const ivec4 out_idx_tl) {
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
  vec4 mat1_tensor_partial_load[TILE_ROWS];
  vec4 mat2_tensor_partial_load[FOUR];

#ifdef MAT2_IS_TRANSPOSED
  const int mat2_k_axis = mat2_axis_map.x;
  const int mat2_row_axis = mat2_axis_map.y;
#else
  const int mat2_k_axis = mat2_axis_map.y;
  const int mat2_row_axis = mat2_axis_map.x;
#endif // MAT2_IS_TRANSPOSED

#ifdef BATCH_MODE
  for (int batch_idx = 0; batch_idx < FOUR; batch_idx++) {
    if (out_idx_tl.z + batch_idx >= out_sizes.z) {
      break;
    }
#endif // BATCH_MODE
  for (int k = 0; k < mat1_sizes.x; k+=4) {
    const int k_div4 = k >> 2;
    // read and cache (4 x TILE_ROWS) tile of mat1
    for (int r = 0; r < TILE_ROWS; r++) {
      ivec3 mat1_pos = ivec3(0);
      mat1_pos[mat1_axis_map.x] = k_div4;
      mat1_pos[mat1_axis_map.y] = out_idx_tl.y + r;
#ifdef BATCH_MODE
      mat1_pos[mat1_axis_map.z] = out_idx_tl.z + batch_idx;
#endif // BATCH_MODE

      mat1_tensor_partial_load[r] = texelFetch(mat1_tensor, mat1_pos, 0);
    }

    // read and cache (4 x 4) tile of mat2
    for (int r = 0; r < FOUR; ++r) {
      ivec3 mat2_pos = ivec3(0);
      mat2_pos[mat2_k_axis] = k_div4;
      mat2_pos[mat2_row_axis] = out_idx_tl.x + r;
#if defined(BATCH_MODE) && !defined(MAT2_IS_TRANSPOSED)
      mat2_pos[mat2_axis_map.z] = out_idx_tl.z + batch_idx;
#endif // BATCH_MODE

      mat2_tensor_partial_load[r] = texelFetch(mat2_tensor, mat2_pos, 0);
    }

    // perform partial dot products and add partial result to results
    for (int out_row = 0; out_row < TILE_ROWS; out_row++) {
      for (int out_col = 0; out_col < FOUR; out_col++) {
#ifdef BATCH_MODE
        results.data[out_row][out_col][batch_idx] +=
#else
        results.data[out_row][out_col] +=
#endif // BATCH_MODE
            dot(mat1_tensor_partial_load[out_row], mat2_tensor_partial_load[out_col]);
      }
    }
  }
#ifdef BATCH_MODE
  }
#endif // BATCH_MODE

  return results;
}

//
// Write result matrix to output (3D matmul)
//

void write_results_C_packed(const ivec4 out_idx_tl, FloatMatrix results) {
  ivec3 out_pos = to_texture_pos(
      out_idx_tl, out_sizes, out_axis_map, out_packed_dim);

  for (int tile_c = 0;
       tile_c < TILE_ROWS;
       tile_c++, out_pos[out_axis_map.y]++) {
    out_pos[out_axis_map.x] = out_idx_tl.x;

    for (int tile_r = 0;
         tile_r < FOUR;
         tile_r++, out_pos[out_axis_map.x]++) {

#ifdef HAS_BIAS
      ivec2 bias_idx;
      bias_idx[bias_axis_map.x] = out_pos[out_axis_map.x];
      bias_idx[bias_axis_map.y] = out_pos[out_axis_map.y];
      float bias_val = get_texel_C_packed(bias_idx).x;
#ifdef BATCH_MODE
      vec4 bias_texel = vec4(bias_val);
#else
      vec4 bias_texel = vec4(bias_val, 0, 0, 0);
#endif // BATCH_MODE
#endif // HAS_BIAS

#ifdef BATCH_MODE
      vec4 out_texel = vec4(
            results.data[tile_c][tile_r][0],
            results.data[tile_c][tile_r][1],
            results.data[tile_c][tile_r][2],
            results.data[tile_c][tile_r][3]);
#else
      vec4 out_texel = vec4(
            results.data[tile_c][tile_r],
            0.0,
            0.0,
            0.0);
#endif // BATCH_MODE

#ifdef HAS_BIAS
      imageStore(out_tensor, out_pos, beta * bias_texel + alpha * out_texel);
#else
      imageStore(out_tensor, out_pos, out_texel);
#endif // HAS_BIAS
    }
  }
}

void main() {
  // Each thread is responsible for calculating a (4 x TILE_ROWS x 1) tile of
  // output elements. If the input matrices are 3D, then a (4 x TILE_ROWS x 4)
  // tile of output elements will be computed. Note the sizes are written in
  // (W x H x C) format.
  const ivec3 tile_idx = ivec3(gl_GlobalInvocationID);

  // Calculate the tensor index of the top left element in the output tile
  const ivec4 out_idx_topleft = ivec4(
      tile_idx.x * 4,
      tile_idx.y * TILE_ROWS,
#ifdef BATCH_MODE
      tile_idx.z * 4,
#else
      tile_idx.z,
#endif // BATCH_MODE
      0);

  // If the top left element is already out of range, then skip
  if (any(greaterThanEqual(out_idx_topleft, out_sizes))) {
    return;
  }

  FloatMatrix results = matmul_partial(out_idx_topleft);

  write_results_C_packed(out_idx_topleft, results);
}
