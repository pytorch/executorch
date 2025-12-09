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

$if HAS_BIAS:
  #define HAS_BIAS

${layout_declare_tensor(B, "w", "out_tensor", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "mat1_tensor", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "mat2_tensor", DTYPE, "texture3d")}
$if HAS_BIAS:
  ${layout_declare_tensor(B, "r", "bias_tensor", DTYPE, "texture3d")}

layout(push_constant) uniform restrict Block {
  ivec4 out_sizes;
  ivec4 mat1_sizes;
  ivec4 mat2_sizes;
  ivec3 out_limits;
  $if HAS_BIAS:
    ivec4 bias_sizes;
    float alpha;
    float beta;
};

#include "indexing_utils.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "out_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 out_axis_map = unhash_axis_map(out_layout);
const lowp int out_packed_dim = unhash_packed_dim(out_layout);

${layout_declare_spec_const(C, "int", "mat1_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 mat1_axis_map = unhash_axis_map(mat1_layout);
const lowp int mat1_packed_dim = unhash_packed_dim(mat1_layout);

${layout_declare_spec_const(C, "int", "mat2_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 mat2_axis_map = unhash_axis_map(mat2_layout);
const lowp int mat2_packed_dim = unhash_packed_dim(mat2_layout);

$if HAS_BIAS:
  ${layout_declare_spec_const(C, "int", "bias_layout", "DEFAULT_LAYOUT")}
  const lowp ivec4 bias_axis_map = unhash_axis_map(bias_layout);
  const lowp int bias_packed_dim = unhash_packed_dim(bias_layout);

#ifdef HAS_BIAS
vec4 get_bias_texel_W_packed(ivec3 logical_pos) {
  ivec3 bias_pos = ivec3(0);
  if (bias_sizes.y == 1) {
    bias_pos[bias_axis_map.y] = 0;
  } else {
    bias_pos[bias_axis_map.y] = logical_pos.y;
  }
  if (bias_sizes.x == 1) {
    bias_pos[bias_axis_map.x] = 0;
    vec4 bias_texel = texelFetch(bias_tensor, bias_pos, 0);
    // Only the first value is valid, the rest is 0 padding
    return vec4(bias_texel.x);
  } else {
    bias_pos[bias_axis_map.x] = logical_pos.x;
  }

  return texelFetch(bias_tensor, bias_pos, 0);
}
#endif // HAS_BIAS

vec4 matmul_naive_k_dim_packed(const ivec3 out_lpos) {
  ivec3 mat1_pos;
  mat1_pos[mat1_axis_map.x] = 0;
  mat1_pos[mat1_axis_map.y] = out_lpos.y;
  mat1_pos[mat1_axis_map.z] = out_lpos.z;
#ifdef MAT2_IS_TRANSPOSED
  const int mat2_k_axis = mat2_axis_map.x;
  const int mat2_row_axis = mat2_axis_map.y;
#else
  const int mat2_k_axis = mat2_axis_map.y;
  const int mat2_row_axis = mat2_axis_map.x;
#endif // MAT2_IS_TRANSPOSED

  vec4 texel = vec4(0);
  const int K = divup4(mat1_sizes.x);

  for (int i = 0; i < K; ++i) {
    const vec4 mat1_tex = texelFetch(mat1_tensor, mat1_pos, 0);

    vec4 sums;
    for (int r = 0; r < 4; ++r) {
      // On-demand construction of mat2_pos appears to provide the lowest
      // latency. Surprisingly, this doesn't translate to mat1_pos.
      ivec3 mat2_pos = ivec3(0);
      mat2_pos[mat2_k_axis] = i;
      mat2_pos[mat2_row_axis] = out_lpos.x * 4 + r;
#ifndef MAT2_IS_TRANSPOSED
      mat2_pos[mat2_axis_map.z] = out_lpos.z;
#endif // MAT2_IS_TRANSPOSED
      sums[r] = dot(mat1_tex, texelFetch(mat2_tensor, mat2_pos, 0));
    }

    texel += sums;

    mat1_pos[mat1_axis_map.x]++;
  }

  return texel;
}

vec4 matmul_naive_k_dim_packed_row_dim_packed(const ivec3 out_lpos) {
  ivec3 mat1_pos;
  mat1_pos[mat1_axis_map.x] = 0;
  mat1_pos[mat1_axis_map.y] = out_lpos.y;
  mat1_pos[mat1_axis_map.z] = out_lpos.z;

  ivec3 mat2_pos;
  mat2_pos[mat2_axis_map.x] = out_lpos.x;
  mat2_pos[mat2_axis_map.y] = 0;
  mat2_pos[mat2_axis_map.z] = out_lpos.z;

  ivec3 mat2_pos_offset = ivec3(0);
  mat2_pos_offset[mat2_axis_map.y] = 1;

  const int mat2_y_axis = mat2_axis_map.y;

  vec4 texel = vec4(0);
  const int K = divup4(mat1_sizes.x);

  for (int i = 0;
       i < K;
       ++i, mat1_pos[mat1_axis_map.x]++, mat2_pos[mat2_axis_map.y]+=4) {
    const vec4 mat1_tex = texelFetch(mat1_tensor, mat1_pos, 0);

    for (int r = 0; r < 4; ++r) {
      if (4 * i + r >= mat2_sizes.y) {
        continue;
      }
      // On-demand construction of mat2_pos appears to provide the lowest
      // latency. Surprisingly, this doesn't translate to mat1_pos.
      ivec3 mat2_pos = ivec3(0);
      mat2_pos[mat2_axis_map.x] = out_lpos.x;
      mat2_pos[mat2_axis_map.y] = 4 * i + r;
      mat2_pos[mat2_axis_map.z] = out_lpos.z;

      vec4 mat1_comp_vec = vec4(mat1_tex[r]);
      texel = fma(mat1_comp_vec, texelFetch(mat2_tensor, mat2_pos, 0), texel);
    }
  }

  return texel;
}

void main() {
  const ivec3 out_lpos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(out_lpos, out_limits))) {
    return;
  }

  vec4 texel = vec4(0);

#ifdef MAT2_IS_TRANSPOSED
  if (mat2_packed_dim == W_DIM) {
    texel = matmul_naive_k_dim_packed(out_lpos);
  } else {
    texel = matmul_naive_k_dim_packed_row_dim_packed(out_lpos);
  }
#else
  if (mat2_packed_dim == W_DIM) {
    texel = matmul_naive_k_dim_packed_row_dim_packed(out_lpos);
  } else {
    texel = matmul_naive_k_dim_packed(out_lpos);
  }
#endif // MAT2_IS_TRANSPOSED

#ifdef HAS_BIAS
  vec4 bias_texel = get_bias_texel_W_packed(out_lpos);
  texel = beta * bias_texel + alpha * texel;
#endif // HAS_BIAS

  write_texel_lpos(out_tensor, out_lpos, texel, out_axis_map);
}
