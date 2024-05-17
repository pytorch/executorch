/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// To convince the SPIR-V compiler to unroll the loops optimally, need this
// macro
#define FOUR 4

// we avoid mat4 and vec4 usage here as they compile to much less efficient
// SPIR-V
struct FloatMatrix {
  float data[FOUR][FOUR][FOUR];
};

#ifdef MAT2_IS_TRANSPOSED
vec4 matmul_naive_W_packed_W_packed(
#else
vec4 matmul_naive_W_packed_H_packed(
#endif
    const sampler3D im_mat1,
    const sampler3D im_mat2,
    const ivec3 out_pos,
    const int width) {
  ivec3 mat1_pos = ivec3(0, out_pos.y, out_pos.z);
#ifdef MAT2_IS_TRANSPOSED
  ivec3 mat2_pos = ivec3(0, out_pos.x * 4, 0);
#else
  ivec3 mat2_pos = ivec3(out_pos.x * 4, 0, out_pos.z);
#endif

  vec4 texel = vec4(0);
  const int K = (width + 3) / 4;

  for (int i = 0; i < K; ++i) {
    const vec4 mat1_tex = texelFetch(im_mat1, mat1_pos, 0);
#ifdef MAT2_IS_TRANSPOSED
    const vec4 sums = vec4(
        dot(mat1_tex, texelFetch(im_mat2, mat2_pos, 0)),
        dot(mat1_tex, texelFetch(im_mat2, mat2_pos + ivec3(0, 1, 0), 0)),
        dot(mat1_tex, texelFetch(im_mat2, mat2_pos + ivec3(0, 2, 0), 0)),
        dot(mat1_tex, texelFetch(im_mat2, mat2_pos + ivec3(0, 3, 0), 0)));
#else
    const vec4 sums = vec4(
        dot(mat1_tex, texelFetch(im_mat2, mat2_pos, 0)),
        dot(mat1_tex, texelFetch(im_mat2, mat2_pos + ivec3(1, 0, 0), 0)),
        dot(mat1_tex, texelFetch(im_mat2, mat2_pos + ivec3(2, 0, 0), 0)),
        dot(mat1_tex, texelFetch(im_mat2, mat2_pos + ivec3(3, 0, 0), 0)));
#endif

    texel += sums;

    mat1_pos.x++;
#ifdef MAT2_IS_TRANSPOSED
    mat2_pos.x++;
#else
    mat2_pos.y++;
#endif
  }

  return texel;
}

#ifdef MAT2_IS_TRANSPOSED
vec4 matmul_naive_W_packed_H_packed(
#else
vec4 matmul_naive_W_packed_W_packed(
#endif
    const sampler3D im_mat1,
    const sampler3D im_mat2,
    const ivec3 out_pos,
    const int width) {
  ivec3 mat1_pos = ivec3(0, out_pos.y, out_pos.z);
  ivec3 mat2_pos = ivec3(out_pos.x, 0, out_pos.z);

  vec4 texel = vec4(0);
  int K = divup4(width);

  for (int i = 0; i < K; ++i) {
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
  }

  return texel;
}

// get texel from self tensor (width_packed) in addmm
vec4 get_texel_W_packed(
    sampler3D im_self,
    const ivec3 pos,
    const bool broadcast_at_width,
    const bool broadcast_at_height) {
  vec4 self_texel;
  // self is of shape {1}
  if (broadcast_at_width && broadcast_at_height) {
    self_texel = texelFetch(im_self, ivec3(0, 0, 0), 0).xxxx;
  }
  // self is of shape {*, 1}
  else if (broadcast_at_width) {
    self_texel = texelFetch(im_self, ivec3(0, pos.y, 0), 0).xxxx;
  }
  // self is of shape {1, *}
  else if (broadcast_at_height) {
    self_texel = texelFetch(im_self, ivec3(pos.x, 0, 0), 0);
  } else {
    self_texel = texelFetch(im_self, ivec3(pos.x, pos.y, 0), 0);
  }

  return self_texel;
}

// get texel from self tensor (channel_packed) in addmm
vec4 get_texel_C_packed(
    sampler3D im_self,
    const ivec3 pos,
    const bool broadcast_at_width,
    const bool broadcast_at_height) {
  vec4 self_texel;
  // self is of shape {1}
  if (broadcast_at_width && broadcast_at_height) {
    self_texel = texelFetch(im_self, ivec3(0, 0, 0), 0);
  }
  // self is of shape {*, 1}
  else if (broadcast_at_width) {
    self_texel = texelFetch(im_self, ivec3(0, pos.y, 0), 0);
  }
  // self is of shape {1, *}
  else if (broadcast_at_height) {
    self_texel = texelFetch(im_self, ivec3(pos.x, 0, 0), 0);
  } else {
    self_texel = texelFetch(im_self, ivec3(pos.x, pos.y, 0), 0);
  }

  return self_texel;
}

FloatMatrix matmul_partial_4x4(
    sampler3D im_mat1,
    sampler3D im_mat2,
    const ivec3 pos,
    const int batch_size,
    const int K_texel_len) {
  FloatMatrix results;
  for (int i = 0; i < FOUR; i++) {
    for (int j = 0; j < FOUR; j++) {
      for (int k = 0; k < FOUR; k++) {
        results.data[i][j][k] = 0.0f;
      }
    }
  }
  vec4 im_mat1_partial_load[FOUR];
  vec4 im_mat2_partial_load[FOUR];

  for (int batch_idx = 0; batch_idx < FOUR; batch_idx++) {
    if (FOUR * pos.z + batch_idx >= batch_size) {
      break;
    }
    int mat_z = FOUR * pos.z + batch_idx;
    for (int mat1_x = 0; mat1_x < K_texel_len; mat1_x++) {
      for (int offset = 0; offset < FOUR; offset++) {
        // read and cache 4x4 tile of im_mat1
        const int mat1_y = (FOUR * pos.y) + offset;
        const ivec3 mat1_pos = ivec3(mat1_x, mat1_y, mat_z);
        im_mat1_partial_load[offset] = texelFetch(im_mat1, mat1_pos, 0);
        // read and cache 4x4 tile of im_mat2
#ifdef MAT2_IS_TRANSPOSED
        const int mat2_y = (FOUR * pos.x) + offset;
        const ivec3 mat2_pos = ivec3(mat1_x, mat2_y, 0);
        im_mat2_partial_load[offset] = texelFetch(im_mat2, mat2_pos, 0);
#else
        const int mat2_x = (FOUR * pos.x) + offset;
        const ivec3 mat2_pos = ivec3(mat2_x, mat1_x, mat_z);
        im_mat2_partial_load[offset] = texelFetch(im_mat2, mat2_pos, 0);
#endif
      }
      // perform partial dot products and add partial result to results
      for (int out_row = 0; out_row < FOUR; out_row++) {
        for (int out_col = 0; out_col < FOUR; out_col++) {
          results.data[out_row][out_col][batch_idx] +=
              dot(im_mat1_partial_load[out_row], im_mat2_partial_load[out_col]);
        }
      }
    }
  }
  return results;
}
