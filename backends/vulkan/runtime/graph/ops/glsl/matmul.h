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

vec4 matmul_naive_W_packed_H_packed(
    sampler3D im_mat1,
    sampler3D im_mat2,
    ivec3 mat1_pos,
    ivec3 mat2_pos,
    const int width) {
  vec4 texel = vec4(0);
  int K = (width + 3) / 4;

  for (int i = 0; i < K; ++i) {
    vec4 mat1_tex = texelFetch(im_mat1, mat1_pos, 0);
    vec4 sums = vec4(
        dot(mat1_tex, texelFetch(im_mat2, mat2_pos, 0)),
        dot(mat1_tex, texelFetch(im_mat2, mat2_pos + ivec3(1, 0, 0), 0)),
        dot(mat1_tex, texelFetch(im_mat2, mat2_pos + ivec3(2, 0, 0), 0)),
        dot(mat1_tex, texelFetch(im_mat2, mat2_pos + ivec3(3, 0, 0), 0)));

    texel += sums;

    mat1_pos.x++;
    mat2_pos.y++;
  }

  return texel;
}

vec4 matmul_naive_W_packed_W_packed(
    sampler3D im_mat1,
    sampler3D im_mat2,
    ivec3 mat1_pos,
    ivec3 mat2_pos,
    const int width) {
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
    self_texel = texelFetch(im_self, pos, 0);
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
    self_texel = texelFetch(im_self, pos, 0);
  }

  return self_texel;
}

FloatMatrix matmul_partial_4x4(
    sampler3D im_mat1,
    sampler3D im_mat2,
    const ivec3 pos,
    const int batch_size,
    const int K_texel_len,
    const int packed_dim_padding) {
  FloatMatrix results;
  for (int i = 0; i < FOUR; i++) {
    for (int j = 0; j < FOUR; j++) {
      for (int k = 0; k < FOUR; k++) {
        results.data[i][j][k] = 0.0f;
      }
    }
  }
  vec4 im_mat1_partial_rows[FOUR];
  vec4 im_mat2_partial_cols[FOUR];

  for (int batch_idx = 0; batch_idx < FOUR; batch_idx++) {
    if (FOUR * pos.z + batch_idx >= batch_size) {
      break;
    }
    // read and cache 4x4 tile of im_mat1 (4 adjacent rows)
    for (int mat1_x = 0; mat1_x < K_texel_len; mat1_x++) {
      for (int mat1_row = 0; mat1_row < FOUR; mat1_row++) {
        const int mat1_y = (FOUR * pos.y) + mat1_row;
        const ivec3 mat1_pos = ivec3(mat1_x, mat1_y, FOUR * pos.z + batch_idx);
        im_mat1_partial_rows[mat1_row] = texelFetch(im_mat1, mat1_pos, 0);
        // set the value out of the boundary to be 0
        if (mat1_x == K_texel_len - 1 && packed_dim_padding > 0) {
          for (int kk = 0; kk < packed_dim_padding; kk++) {
            im_mat1_partial_rows[mat1_row][3 - kk] = 0;
          }
        }
      }
      // read and cache 4x4 tile of im_mat2 (4 adjacent columns)
      for (int mat2_col = 0; mat2_col < FOUR; mat2_col++) {
        const int mat2_x = (FOUR * pos.x) + mat2_col;
        const ivec3 pos_rd = ivec3(mat2_x, mat1_x, FOUR * pos.z + batch_idx);
        im_mat2_partial_cols[mat2_col] = texelFetch(im_mat2, pos_rd, 0);
        // set the value out of the boundary to be 0
        if (mat1_x == K_texel_len - 1 && packed_dim_padding > 0) {
          for (int kk = 0; kk < packed_dim_padding; kk++) {
            im_mat2_partial_cols[mat2_col][3 - kk] = 0;
          }
        }
      }
      // perform partial dot products and add partial result to results
      for (int out_row = 0; out_row < FOUR; out_row++) {
        for (int out_col = 0; out_col < FOUR; out_col++) {
          results.data[out_row][out_col][batch_idx] +=
              dot(im_mat1_partial_rows[out_row], im_mat2_partial_cols[out_col]);
        }
      }
    }
  }
  return results;
}
