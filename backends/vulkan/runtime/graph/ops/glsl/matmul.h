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
    int width) {
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
    int width) {
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
    ivec3 pos,
    int broadcast_at_width,
    int broadcast_at_height) {
  vec4 self_texel;
  // self is of shape {1}
  if (broadcast_at_width == 1 && broadcast_at_height == 1) {
    self_texel = texelFetch(im_self, ivec3(0, 0, 0), 0).xxxx;
  }
  // self is of shape {*, 1}
  else if (broadcast_at_width == 1) {
    self_texel = texelFetch(im_self, ivec3(0, pos.y, 0), 0).xxxx;
  }
  // self is of shape {1, *}
  else if (broadcast_at_height == 1) {
    self_texel = texelFetch(im_self, ivec3(pos.x, 0, 0), 0);
  } else {
    self_texel = texelFetch(im_self, pos, 0);
  }

  return self_texel;
}

// get texel from self tensor (channel_packed) in addmm
vec4 get_texel_C_packed(
    sampler3D im_self,
    ivec3 pos,
    int broadcast_at_width,
    int broadcast_at_height) {
  vec4 self_texel;
  // self is of shape {1}
  if (broadcast_at_width == 1 && broadcast_at_height == 1) {
    self_texel = texelFetch(im_self, ivec3(0, 0, 0), 0);
  }
  // self is of shape {*, 1}
  else if (broadcast_at_width == 1) {
    self_texel = texelFetch(im_self, ivec3(0, pos.y, 0), 0);
  }
  // self is of shape {1, *}
  else if (broadcast_at_height == 1) {
    self_texel = texelFetch(im_self, ivec3(pos.x, 0, 0), 0);
  } else {
    self_texel = texelFetch(im_self, pos, 0);
  }

  return self_texel;
}

FloatMatrix matmul_partial_4x4(
    sampler3D im_mat1,
    sampler3D im_mat2,
    ivec3 pos,
    int batch_size,
    int step_size,
    int reminder) {
  FloatMatrix results;
  for (int i = 0; i < FOUR; i++) {
    for (int j = 0; j < FOUR; j++) {
      for (int k = 0; k < FOUR; k++) {
        results.data[i][j][k] = 0.0f;
      }
    }
  }
  // read and cache 4x4 tile of im_mat1 (4 adjacent rows)
  vec4 im_mat1_partial_rows[FOUR];
  vec4 im_mat2_partial_cols[FOUR];

  for (int c = 0; c < FOUR; c++) {
    if (FOUR * pos.z + c >= batch_size) {
      break;
    }
    for (int j = 0; j < step_size; j++) {
      for (int k = 0; k < FOUR; k++) {
        const int pos_y_offset = (FOUR * pos.y) + k;
        const ivec3 pos_rd = ivec3(j, pos_y_offset, FOUR * pos.z + c);
        im_mat1_partial_rows[k] = texelFetch(im_mat1, pos_rd, 0);
        // set the value out of the boundary to be 0
        if (j == step_size - 1 && reminder > 0) {
          for (int kk = 0; kk < 4 - reminder; kk++) {
            im_mat1_partial_rows[k][3 - kk] = 0;
          }
        }
      }
      // read and cache 4x4 tile of im_mat2 (4 adjacent columns)
      for (int k = 0; k < FOUR; k++) {
        const int pos_x_offset = (FOUR * pos.x) + k;
        const ivec3 pos_rd = ivec3(pos_x_offset, j, FOUR * pos.z + c);
        im_mat2_partial_cols[k] = texelFetch(im_mat2, pos_rd, 0);
        // set the value out of the boundary to be 0
        if (j == step_size - 1 && reminder > 0) {
          for (int kk = 0; kk < 4 - reminder; kk++) {
            im_mat2_partial_cols[k][3 - kk] = 0;
          }
        }
      }
      // perform partial dot products and add partial result to results
      for (int idx_r = 0; idx_r < FOUR; idx_r++) {
        for (int idx_c = 0; idx_c < FOUR; idx_c++) {
          results.data[idx_r][idx_c][c] +=
              dot(im_mat1_partial_rows[idx_r], im_mat2_partial_cols[idx_c]);
        }
      }
    }
  }
  return results;
}
