/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define T ${buffer_scalar_type(DTYPE)}
#define VEC4_T ${buffer_gvec_type(DTYPE, 4)}

#define TILE_ROWS ${TILE_ROWS}

#define NGROUPS 8
#define NWORKERS 8

${define_required_extensions(DTYPE)}
$if WEIGHT_STORAGE == "buffer":
  ${define_required_extensions("uint8")}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, OUT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_mat1", DTYPE, IN_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_qmat2", "uint8", WEIGHT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_qparams", DTYPE, "buffer", is_scalar_array=False)}

layout(push_constant) uniform restrict Block {
  ivec4 out_sizes;
  ivec4 mat1_sizes;
  ivec4 qmat2_sizes;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int group_size = 64;

shared VEC4_T partial_sums[NGROUPS][NWORKERS][TILE_ROWS][2];

/*
 * This shader computes a linear operator between a floating point input matrix
 * x and a weights matrix that is quantized to 4 bits. Please refer to the
 * q_4w_linear shader for more details.
 *
 * This shader implements a co-operative algorithm to compute the output. The
 * work group size is {NGROUP, 1, NWORKERS}, and each group of NWORKERS threads
 * cooperative to compute TILE_ROWS * 2 output texels. Therefore,
 * NGROUP * TILE_ROWS * 2 output texels are computed across one work group.
 *
 * The threads co-operate by each thread computing a partial reduction along the
 * K dimension. To illustrate the computation, consider a scalar variant of the
 * algorithm that computes the dot product of 2 vectors. Also assume that
 * NWORKERS is 8.
 *
 * Thread 1 in each group will compute:
 * (mat1[0] * mat2[0]) + (mat1[8] * mat2[8]) + (mat1[16] * mat2[16]) + ...
 *
 * Thread 2 in each group will compute:
 * (mat1[1] * mat2[1]) + (mat2[9] * mat2[9]) + (mat1[17] * mat2[17]) + ...
 *
 * Thread 3 in each group will compute:
 * (mat1[2] * mat2[2]) + (mat2[10] * mat2[10]) + (mat1[18] * mat2[18]) + ...
 *
 * The partial accumulations is structured such that memory accesses in each
 * loop iteration can be coalesced.
 *
 * Then, at the end first thread in each group will accumulate the partial
 * accumulations computed by each thread to obtain the final result.
 *
 * Note that this shader assumes that all tensors are width packed.
 */
void main() {
  const uint out_row = gl_GlobalInvocationID.y * TILE_ROWS;
  // Each thread writes out 2 texels along the width axis, equivalent to 8
  // scalar elements. Therefore multiply the thread_idx.x by 8.
  const uint out_col = gl_GlobalInvocationID.x << 3;
  // Similar reasoning to the above, each thread works on 2 texels along the
  // width axis so multiply thread_idx.x by 2.
  const int out_col_texel_idx = int(gl_GlobalInvocationID.x) << 1;

  const uint gid = gl_LocalInvocationID.x; // group id
  const uint wid = gl_LocalInvocationID.z; // worker id

  if (out_col >= out_sizes.x || out_row >= out_sizes.y) {
    return;
  }

  const int num_blocks = mat1_sizes.x / group_size;

  VEC4_T mat1[TILE_ROWS];
  VEC4_T qmat2[4][2];
  VEC4_T local_sums[TILE_ROWS][2];

  [[unroll]] for (int r = 0; r < TILE_ROWS; ++r) {
    local_sums[r][0] = VEC4_T(0);
    local_sums[r][1] = VEC4_T(0);
  }

  VEC4_T scales[2];
  VEC4_T zeros[2];

  $if WEIGHT_STORAGE == "buffer":
    const int qmat2_stride = qmat2_sizes.x >> 2;
  $if PARAMS_STORAGE == "buffer":
    const int qparams_y_stride = out_sizes.x >> 2;
    const int qparams_z_stride = qparams_y_stride * 2;

  for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
    $if PARAMS_STORAGE == "buffer":
      scales[0] = t_qparams[block_idx * qparams_z_stride + out_col_texel_idx];
      zeros[0] = t_qparams[block_idx * qparams_z_stride + out_col_texel_idx + qparams_y_stride];

      scales[1] = t_qparams[block_idx * qparams_z_stride + out_col_texel_idx + 1];
      zeros[1] = t_qparams[block_idx * qparams_z_stride + out_col_texel_idx + 1 + qparams_y_stride];
    $else:
      scales[0] = texelFetch(t_qparams, ivec3(out_col_texel_idx, 0, block_idx), 0);
      zeros[0] = texelFetch(t_qparams, ivec3(out_col_texel_idx, 1, block_idx), 0);

      scales[1] = texelFetch(t_qparams, ivec3(out_col_texel_idx + 1, 0, block_idx), 0);
      zeros[1] = texelFetch(t_qparams, ivec3(out_col_texel_idx + 1, 1, block_idx), 0);

    for (uint g_idx = 4 * wid; g_idx < group_size; g_idx += (4 * NWORKERS)) {
      const uint k = block_idx * group_size + g_idx;

      // Preload B
      [[unroll]] for (int r = 0; r < 4; ++r) {
        $if WEIGHT_STORAGE == "buffer":
          const u8vec4 packed_weight_tex = t_qmat2[(k + r) * qmat2_stride + gl_GlobalInvocationID.x];
        $else:
          const uvec4 packed_weight_tex = texelFetch(
              t_qmat2,
              ivec2(gl_GlobalInvocationID.x, k + r),
              0);

        qmat2[r][0] = (VEC4_T((packed_weight_tex & 0xF0) >> 4) - 8.0) * scales[0] + zeros[0];
        qmat2[r][1] = (VEC4_T(packed_weight_tex & 0x0F) - 8.0) * scales[1] + zeros[1];
      }

      // Preload A
      [[unroll]] for (int r = 0; r < TILE_ROWS; ++r) {
        $if IN_STORAGE == "buffer":
          mat1[r] = t_mat1[((out_row + r) * mat1_sizes.x + k) >> 2];
        $else:
          mat1[r] = texelFetch(t_mat1, ivec3(k >> 2, out_row + r, 0), 0);
      }

      // Accumulate local output tile
      [[unroll]] for (int r = 0; r < TILE_ROWS; ++r) {
        local_sums[r][0] +=   mat1[r].x * qmat2[0][0]
                      + mat1[r].y * qmat2[1][0]
                      + mat1[r].z * qmat2[2][0]
                      + mat1[r].w * qmat2[3][0];

        local_sums[r][1] +=   mat1[r].x * qmat2[0][1]
                      + mat1[r].y * qmat2[1][1]
                      + mat1[r].z * qmat2[2][1]
                      + mat1[r].w * qmat2[3][1];
      }
    }
  }

  [[unroll]] for (int r = 0; r < TILE_ROWS; ++r) {
    partial_sums[gid][wid][r][0] = local_sums[r][0];
    partial_sums[gid][wid][r][1] = local_sums[r][1];
  }

  memoryBarrierShared();
  barrier();

  if (wid != 0) {
    return;
  }

  VEC4_T sums[TILE_ROWS][2];

  for (int r = 0; r < TILE_ROWS; ++r) {
    sums[r][0] = VEC4_T(0);
    sums[r][1] = VEC4_T(0);
    [[unroll]] for (int worker = 0; worker < NWORKERS; ++ worker) {
      sums[r][0] += partial_sums[gid][worker][r][0];
      sums[r][1] += partial_sums[gid][worker][r][1];
    }
  }

  [[unroll]] for (int r = 0; r < TILE_ROWS; ++r) {
    $if OUT_STORAGE == "buffer":
      t_out[((out_row + r) * out_sizes.x + out_col) >> 2] = sums[r][0];
      t_out[((out_row + r) * out_sizes.x + out_col + 4) >> 2] = sums[r][1];
    $else:
      imageStore(t_out, ivec3(out_col_texel_idx, out_row + r, 0), sums[r][0]);
      imageStore(t_out, ivec3(out_col_texel_idx + 1, out_row + r, 0), sums[r][1]);
  }
}
