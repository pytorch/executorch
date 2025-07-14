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
$if IN_STORAGE == "buffer":
  ${define_required_extensions("int8")}
$if WEIGHT_STORAGE == "buffer":
  ${define_required_extensions("uint8")}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, OUT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_mat1", "int8", IN_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_qmat2", "uint8", WEIGHT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_scales", "float", PARAMS_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_zeros", "int", PARAMS_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_input_scale", "float", PARAMS_STORAGE, is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_input_zero_point", "int", PARAMS_STORAGE, is_scalar_array=True)}

layout(push_constant) uniform restrict Block {
  ivec4 out_sizes;
  ivec4 mat1_sizes;
  ivec4 qmat2_sizes;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int group_size = 64;

shared vec4 partial_results[NGROUPS][NWORKERS][TILE_ROWS][2];

/*
 * This shader computes a linear operator between a quantized int8 input matrix
 * x and a weights matrix that is quantized to 4 bits, producing a float output.
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
  const uint out_col = gl_GlobalInvocationID.x << 3;
  const int out_col_texel_idx = int(gl_GlobalInvocationID.x) << 1;

  const uint gid = gl_LocalInvocationID.x; // group id
  const uint wid = gl_LocalInvocationID.z; // worker id

  if (out_col >= out_sizes.x || out_row >= out_sizes.y) {
    return;
  }

  const int num_blocks = mat1_sizes.x / group_size;

  ivec4 mat1_quantized[TILE_ROWS];
  ivec4 qmat2_quantized[4][2];
  vec4 final_result[TILE_ROWS][2];

  // Initialize accumulators
  [[unroll]] for (int r = 0; r < TILE_ROWS; ++r) {
    final_result[r][0] = vec4(0.0);
    final_result[r][1] = vec4(0.0);
  }

  vec4 scales[2];
  vec4 zeros[2];

  $if WEIGHT_STORAGE == "buffer":
    const int qmat2_stride = qmat2_sizes.x >> 2;
  $if PARAMS_STORAGE == "buffer":
    const int qparams_stride = out_sizes.x >> 2;

  for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
    $if PARAMS_STORAGE == "buffer":
      scales[0] = t_weight_scales[block_idx * qparams_stride + out_col_texel_idx];
      scales[1] = t_weight_scales[block_idx * qparams_stride + out_col_texel_idx + 1];

      zeros[0] = vec4(t_weight_zeros[block_idx * qparams_stride + out_col_texel_idx]);
      zeros[1] = vec4(t_weight_zeros[block_idx * qparams_stride + out_col_texel_idx + 1]);
    $else:
      scales[0] = texelFetch(t_weight_scales, ivec3(out_col_texel_idx, block_idx, 0), 0);
      scales[1] = texelFetch(t_weight_scales, ivec3(out_col_texel_idx + 1, block_idx, 0), 0);

      zeros[0] = vec4(texelFetch(t_weight_zeros, ivec3(out_col_texel_idx, block_idx, 0), 0));
      zeros[1] = vec4(texelFetch(t_weight_zeros, ivec3(out_col_texel_idx + 1, block_idx, 0), 0));

    ivec4 int32_sums[TILE_ROWS][2];
    int input_sums[TILE_ROWS];

    // Initialize accumulators for this block
    [[unroll]] for (int r = 0; r < TILE_ROWS; ++r) {
      int32_sums[r][0] = ivec4(0);
      int32_sums[r][1] = ivec4(0);
      input_sums[r] = 0;
    }

    for (int g_idx = 4 * int(wid); g_idx < group_size; g_idx += (4 * NWORKERS)) {
      const int k = block_idx * group_size + g_idx;

      // Preload B (weights) - keep as quantized integers
      [[unroll]] for (int r = 0; r < 4; ++r) {
        $if WEIGHT_STORAGE == "buffer":
          const u8vec4 packed_weight_tex = t_qmat2[(k + r) * qmat2_stride + gl_GlobalInvocationID.x];
        $else:
          const uvec4 packed_weight_tex = texelFetch(
              t_qmat2,
              ivec2(gl_GlobalInvocationID.x, k + r),
              0);

        // Unpack 4-bit weights to integers and subtract zero point (8 for 4-bit)
        qmat2_quantized[r][0] = ivec4((packed_weight_tex & 0xF0) >> 4) - 8;
        qmat2_quantized[r][1] = ivec4(packed_weight_tex & 0x0F) - 8;
      }

      // Preload A (quantized input) - keep as quantized integers
      [[unroll]] for (int r = 0; r < TILE_ROWS; ++r) {
        $if IN_STORAGE == "buffer":
          mat1_quantized[r] = t_mat1[((out_row + r) * mat1_sizes.x + k) >> 2] - t_input_zero_point[int(out_row) + r];
        $else:
          mat1_quantized[r] = texelFetch(t_mat1, ivec3(k >> 2, out_row + r, 0), 0) - t_input_zero_point[int(out_row) + r];
      }

      // Accumulate in integer arithmetic: (input_quantized - input_zero_point) * (weight_quantized - weight_zero_point)
      [[unroll]] for (int r = 0; r < TILE_ROWS; ++r) {
        input_sums[r] += mat1_quantized[r].x + mat1_quantized[r].y + mat1_quantized[r].z + mat1_quantized[r].w;

        int32_sums[r][0] +=   mat1_quantized[r].x * qmat2_quantized[0][0]
                            + mat1_quantized[r].y * qmat2_quantized[1][0]
                            + mat1_quantized[r].z * qmat2_quantized[2][0]
                            + mat1_quantized[r].w * qmat2_quantized[3][0];

        int32_sums[r][1] +=   mat1_quantized[r].x * qmat2_quantized[0][1]
                            + mat1_quantized[r].y * qmat2_quantized[1][1]
                            + mat1_quantized[r].z * qmat2_quantized[2][1]
                            + mat1_quantized[r].w * qmat2_quantized[3][1];
      }
    }

    // Incorporates this block's results into the final accumulation
    // Following proper quantization paradigm: result = input_scale * weight_scale *
    // Sum((input_quantized - input_zero) * (weight_quantized - weight_zero))
    [[unroll]] for (int r = 0; r < TILE_ROWS; ++r) {
      if (out_row + r >= out_sizes.y) {
        continue;
      }

      float input_scale = t_input_scale[int(out_row) + r];
      float input_sum_scalar = float(input_sums[r]);

      // Apply proper quantization paradigm: input_scale * weight_scale * (accumulator - weight_zero * input_sum)
      final_result[r][0] += input_scale * scales[0] * (vec4(int32_sums[r][0]) - zeros[0] * input_sum_scalar);
      final_result[r][1] += input_scale * scales[1] * (vec4(int32_sums[r][1]) - zeros[1] * input_sum_scalar);
    }
  }

  // Store worker results in shared memory
  [[unroll]] for (int r = 0; r < TILE_ROWS; ++r) {
    partial_results[gid][wid][r][0] = final_result[r][0];
    partial_results[gid][wid][r][1] = final_result[r][1];
  }

  memoryBarrierShared();
  barrier();

  // Only the first worker in each group accumulates and writes output
  if (wid != 0) {
    return;
  }

  vec4 cooperative_result[TILE_ROWS][2];

  for (int r = 0; r < TILE_ROWS; ++r) {
    cooperative_result[r][0] = vec4(0.0);
    cooperative_result[r][1] = vec4(0.0);
    [[unroll]] for (int worker = 0; worker < NWORKERS; ++worker) {
      cooperative_result[r][0] += partial_results[gid][worker][r][0];
      cooperative_result[r][1] += partial_results[gid][worker][r][1];
    }
  }

  // Apply final output quantization
  [[unroll]] for (int r = 0; r < TILE_ROWS; ++r) {
    $if OUT_STORAGE == "buffer":
      t_out[((out_row + r) * out_sizes.x + out_col) >> 2] = cooperative_result[r][0];
      t_out[((out_row + r) * out_sizes.x + out_col + 4) >> 2] = cooperative_result[r][1];
    $else:
      imageStore(t_out, ivec3(out_col_texel_idx, out_row + r, 0), cooperative_result[r][0]);
      imageStore(t_out, ivec3(out_col_texel_idx + 1, out_row + r, 0), cooperative_result[r][1]);
  }
}
