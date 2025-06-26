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

${define_required_extensions(DTYPE)}
$if WEIGHT_STORAGE == "buffer" and WEIGHT_DTYPE == "uint8":
  ${define_required_extensions("uint8")}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, OUT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_mat1", DTYPE, IN_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_qmat2", WEIGHT_DTYPE, WEIGHT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_qparams", "float", PARAMS_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_input_scale", "float", "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_input_zero_point", "int", "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_output_scale", "float", "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_output_zero_point", "int", "buffer", is_scalar_array=True)}

layout(push_constant) uniform restrict Block {
  ivec4 out_sizes;
  ivec4 mat1_sizes;
  ivec4 qmat2_sizes;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int group_size = 64;

/*
 * This shader computes a linear operator between a quantized int8 input matrix
 * x and a weights matrix that is quantized to 4 bits, producing a quantized int8 output.
 *
 * The (W, H, C) shape of each tensor is:
 * - x: (K, M) - quantized int8 input
 * - weights: (N / 2, K)
 *   - The weights tensor has a data type of `uint8`. Each element in the tensor
 *     contains 2 4-bit values packed into a uint8.
 *   - See the pack_int4_linear_weight_transposed_interleave shader to see more
 *     details on how the weight tensor is stored.
 * - qparams: (2, N, number_of_groups)
 *   - This tensor contains the scales and zeros quantization parameters for the
 *     weights tensor. The weight tensor is quantized group-wise, which means
 *     that every `group_size` elements along the K dimension of the weights
 *     tensor has independent quantization parameters. Along the width dim, the
 *     first value contains the scale for the group and the second value
 *     contains the zero point for the group.
 * - output: (N, M) - quantized int8 output
 *
 * Each thread computes a tile of TILE_ROWS * 2 texels of the output tensor.
 *
 * Note that this shader assumes that all tensors are width packed.
 */

bool is_main_thread() {
  return gl_GlobalInvocationID.x == 0 && gl_GlobalInvocationID.y == 0 && gl_GlobalInvocationID.z == 0;
}

void main() {
  const uint out_row = gl_GlobalInvocationID.y * TILE_ROWS;
  const uint out_col = gl_GlobalInvocationID.x << 3;
  const int out_col_texel_idx = int(gl_GlobalInvocationID.x) << 1;

  if (out_col >= out_sizes.x || out_row >= out_sizes.y) {
    return;
  }

  const int num_blocks = mat1_sizes.x / group_size;

  VEC4_T mat1_quantized[TILE_ROWS];
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

    ivec4 int32_sums[TILE_ROWS][2];
    int input_sums[TILE_ROWS];

    // Initialize accumulators
    [[unroll]] for (int r = 0; r < TILE_ROWS; ++r) {
      int32_sums[r][0] = ivec4(0);
      int32_sums[r][1] = ivec4(0);
      input_sums[r] = 0;
    }

    for (int g_idx = 0; g_idx < group_size; g_idx += 4) {
      const int k = block_idx * group_size + g_idx;

      // Preload B (weights) - keep as quantized integers
      [[unroll]] for (int r = 0; r < 4; ++r) {
        $if WEIGHT_STORAGE == "buffer":
          const uvec4 packed_weight_tex = t_qmat2[(k + r) * qmat2_stride + gl_GlobalInvocationID.x];
        $else:
          const uvec4 packed_weight_tex = texelFetch(
              t_qmat2,
              ivec2(gl_GlobalInvocationID.x, k + r),
              0);

        // Unpack 4-bit weights to integers (subtract 8 as the 4-bit zero point)
        qmat2_quantized[r][0] = ivec4((packed_weight_tex & 0xF0) >> 4) - 8;
        qmat2_quantized[r][1] = ivec4(packed_weight_tex & 0x0F) - 8;
      }

      // Preload A (quantized input) - keep as quantized integers
      [[unroll]] for (int r = 0; r < TILE_ROWS; ++r) {
        $if IN_STORAGE == "buffer":
          mat1_quantized[r] = VEC4_T(t_mat1[((out_row + r) * mat1_sizes.x + k) >> 2] - t_input_zero_point[int(out_row) + r]);
        $else:
          mat1_quantized[r] = VEC4_T(texelFetch(t_mat1, ivec3(k >> 2, out_row + r, 0), 0) - t_input_zero_point[int(out_row) + r]);

        input_sums[r] += mat1_quantized[r].x + mat1_quantized[r].y + mat1_quantized[r].z + mat1_quantized[r].w;
      }

      // Accumulate in integer arithmetic: (input_quantized - input_zero_point) * (weight_quantized - weight_zero_point)
      [[unroll]] for (int r = 0; r < TILE_ROWS; ++r) {
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
    [[unroll]] for (int r = 0; r < TILE_ROWS; ++r) {
      if (out_row + r >= out_sizes.y) {
        continue;
      }

      float input_scale = t_input_scale[int(out_row) + r];
      float input_sum_scalar = float(input_sums[r]);

      final_result[r][0] += input_scale * (vec4(int32_sums[r][0]) * scales[0] + input_sum_scalar * zeros[0]);
      final_result[r][1] += input_scale * (vec4(int32_sums[r][1]) * scales[1] + input_sum_scalar * zeros[1]);
    }
  }

  // Apply ALL scaling at the very end
  [[unroll]] for (int r = 0; r < TILE_ROWS; ++r) {
    if (out_row + r >= out_sizes.y) {
      continue;
    }

    int token_idx = int(out_row) + r;
    float output_scale = t_output_scale[token_idx];
    int output_zero_point = t_output_zero_point[token_idx];

    VEC4_T quantized_out_0 = VEC4_T(clamp(
      ivec4(round(final_result[r][0] / output_scale)) + float(output_zero_point),
      -128, 127));
    VEC4_T quantized_out_1 = VEC4_T(clamp(
      ivec4(round(final_result[r][1] / output_scale)) + float(output_zero_point),
      -128, 127));

    $if OUT_STORAGE == "buffer":
      t_out[((out_row + r) * out_sizes.x + out_col) >> 2] = quantized_out_0;
      t_out[((out_row + r) * out_sizes.x + out_col + 4) >> 2] = quantized_out_1;
    $else:
      imageStore(t_out, ivec3(out_col_texel_idx, out_row + r, 0), quantized_out_0);
      imageStore(t_out, ivec3(out_col_texel_idx + 1, out_row + r, 0), quantized_out_1);
  }
}
