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

${define_required_extensions(DTYPE)}
${define_required_extensions("int8")}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, OUT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_mat1", DTYPE, IN_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_qmat2", "uint8", WEIGHT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_qparams", DTYPE, "texture3D")}

layout(push_constant) uniform restrict Block {
  ivec4 out_sizes;
  ivec4 mat1_sizes;
  ivec4 qmat2_sizes;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int group_size = 64;

/*
 * This shader computes a linear operator between a floating point input matrix
 * x and a weights matrix that is quantized to 4 bits.
 *
 * The (W, H, C) shape of each tensor is:
 * - x: (K, M)
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
 *
 * Note that this shader assumes that all tensors are width packed.
 */
void main() {
  const uint out_row = gl_GlobalInvocationID.y;
  // Each thread writes out 2 texels along the width axis, equivalent to 8
  // scalar elements. Therefore multiply the thread_idx.x by 8.
  const uint out_col = gl_GlobalInvocationID.x << 3;
  // Similar reasoning to the above, each thread works on 2 texels along the
  // width axis so multiply thread_idx.x by 2.
  const int out_col_texel_idx = int(gl_GlobalInvocationID.x) << 1;

  if (out_col >= out_sizes.x || out_row >= out_sizes.y) {
    return;
  }

  const int num_blocks = mat1_sizes.x / group_size;

  VEC4_T sums[2];

  sums[0] = VEC4_T(0);
  sums[1] = VEC4_T(0);

  VEC4_T scales[2];
  VEC4_T zeros[2];

  $if WEIGHT_STORAGE == "buffer":
    const int qmat2_stride = qmat2_sizes.x >> 2;

  for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
    scales[0] = texelFetch(t_qparams, ivec3(out_col_texel_idx, 0, block_idx), 0);
    zeros[0] = texelFetch(t_qparams, ivec3(out_col_texel_idx, 1, block_idx), 0);

    scales[1] = texelFetch(t_qparams, ivec3(out_col_texel_idx + 1, 0, block_idx), 0);
    zeros[1] = texelFetch(t_qparams, ivec3(out_col_texel_idx + 1, 1, block_idx), 0);

    for (int g_idx = 0; g_idx < group_size; g_idx += 4) {
      const int k = block_idx * group_size + g_idx;

      $if IN_STORAGE == "buffer":
        const VEC4_T mat1_tex = t_mat1[(out_row * mat1_sizes.x + k) >> 2];
      $else:
        const VEC4_T mat1_tex = texelFetch(t_mat1, ivec3(k >> 2, out_row, 0), 0);

      for (int comp = 0; comp < 4; ++comp) {
        $if WEIGHT_STORAGE == "buffer":
          const u8vec4 packed_weight_tex = t_qmat2[(k + comp) * qmat2_stride + gl_GlobalInvocationID.x];
        $else:
          const uvec4 packed_weight_tex = texelFetch(
              t_qmat2,
              ivec3(gl_GlobalInvocationID.x, k + comp, 0),
              0);

        const uvec4 weight_tex_1 = (packed_weight_tex & 0xF0) >> 4;
        const uvec4 weight_tex_2 = packed_weight_tex & 0x0F;

        sums[0] += mat1_tex[comp] * ((vec4(weight_tex_1) - 8.0) * scales[0] + zeros[0]);
        sums[1] += mat1_tex[comp] * ((vec4(weight_tex_2) - 8.0) * scales[1] + zeros[1]);
      }
    }
  }

  $if OUT_STORAGE == "buffer":
    t_out[(out_row * out_sizes.x + out_col) >> 2] = sums[0];
    t_out[(out_row * out_sizes.x + out_col + 4) >> 2] = sums[1];
  $else:
    imageStore(t_out, ivec3(out_col_texel_idx, out_row, 0), sums[0]);
    imageStore(t_out, ivec3(out_col_texel_idx + 1, out_row, 0), sums[1]);
}
