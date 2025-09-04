/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define T ${texel_load_component_type(DTYPE, IO_STORAGE)}
#define VEC4_T ${texel_load_type(DTYPE, IO_STORAGE)}

#define WGS ${WGS}

${define_required_extensions(DTYPE)}

layout(std430) buffer;

#include "indexing_utils.h"

${layout_declare_tensor(B, "w", "t_output", DTYPE, IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_input", DTYPE, IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_qmat2", "uint", WEIGHT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_qparams", DTYPE, "buffer", is_scalar_array=False)}

layout(push_constant) uniform restrict Block {
  ivec4 output_sizes;
  ivec4 input_sizes;
  ivec4 weight_sizes;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int group_size = 64;

shared VEC4_T partial_sums[WGS][2];

$if IO_STORAGE == "buffer":
  #define BUFFER_IO
$if WEIGHT_STORAGE == "buffer":
  #define BUFFER_WEIGHT

#include "qlinear_utils.glslh"

void main() {
  const uint lid = gl_LocalInvocationID.x;
  const uint n8 = gl_GlobalInvocationID.y;
  // The output tensor will have a shape of [n, 1, 1, 1]. Each thread computes
  // 8 output elements, so each thread will write to 8 elements starting at the
  // tensor index (gid.x * 8, 0, 0, 0).
  const uint n = MUL_8(n8);
  const uint K4 = DIV_UP_4(input_sizes.x);

  if (n >= output_sizes.x) {
    return;
  }

  VEC4_T out_texels[2];
  out_texels[0] = VEC4_T(0);
  out_texels[1] = VEC4_T(0);

  // initialize the group index to a value larger than the largest possible
  uint cur_group_idx = input_sizes.x;

  // Each thread in the work group accumulates a partial result.
  for (uint k4 = lid; k4 < DIV_UP_4(input_sizes.x); k4 += WGS) {
    const uint k = MUL_4(k4);
    const uint group_idx = k / group_size;

    VEC4_T scales[2];
    VEC4_T zeros[2];

    // Only update the scales/zeros if the current iteration is now working on a
    // new quantization group.
    if (group_idx != cur_group_idx) {
      // The qparams tensor contains the quantization scales and zeros, with
      // shape [2, N, K / group_size, 1].
      // Loading a texel from the qparams tensor will return 2 scales and 2
      // zeros for 2 adjacent output channels.
      uint qparams_bufi = group_idx * DIV_2(output_sizes.x) + DIV_2(n);
      VEC4_T scales_zeros_texels[4];
      $for comp in range(4):
        scales_zeros_texels[${comp}] = t_qparams[qparams_bufi++];

      scales[0] = VEC4_T(scales_zeros_texels[0].xz, scales_zeros_texels[1].xz);
      zeros[0] = VEC4_T(scales_zeros_texels[0].yw, scales_zeros_texels[1].yw);

      scales[1] = VEC4_T(scales_zeros_texels[2].xz, scales_zeros_texels[3].xz);
      zeros[1] = VEC4_T(scales_zeros_texels[2].yw, scales_zeros_texels[3].yw);

      cur_group_idx = group_idx;
    }
    // The input tensor will have a shape of [K, 1, 1, 1]; in each iteration,
    // load 4 elements starting from the tensor index (k, 0, 0, 0).
    VEC4_T in_texel = load_input_texel_1d(k4);
    // Extract each element of the in_texel into a separate vectorized variable;
    // these are used to "broadcast" the input values in subsequent fma calls.
    VEC4_T in_texel_val[4];
    $for comp in range(4):
      in_texel_val[${comp}] = VEC4_T(in_texel[${comp}]);

    uvec4 packed_weight_block = load_transposed_weight_block(k4, n8, K4);

    VEC4_T weight_texels[2];
    $for comp in range(4):
      {
        weight_texels[0].x = extract_4bit_from_transposed_block(packed_weight_block, 0, ${comp});
        weight_texels[0].y = extract_4bit_from_transposed_block(packed_weight_block, 1, ${comp});
        weight_texels[0].z = extract_4bit_from_transposed_block(packed_weight_block, 2, ${comp});
        weight_texels[0].w = extract_4bit_from_transposed_block(packed_weight_block, 3, ${comp});

        weight_texels[1].x = extract_4bit_from_transposed_block(packed_weight_block, 4, ${comp});
        weight_texels[1].y = extract_4bit_from_transposed_block(packed_weight_block, 5, ${comp});
        weight_texels[1].z = extract_4bit_from_transposed_block(packed_weight_block, 6, ${comp});
        weight_texels[1].w = extract_4bit_from_transposed_block(packed_weight_block, 7, ${comp});

        weight_texels[0] = fma(weight_texels[0], scales[0], zeros[0]);
        weight_texels[1] = fma(weight_texels[1], scales[1], zeros[1]);

        out_texels[0] = fma(in_texel_val[${comp}], weight_texels[0], out_texels[0]);
        out_texels[1] = fma(in_texel_val[${comp}], weight_texels[1], out_texels[1]);
      }
  }

  partial_sums[lid][0] = out_texels[0];
  partial_sums[lid][1] = out_texels[1];

  memoryBarrierShared();
  barrier();

  // Tree reduction to compute the overall result.
  for (int i = WGS / 2; i > 0; i /= 2) {
    if (lid < i) {
      partial_sums[lid][0] = partial_sums[lid][0] + partial_sums[lid + i][0];
      partial_sums[lid][1] = partial_sums[lid][1] + partial_sums[lid + i][1];
    }
    memoryBarrierShared();
    barrier();
  }

  // Only the first thread will write out result
  if (lid == 0) {
    out_texels[0] = partial_sums[0][0];
    out_texels[1] = partial_sums[0][1];

    uint n4 = DIV_4(n);
    write_output_texel_1d(out_texels[0], n4);
    if (n + 4 < output_sizes.x) {
      write_output_texel_1d(out_texels[1], n4 + 1);
    }
  }
}
