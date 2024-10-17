/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#include "indexing_utils.h"

#define PRECISION ${PRECISION}

#define FOUR 4

#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}
#define FLOAT_T ${buffer_scalar_type(DTYPE)}

${define_active_storage_type(STORAGE)}

${define_required_extensions([DTYPE, "uint8", "uint16"])}
#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

${layout_declare_tensor(B, "w", "ret", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "x", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "weights", "uint8", "buffer")}
${layout_declare_tensor(B, "r", "qparams", DTYPE, STORAGE)}
${layout_declare_ubo(B, "ivec3", "ret_limits")}
${layout_declare_ubo(B, "ivec4", "x_sizes")}
${layout_declare_ubo(B, "ivec4", "weights_strides")}
${layout_declare_ubo(B, "ivec4", "qparams_strides")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int group_size = 1;

/*
 * This shader computes a linear operator between a floating point input matrix
 * x and a weights matrix that is quantized to 4 bits.
 *
 * The (W, H, C) shape of each tensor is:
 * - x: (K, M)
 * - weights: (K / 2, N)
 *   - The weights tensor has a data type of `uint8`. Each element in the tensor
 *     contains 2 4-bit values packed into a uint8.
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
  // output positions being calculated are (n, m), (n + 1, m), ...
  // This means multiplying the m-th row of x with the n-th, (n+1)-th, ... rows
  // of the weights tensor.
  const u16vec3 ret_pos = u16vec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(ret_pos, ret_limits))) {
    return;
  }

  // Since ret is width packed, need to multiply by 4
  const uint16_t n = uint16_t(ret_pos.x * 4);

  // K is guaranteed to be a multiple of group size
  const uint16_t num_blocks = uint16_t(x_sizes.x / group_size);

  uint16_t k_texel_i = uint16_t(0);
  vec4 sums = vec4(0.0);
  for (uint16_t block_idx = uint16_t(0); block_idx < num_blocks; block_idx++) {
    vec4 scales;
    vec4 zeros;

    [[unroll]] for (int comp = 0; comp < 4; comp++) {
      const vec4 scale_and_zero = load_texel(
          qparams, u16vec3(0, n + comp, block_idx));
      scales[comp] = scale_and_zero.x;
      zeros[comp] = scale_and_zero.y;
    }

    for (uint16_t i = uint16_t(0); i < group_size; i += uint16_t(4), k_texel_i++) {
      const VEC4_T x_texel = load_texel(
          x, u16vec3(k_texel_i, ret_pos.y, ret_pos.z));

      [[unroll]] for (int comp = 0; comp < 4; comp++) {
        const int weights_bufi = (n + comp) * weights_strides.y + (k_texel_i * 2);
        // Need to read 4 unpacked values, which corresponds to 2 packed values
        const uint8_t weights_val_1 = weights[weights_bufi];
        const uint8_t weights_val_2 = weights[weights_bufi + 1];

        const u8vec4 weights_texel = u8vec4(
          (weights_val_1 & 0xF0) >> 4,
          weights_val_1 & 0x0F,
          (weights_val_2 & 0xF0) >> 4,
          weights_val_2 & 0x0F);

        // Note that the unpacked 4-bit values are unsigned, therefore they must
        // first be "centered" around 0 by subtracting 8 before applying the
        // scale and zero point.
        sums[comp] += dot(
            x_texel, (vec4(weights_texel) - 8.0) * scales[comp] + zeros[comp]);
      }
    }
  }
  write_texel(ret, ret_pos, sums);
}
