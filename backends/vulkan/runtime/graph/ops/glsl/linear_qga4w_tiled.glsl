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

$if IO_STORAGE == "buffer":
  #define BUFFER_IO
$if WEIGHT_STORAGE == "buffer":
  #define BUFFER_WEIGHT

#include "qlinear_utils.glslh"

void main() {
  // Each thread writes out a 8 wide x 4 high tile of output values
  const uint n8 = gl_GlobalInvocationID.x;
  const uint m4 = gl_GlobalInvocationID.y;

  const uint n = MUL_8(n8); // output col idx
  const uint m = MUL_4(m4); // output row idx
  const uint n4 = MUL_2(n8); // output col texel idx

  const uint group_num = input_sizes.x / group_size;
  const uint group_ntexels = DIV_UP_4(group_size);

  if (n >= output_sizes.x || m >= output_sizes.y) {
    return;
  }

  const uint K4 = DIV_UP_4(input_sizes.x);
  const uint N4 = DIV_UP_4(output_sizes.x); // number of texels in each row

  VEC4_T out_texels[4][2];
  // Initialize to 0
  $for row_i in range(4):
    $for col_i in range(2):
      out_texels[${row_i}][${col_i}] = VEC4_T(0.00);

  for (uint group_i = 0; group_i < group_num; ++group_i) {
    // Load quantization scales and zeros for the current group
    VEC4_T scales[2];
    VEC4_T zeros[2];
    {
      uint qparams_bufi = group_i * DIV_2(output_sizes.x) + DIV_2(n);

      VEC4_T scales_zeros_texels[4];
      $for comp in range(4):
        scales_zeros_texels[${comp}] = t_qparams[qparams_bufi++];

      scales[0] = VEC4_T(scales_zeros_texels[0].xz, scales_zeros_texels[1].xz);
      zeros[0] = VEC4_T(scales_zeros_texels[0].yw, scales_zeros_texels[1].yw);

      scales[1] = VEC4_T(scales_zeros_texels[2].xz, scales_zeros_texels[3].xz);
      zeros[1] = VEC4_T(scales_zeros_texels[2].yw, scales_zeros_texels[3].yw);
    }

    for (uint inner_k4 = 0; inner_k4 < group_ntexels; inner_k4++) {
      const uint k4 = group_i * group_ntexels + inner_k4;

      // Load 4x4 block of the input tensor, with the top left corner of the
      // block at (k, m)
      VEC4_T in_texels[4];
      $for comp in range(4):
        in_texels[${comp}] = load_input_texel_2d(k4, m + ${comp}, K4);

      uvec4 packed_weight_block = load_transposed_weight_block(k4, n8, K4);

      VEC4_T weight_texels[2];
      $for tile_k in range(4):
        // Process weight row k + comp
        {
          // Weight columns n + 0, 1, 2, 3
          weight_texels[0].x = extract_4bit_from_transposed_block(packed_weight_block, 0, ${tile_k});
          weight_texels[0].y = extract_4bit_from_transposed_block(packed_weight_block, 1, ${tile_k});
          weight_texels[0].z = extract_4bit_from_transposed_block(packed_weight_block, 2, ${tile_k});
          weight_texels[0].w = extract_4bit_from_transposed_block(packed_weight_block, 3, ${tile_k});

          // Weight colums n + 4, 5, 6, 7
          weight_texels[1].x = extract_4bit_from_transposed_block(packed_weight_block, 4, ${tile_k});
          weight_texels[1].y = extract_4bit_from_transposed_block(packed_weight_block, 5, ${tile_k});
          weight_texels[1].z = extract_4bit_from_transposed_block(packed_weight_block, 6, ${tile_k});
          weight_texels[1].w = extract_4bit_from_transposed_block(packed_weight_block, 7, ${tile_k});

          weight_texels[0] = fma(weight_texels[0], scales[0], zeros[0]);
          weight_texels[1] = fma(weight_texels[1], scales[1], zeros[1]);

          $for tile_m in range(4):
            out_texels[${tile_m}][0] = fma(VEC4_T(in_texels[${tile_m}][${tile_k}]), weight_texels[0], out_texels[${tile_m}][0]);
            out_texels[${tile_m}][1] = fma(VEC4_T(in_texels[${tile_m}][${tile_k}]), weight_texels[1], out_texels[${tile_m}][1]);
        }
    }
  }

  for (uint row_i = 0; row_i < 4 && m + row_i < output_sizes.y; ++row_i) {
    write_output_texel_2d(out_texels[row_i][0], n4,     m + row_i, N4);
    if (n + 4 < output_sizes.x) {
      write_output_texel_2d(out_texels[row_i][1], n4 + 1, m + row_i, N4);
    }
  }
}
