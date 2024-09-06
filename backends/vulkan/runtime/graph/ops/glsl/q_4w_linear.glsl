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

${define_required_extensions(DTYPE)}
${define_required_extensions("int8")}

layout(std430) buffer;

${layout_declare_tensor(0, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(1, "r", "t_mat1", DTYPE, STORAGE)}
${layout_declare_tensor(2, "r", "t_mat2", "int8", STORAGE)}
${layout_declare_tensor(3, "r", "t_scales_and_zeros", DTYPE, STORAGE)}

$if STORAGE == "texture3d":
  ${layout_declare_ubo(4, "ivec4", "out_sizes")}
  ${layout_declare_ubo(5, "ivec4", "mat1_sizes")}
  ${layout_declare_ubo(6, "ivec4", "scales_strides")}
$else:
  ${layout_declare_ubo(4, "ivec4", "out_sizes")}
  ${layout_declare_ubo(5, "ivec4", "out_strides")}
  ${layout_declare_ubo(6, "ivec4", "mat1_sizes")}
  ${layout_declare_ubo(7, "ivec4", "mat1_strides")}
  ${layout_declare_ubo(8, "ivec4", "mat2_strides")}
  ${layout_declare_ubo(9, "ivec4", "scales_strides")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int group_size = 1;

void main() {

    const ivec4 out_pos = ivec4(
      gl_GlobalInvocationID.x, // n = 0..N-1
      gl_GlobalInvocationID.y, // m = 0..M-1
      gl_GlobalInvocationID.z % out_sizes.z,
      gl_GlobalInvocationID.z / out_sizes.z);

    if (any(greaterThanEqual(out_pos, out_sizes))) {
      return;
    }

    const uint K = mat1_sizes.x;
    const uint n = out_pos.x;
    const uint m = out_pos.y;
    const uint mask = uint(0x0f);

    float rc = 0.0;
    int k = 0;

    #ifdef USING_BUFFER
      const uint k_block = (K + group_size - 1) / group_size;
      ivec4 mat1_pos = ivec4(0, m, out_pos.z, out_pos.w);
      ivec4 mat2_pos = ivec4(0, n, out_pos.z, out_pos.w);
      ivec4 scale_pos = ivec4(0, n, 0, out_pos.w);
      ivec4 zero_pos = ivec4(0, n, 1, out_pos.w);

      for (int kb = 0; kb < k_block; kb++) {
        scale_pos.x = kb;
        const int scale_id = to_buffer_id(scale_pos, scales_strides);
        const float scale = float(t_scales_and_zeros[scale_id]);

        zero_pos.x = kb;
        const int zero_id = to_buffer_id(zero_pos, scales_strides);
        const float zero = float(t_scales_and_zeros[zero_id]) - scale * 8.0;

        for(uint idx = 0; idx < group_size && k < K; idx++, k++) {
          mat1_pos.x = k;
          const int mat1_id = to_buffer_id(mat1_pos, mat1_strides);
          const float mat1_val = float(t_mat1[mat1_id]);

          mat2_pos.x = k / 2;
          const int mat2_id = to_buffer_id(mat2_pos, mat2_strides);
          // Bitwise op treats sign bit from int8 as a value bit instead,
          // since there is no uint8_t datatype
          uint mat2_val = (t_mat2[mat2_id] & 0xFF);
          mat2_val = (k & 1) == 0 ? mat2_val & mask : (mat2_val >> 4);

          rc += mat1_val * (scale * float(mat2_val) + zero);
        }
      }

      const int out_id = to_buffer_id(out_pos, out_strides);
      t_out[out_id] = FLOAT_T(rc);

    #else // Using texture
      const uint texel_group_size = group_size / FOUR;
      const uint k_block = (K + texel_group_size - 1) / texel_group_size;
      ivec3 mat1_pos = ivec3(0, m, out_pos.z);
      ivec3 mat2_pos = ivec3(0, n, out_pos.z);
      ivec3 scale_pos = ivec3(0, n, 0);
      ivec3 zero_pos = ivec3(0, n, 1);

      for (int kb = 0; kb < k_block; kb++) {
        const int texel_kb = kb / FOUR;
        const int kb_offset = kb % FOUR;

        scale_pos.x = texel_kb;
        const VEC4_T scale_texel = load_texel(t_scales_and_zeros, scale_pos);
        const float scale = float(scale_texel[kb_offset]);

        zero_pos.x = texel_kb;
        const VEC4_T zero_texel = load_texel(t_scales_and_zeros, zero_pos);
        const float zero = float(zero_texel[kb_offset]) - scale * 8.0;

        for(uint idx = 0; idx < texel_group_size && k < K; idx++, k++) {
          mat1_pos.x = k;
          const VEC4_T mat1_tex = load_texel(t_mat1, mat1_pos);

          mat2_pos.x = k / 2;
          const i8vec4 mat2_tex = i8vec4(load_texel(t_mat2, mat2_pos));

          // Every two texels of mat1 correspond to one texel of mat2
          // Even mat1 indeces correspond to first half of mat2 texel and
          // odd indeces correspond to second half
          const int mat2_offset = (k & 1) == 0 ? 0 : 2;
          for (int texel_idx = 0; texel_idx < FOUR; texel_idx++){
            // Bitwise op treats sign bit from int8 as a value bit instead,
            // since there is no uint8_t datatype
            uint mat2_val = (mat2_tex[mat2_offset + texel_idx / 2] & 0xFF);
            mat2_val = (texel_idx & 1) == 0 ? mat2_val & mask : (mat2_val >> 4);
            rc += mat1_tex[texel_idx] * (scale * float(mat2_val) + zero);
          }
        }
      }
      write_texel(t_out, out_pos.xyz, vec4(rc, 0, 0, 0));

    #endif
}
