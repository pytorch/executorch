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

#define FLOAT_T ${buffer_scalar_type(DTYPE)}

${define_active_storage_type(STORAGE)}

${define_required_extensions(DTYPE)}
${define_required_extensions("int8")}

layout(std430) buffer;

${layout_declare_tensor(0, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(1, "r", "t_mat1", DTYPE, STORAGE)}
${layout_declare_tensor(2, "r", "t_mat2", "int8", STORAGE)}
${layout_declare_tensor(3, "r", "t_scales_and_zeros", DTYPE, STORAGE)}

${layout_declare_ubo(4, "ivec4", "out_sizes")}
${layout_declare_ubo(5, "ivec4", "out_strides")}
${layout_declare_ubo(6, "ivec4", "mat1_strides")}
${layout_declare_ubo(7, "ivec4", "mat2_sizes")}
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

    const uint K = mat2_sizes.x * 2;
    const uint N = mat2_sizes.y;
    const uint n = out_pos.x;
    const uint m = out_pos.y;
    const uint k_block = (K + group_size - 1) / group_size;
    const uint mask = uint(0x0f);
    ivec4 mat1_pos = ivec4(0, m, out_pos.z, out_pos.w);
    ivec4 mat2_pos = ivec4(0, n, out_pos.z, out_pos.w);
    ivec4 scale_pos = ivec4(0, n, 0, out_pos.w);
    ivec4 zero_pos = ivec4(0, n, 1, out_pos.w);

    float rc = 0.0;
    int k = 0;

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
}
