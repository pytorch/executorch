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

${define_required_extensions(DTYPE)}

layout(std430) buffer;

${layout_declare_tensor(0, "w", "t_out", DTYPE, "buffer")}
${layout_declare_tensor(1, "r", "t_mat1", DTYPE, "buffer")}
${layout_declare_tensor(2, "r", "t_mat2", DTYPE, "buffer")}
${layout_declare_ubo(3, "ivec4", "out_sizes")}
${layout_declare_ubo(4, "ivec4", "out_strides")}
${layout_declare_ubo(5, "ivec4", "mat1_sizes")}
${layout_declare_ubo(6, "ivec4", "mat1_strides")}
${layout_declare_ubo(7, "ivec4", "mat2_sizes")}
${layout_declare_ubo(8, "ivec4", "mat2_strides")}
${layout_declare_ubo(9, "int", "out_numel")}

#include "indexing_utils.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec4 out_idx = ivec4(
      gl_GlobalInvocationID.x,
      gl_GlobalInvocationID.y,
      gl_GlobalInvocationID.z % out_sizes.z,
      gl_GlobalInvocationID.z / out_sizes.z);

  if (any(greaterThanEqual(out_idx, out_sizes))) {
    return;
  }

  int mat1_id = to_buffer_id(
      ivec4(0, out_idx.y, out_idx.z, out_idx.w), mat1_strides);
  int mat2_id = to_buffer_id(
      ivec4(out_idx.x, 0, out_idx.z, out_idx.w), mat2_strides);

  T sum = T(0.0);
  for (int i = 0; i < mat1_sizes.x; ++i) {
    sum += t_mat1[mat1_id] * t_mat2[mat2_id];

    mat1_id += mat1_strides.x;
    mat2_id += mat2_strides.y;
  }

  const int out_id = to_buffer_id(out_idx, out_strides);
  t_out[out_id] = T(sum);
}
