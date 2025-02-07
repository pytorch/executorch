/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION highp

layout(std430) buffer;

${layout_declare_tensor(0, "w", "t_out", "float", "buffer")}
${layout_declare_tensor(1, "r", "t_mat1", "float", "buffer")}
${layout_declare_tensor(2, "r", "t_mat2", "float", "buffer")}
${layout_declare_ubo(3, "ivec4", "out_sizes")}
${layout_declare_ubo(4, "ivec4", "out_strides")}
${layout_declare_ubo(5, "ivec4", "mat1_sizes")}
${layout_declare_ubo(6, "ivec4", "mat1_strides")}
${layout_declare_ubo(7, "ivec4", "mat2_sizes")}
${layout_declare_ubo(8, "ivec4", "mat2_strides")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec2 out_idx = ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);
  if (any(greaterThanEqual(out_idx, out_sizes.xy))) {
    return;
  }

  // Initial idx for mat1 is (0, out_idx.y)
  int mat1_id = out_idx.y * mat1_strides.y;
  // Initial idx for mat2 is (out_idx.x, 0)
  int mat2_id = out_idx.x * mat2_strides.x;

  float sum = 0.0;
  for (int i = 0; i < mat1_sizes.x; ++i) {
    sum += t_mat1[mat1_id] * t_mat2[mat2_id];

    mat1_id += mat1_strides.x;
    mat2_id += mat2_strides.y;
  }

  const int out_id = out_idx.x * out_strides.x + out_idx.y * out_strides.y;
  t_out[out_id] = sum;
}
