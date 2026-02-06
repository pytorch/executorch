/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions("buffer", DTYPE)}

#define PRECISION ${PRECISION}

$if HAS_BIAS:
  #define HAS_BIAS

#define T ${buffer_scalar_type(DTYPE)}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_mat1", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_mat2", DTYPE, "buffer")}
$if HAS_BIAS:
  ${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer")}
${layout_declare_ubo(B, "ivec4", "out_sizes")}
${layout_declare_ubo(B, "ivec4", "out_strides")}
${layout_declare_ubo(B, "ivec4", "mat1_sizes")}
${layout_declare_ubo(B, "ivec4", "mat1_strides")}
${layout_declare_ubo(B, "ivec4", "mat2_sizes")}
${layout_declare_ubo(B, "ivec4", "mat2_strides")}
${layout_declare_ubo(B, "int", "out_numel")}
$if HAS_BIAS:
  ${layout_declare_ubo(B, "float", "alpha", "float", "beta")}

#include "indexing_utils.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "mat2_is_transposed", "0")}

void main() {
  const ivec4 out_tidx = ivec4(
      gl_GlobalInvocationID.x,
      gl_GlobalInvocationID.y,
      gl_GlobalInvocationID.z % out_sizes.z,
      gl_GlobalInvocationID.z / out_sizes.z);

  if (any(greaterThanEqual(out_tidx, out_sizes))) {
    return;
  }

  int mat1_bufi = tidx_to_bufi(
      ivec4(0, out_tidx.y, out_tidx.z, out_tidx.w), mat1_strides);
  int mat2_bufi;
  if (mat2_is_transposed > 0) {
    mat2_bufi = tidx_to_bufi(
        ivec4(0, out_tidx.x, 0, 0), mat2_strides);
  } else {
    mat2_bufi = tidx_to_bufi(
        ivec4(out_tidx.x, 0, out_tidx.z, out_tidx.w), mat2_strides);
  }

  int mat2_stride;
  if (mat2_is_transposed > 0) {
    mat2_stride = mat2_strides.x;
  } else {
    mat2_stride = mat2_strides.y;
  }

  T sum = T(0.0);
  for (int i = 0; i < mat1_sizes.x; ++i) {
    sum += t_mat1[mat1_bufi] * t_mat2[mat2_bufi];

    mat1_bufi += mat1_strides.x;
    mat2_bufi += mat2_stride;
  }

  const int out_bufi = tidx_to_bufi(out_tidx, out_strides);
#ifdef HAS_BIAS
  t_out[out_bufi] = T(alpha) * T(sum) + T(beta) * t_bias[out_tidx.x];
#else
  t_out[out_bufi] = T(sum);
#endif // HAS_BIAS
}
