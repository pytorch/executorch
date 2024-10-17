/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}
#define FLOAT_T ${buffer_scalar_type(DTYPE)}

${define_active_storage_type(STORAGE)}

${define_required_extensions(DTYPE)}
${define_required_extensions("int8")}

#include "indexing_utils.h"

layout(std430) buffer;

${layout_declare_tensor(0, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(1, "r", "t_mat1", DTYPE, STORAGE)}
${layout_declare_tensor(2, "r", "t_qmat2", "int8", STORAGE)}
${layout_declare_tensor(3, "r", "t_scales", DTYPE, STORAGE)}

$if STORAGE == "buffer":
  ${layout_declare_ubo(4, "ivec4", "out_sizes")}
  ${layout_declare_ubo(5, "ivec4", "out_strides")}
  ${layout_declare_ubo(6, "int", "out_numel")}
  ${layout_declare_ubo(7, "ivec4", "mat1_sizes")}
  ${layout_declare_ubo(8, "ivec4", "mat1_strides")}
  ${layout_declare_ubo(9, "ivec4", "qmat2_strides")}
  ${layout_declare_ubo(10, "ivec4", "scales_strides")}
$else:
  ${layout_declare_ubo(4, "ivec3", "out_limits")}
  ${layout_declare_ubo(5, "ivec4", "mat1_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// This header file must be defined after the layout descriptors have been
// declared because the functions in the header assume some variables have been
// declared as layout descriptors.

#ifdef USING_BUFFER

#ifndef FLOAT_T
#define FLOAT_T float
#endif

FLOAT_T q_8w_linear(const ivec4 out_idx, const int K) {
  const FLOAT_T scale = t_scales[out_idx.x];

  FLOAT_T outval = FLOAT_T(0.0);

  // Initial mat1 tensor idx will be (0, out_idx.y, out_idx.z, 0)
  int mat1_offset = out_idx.y * mat1_strides.y + out_idx.z * qmat2_strides.z;
  // Initial qmat2 tensor idx wil be (0, out_idx.x, 0, 0); note that the qmat2
  // tensor is transposed
  int qmat2_offset = out_idx.x * qmat2_strides.y;

  // TODO(ssjia): optimize memory access pattern by traversing K in inner loop
  for (int i = 0; i < K; i++) {
    const FLOAT_T mat1_val = t_mat1[mat1_offset];
    const FLOAT_T mat2_val = t_qmat2[qmat2_offset] * scale;

    outval += mat1_val * mat2_val;

    mat1_offset++;
    qmat2_offset++;
  }

  return outval;
}

void main() {
  const int out_bufi = int(gl_GlobalInvocationID.x);
  if (out_bufi >= out_numel) {
    return;
  }

  const ivec4 out_tidx = bufi_to_tidx(out_bufi, out_strides, 0);

  t_out[out_bufi] = q_8w_linear(out_tidx, mat1_sizes.x);
}

#else // USING_TEXTURE

VEC4_T q_8w_linear(const ivec3 out_pos, const int K) {
  ivec3 mat1_pos = ivec3(0, out_pos.yz);
  ivec3 qmat2_pos = ivec3(0, out_pos.x * 4, 0);

  VEC4_T outtex = VEC4_T(0);

  const ivec3 scales_pos = ivec3(out_pos.x, 0, 0);
  const VEC4_T scales = load_texel(t_scales, scales_pos);

  for (int i = 0; i < K; i += 4) {
    const VEC4_T mat1_tex = load_texel(t_mat1, mat1_pos);

    const VEC4_T sums = VEC4_T(
        dot(mat1_tex, load_texel(t_qmat2, qmat2_pos) * scales.x),
        dot(mat1_tex,
            load_texel(t_qmat2, qmat2_pos + ivec3(0, 1, 0)) * scales.y),
        dot(mat1_tex,
            load_texel(t_qmat2, qmat2_pos + ivec3(0, 2, 0)) * scales.z),
        dot(mat1_tex,
            load_texel(t_qmat2, qmat2_pos + ivec3(0, 3, 0)) * scales.w));

    outtex += sums;

    mat1_pos.x++;
    qmat2_pos.x++;
  }

  return outtex;
}

void main() {
  const ivec3 out_pos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(out_pos, out_limits))) {
    return;
  }

  VEC4_T outtex = q_8w_linear(out_pos, mat1_sizes.x);
  write_texel(t_out, out_pos, outtex);
}

#endif
