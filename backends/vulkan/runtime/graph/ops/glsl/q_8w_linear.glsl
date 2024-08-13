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
#include "q_linear.h"

#ifdef USING_BUFFER

void main() {
  const int t_id = int(gl_GlobalInvocationID.x);
  if (t_id >= out_numel) {
    return;
  }

  const ivec4 out_idx = to_tensor_idx(t_id, out_strides, 0);

  t_out[t_id] = q_8w_linear(out_idx, mat1_sizes.x);
}

#else // USING_TEXTURE

void main() {
  const ivec3 out_pos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(out_pos, out_limits))) {
    return;
  }

  VEC4_T outtex = q_8w_linear(out_pos, mat1_sizes.x);
  write_texel(t_out, out_pos, outtex);
}

#endif
