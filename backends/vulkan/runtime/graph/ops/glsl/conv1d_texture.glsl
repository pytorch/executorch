/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_type(DTYPE)}

#define op(X, A, B) ${OPERATOR}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "kernel_in", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "bias_in", DTYPE, STORAGE)}

${layout_declare_ubo(B, "ivec3", "out_limits")}
${layout_declare_ubo(B, "ivec4", "in_sizes")}

${layout_declare_ubo(B,"int", "kernel_size", "int", "stride", "int", "padding", "int", "dilation", "int", "in_group_size", "int", "out_group_size")}

${layout_declare_ubo(B, "float", "out_min", "float", "out_max")}

#include "indexing_utils.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "out_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 out_axis_map = unhash_axis_map(out_layout);

${layout_declare_spec_const(C, "int", "in_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 in_axis_map = unhash_axis_map(in_layout);

${layout_declare_spec_const(C, "int", "kernel_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 kernel_axis_map = unhash_axis_map(kernel_layout);

${layout_declare_spec_const(C, "int", "bias_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 bias_axis_map = unhash_axis_map(bias_layout);

void main() {
  const ivec3 lpos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(lpos, out_limits))) {
    return;
  }

  const int out_c = lpos.y;

  const int c_start = (out_c / out_group_size) * in_group_size;
  const int c_end = c_start + in_group_size;

  const int out_l = lpos.x;

  const int N = lpos.z;

  const int in_l = out_l * stride - padding;
  VEC4_T sum = VEC4_T(0);

  const int out_c_packed_index = out_c >> 2;
  const int out_c_packed_lane = out_c & 0x3;

  for (int in_c = c_start; in_c < c_end; ++in_c) {
    for (int k = 0; k < kernel_size; k++) {
      const ivec3 w_lpos = ivec3(k, in_c % in_group_size, out_c_packed_index);
      const VEC4_T weight_texel = load_texel_lpos(kernel_in, w_lpos, kernel_axis_map);
      VEC4_T weight = VEC4_T(weight_texel[out_c_packed_lane]);

      const ivec3 in_pos = lpos_to_pos(ivec3(in_l + k * dilation, in_c, N), in_axis_map);
      sum = fma(weight, load_texel(t_in, in_pos), sum);
    }
  }

  const VEC4_T bias = load_texel_lpos(bias_in, ivec3(out_c_packed_index, 0, 0), bias_axis_map);
  const ivec3 out_lpos = ivec3(out_l, out_c, N);
  write_texel_lpos(t_out, out_lpos, op(sum + bias[out_c_packed_lane], out_min, out_max), out_axis_map);
}
