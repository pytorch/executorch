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

#include "indexing_utils.h"

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "kernel_in", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "bias_in", DTYPE, STORAGE)}

${layout_declare_ubo(B, "ivec3", "out_limits")}
${layout_declare_ubo(B, "ivec4", "in_sizes")}

${layout_declare_ubo(B, "ivec4", "out_axis_map")}
${layout_declare_ubo(B, "ivec4", "in_axis_map")}
${layout_declare_ubo(B, "ivec4", "kernel_axis_map")}
${layout_declare_ubo(B, "ivec4", "bias_axis_map")}

${layout_declare_ubo(B,"int", "kernel_size", "int", "stride", "int", "padding", "int", "dilation", "int", "in_group_size", "int", "out_group_size")}

${layout_declare_ubo(B, "float", "out_min", "float", "out_max")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Let us define
//
// input = (N, in_C, in_L),
// output = (N, out_C, out_L),
// groups = G,
// kernel = K,
//
// which results in shapes
//
// weight = (out_C, in_C / G, K),
// bias = (out_C,).
//
// This implementation performs out_C shader invocations, where each invocation
// calculates the rolling kernel of the length dimension for each batch, i.e.,
// computes out_L * N results.
//
// Note that we can rewrite this implementation as out_L * out_C * ceil(N / 4)
// shader invocations, where each invocation computes 1 result. But that
// performs worse.
void main() {
  const ivec3 lpos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(lpos, out_limits))) {
    return;
  }

  int in_length = in_sizes.x;
  int batch_size = in_sizes.z;

  // "out_c" is the output's channel index where we write our result.
  // Across shader invocations, this is the only value that varies.
  int out_c = lpos.y;
  VEC4_T bias = load_texel_lpos(bias_in, ivec3(out_c, 0, 0), bias_axis_map);

  // "in_c" tracks the input's channel start index.
  // We iterate over the input group that corresponds to the output group.
  int c_start = (out_c / out_group_size) * in_group_size;
  int c_end = c_start + in_group_size;

  // "in_l" tracks the input's length start index for our input-kernel overlay
  // region.
  int l_start = -padding;
  int l_end = in_length + padding - dilation * (kernel_size - 1);

  // Since the input/output tensors are channel-packed, which is along the
  // batch dimension, we can batch-read/write four elements at a time.
  for (int n = 0; n < batch_size; n += 4) {
    // "out_l" tracks the output's length index where we write our result.
    int out_l = 0;

    for (int in_l = l_start; in_l < l_end; in_l += stride, ++out_l) {
      VEC4_T sum = VEC4_T(0);

      for (int in_c = c_start; in_c < c_end; ++in_c) {
        // "k" tracks the kernel's index for our input-kernel computation.
        // It reads out-of-bound zeros, but trying to avoid them complicates
        // for-loop conditions, which results in worse performance.
        for (int k = 0; k < kernel_size; k += 4) {
          // Since the weight tensor is width-packed, which is along the length
          // dimension, we can batch-read four elements at a time.
          const ivec3 w_lpos = ivec3(k / 4, in_c % in_group_size, out_c);
          const VEC4_T weight = load_texel_lpos(kernel_in, w_lpos, kernel_axis_map);

          ivec3 in_pos = lpos_to_pos(ivec3(in_l + k * dilation, in_c, n / 4), in_axis_map);
          sum = fma(weight.xxxx, load_texel(t_in, in_pos), sum);

          in_pos[in_axis_map.x] += dilation;
          sum = fma(weight.yyyy, load_texel(t_in, in_pos), sum);

          in_pos[in_axis_map.x] += dilation;
          sum = fma(weight.zzzz, load_texel(t_in, in_pos), sum);

          in_pos[in_axis_map.x] += dilation;
          sum = fma(weight.wwww, load_texel(t_in, in_pos), sum);
        }
      }

      const ivec3 out_lpos = ivec3(out_l, out_c, n / 4);
      write_texel_lpos(t_out, out_lpos, op(sum + bias.x, out_min, out_max), out_axis_map);
    }
  }
}
