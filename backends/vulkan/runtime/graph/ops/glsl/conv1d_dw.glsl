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

#define T ${buffer_scalar_type(DTYPE)}

#define op(X, A, B) ${OPERATOR}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_in", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_weight", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer")}

${layout_declare_ubo(B, "ivec4", "out_sizes")}
${layout_declare_ubo(B, "ivec4", "out_strides")}
${layout_declare_ubo(B, "ivec4", "in_sizes")}
${layout_declare_ubo(B, "ivec4", "in_strides")}
${layout_declare_ubo(B, "ivec4", "weight_strides")}
${layout_declare_ubo(B, "int", "kernel_size", "int", "stride", "int", "padding", "int", "dilation", "int", "in_group_size", "int", "out_group_size")}
${layout_declare_ubo(B, "float", "out_min", "float", "out_max")}

#include "indexing_utils.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Computes a depthwise 1D convolution over width-packed buffer tensors. Each
 * shader invocation computes one output element at position (n, c, out_l).
 *
 * For depthwise conv: groups == C_in == C_out, so each output channel uses
 * exactly one input channel. Weight shape is [C, 1, K].
 */
void main() {
  const int out_l = int(gl_GlobalInvocationID.x);
  const int c = int(gl_GlobalInvocationID.y);
  const int n = int(gl_GlobalInvocationID.z);

  // WHCN sizes for [N, C, L]: (L, C, N, 1) -> sizes.y=C, sizes.z=N
  if (out_l >= out_sizes.x || c >= out_sizes.y || n >= out_sizes.z) {
    return;
  }

  T sum = T(0);
  for (int k = 0; k < kernel_size; k++) {
    const int in_l = out_l * stride - padding + k * dilation;
    if (in_l >= 0 && in_l < in_sizes.x) {
      // WHCN tidx for (n, c, in_l) in [N, C, L] tensor: (in_l, c, n, 0)
      const int in_idx = tidx_to_bufi(ivec4(in_l, c, n, 0), in_strides);
      // WHCN tidx for weight (k, 0, c) in [C, 1, K]: (k, 0, c, 0)
      const int w_idx = tidx_to_bufi(ivec4(k, 0, c, 0), weight_strides);
      sum += t_in[in_idx] * t_weight[w_idx];
    }
  }

  sum += T(t_bias[c]);

  // WHCN tidx for (n, c, out_l): (out_l, c, n, 0)
  const int out_idx = tidx_to_bufi(ivec4(out_l, c, n, 0), out_strides);
  t_out[out_idx] = op(sum, T(out_min), T(out_max));
}
