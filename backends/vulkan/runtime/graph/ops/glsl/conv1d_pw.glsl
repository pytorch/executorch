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

${define_active_storage_type(STORAGE)}

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_in", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_weight", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer")}

${layout_declare_ubo(B, "BufferMetadata", "out_meta")}
${layout_declare_ubo(B, "BufferMetadata", "in_meta")}
${layout_declare_ubo(B, "ivec4", "weight_strides")}
${layout_declare_ubo(B, "int", "kernel_size", "int", "stride", "int", "padding", "int", "dilation", "int", "in_group_size", "int", "out_group_size")}
${layout_declare_ubo(B, "float", "out_min", "float", "out_max")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Computes a pointwise (kernel_size=1) 1D convolution over width-packed buffer
 * tensors. Each shader invocation computes one output element at (n, out_c, out_l).
 *
 * Since kernel_size=1 there is no spatial loop; only the channel reduction loop.
 */
void main() {
  const int out_l = int(gl_GlobalInvocationID.x);
  const int out_c = int(gl_GlobalInvocationID.y);
  const int n = int(gl_GlobalInvocationID.z);

  if (out_l >= int(size_at(out_meta, 0)) ||
      out_c >= int(size_at(out_meta, 1)) ||
      n >= int(size_at(out_meta, 2))) {
    return;
  }

  const int c_start = (out_c / out_group_size) * in_group_size;

  // Pointwise: kernel_size=1, k=0 always
  const int in_l = out_l * stride - padding;

  T sum = T(0);
  if (in_l >= 0 && in_l < int(size_at(in_meta, 0))) {
    for (int ic = 0; ic < in_group_size; ic++) {
      const int in_c = c_start + ic;
      TensorIndex4D in_tidx;
      in_tidx.data = ivec4(in_l, in_c, n, 0);
      const uint in_idx = tensor4d_idx_to_linear_idx(in_meta, in_tidx);
      // Weight tidx (0, ic, out_c) in [C_out, C_in/g, 1]: (0, ic, out_c, 0)
      const int w_idx = ic * weight_strides.y + out_c * weight_strides.z;
      sum += t_in[in_idx] * t_weight[w_idx];
    }
  }

  sum += T(t_bias[out_c]);

  TensorIndex4D out_tidx;
  out_tidx.data = ivec4(out_l, out_c, n, 0);
  const uint out_idx = tensor4d_idx_to_linear_idx(out_meta, out_tidx);
  t_out[out_idx] = op(sum, T(out_min), T(out_max));
}
