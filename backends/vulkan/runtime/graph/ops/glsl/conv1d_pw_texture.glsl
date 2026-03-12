/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions("texture3d", DTYPE)}
${define_required_extensions("buffer", DTYPE)}

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_type(DTYPE)}
#define T ${buffer_scalar_type(DTYPE)}

#define op(X, A, B) ${OPERATOR}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_in", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_weight", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer")}

${layout_declare_ubo(B, "ivec3", "out_limits")}
${layout_declare_ubo(B, "int", "in_group_size", "int", "out_group_size")}
${layout_declare_ubo(B, "float", "out_min", "float", "out_max")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Pointwise (kernel_size=1) 1D convolution for width-packed texture3d tensors.
 *
 * Each invocation computes one output texel containing 4 adjacent width
 * positions. The reduction loops over input channels, multiplying a scalar
 * weight by the full 4-wide input texel.
 *
 * Tensor layout: [N, C, L] stored as width-packed texture3d where
 *   texture x = L / 4 (packed width), y = C, z = N.
 *
 * Weight is [C_out, C_in/groups, 1] stored as a contiguous buffer with
 * row-major layout: weight[c_out, ic] = t_weight[c_out * in_group_size + ic].
 */
void main() {
  const int out_l = int(gl_GlobalInvocationID.x);
  const int out_c = int(gl_GlobalInvocationID.y);
  const int n = int(gl_GlobalInvocationID.z);

  if (out_l >= out_limits.x || out_c >= out_limits.y || n >= out_limits.z) {
    return;
  }

  const int c_start = (out_c / out_group_size) * in_group_size;
  const int w_base = out_c * in_group_size;

  VEC4_T sum = VEC4_T(0);
  for (int ic = 0; ic < in_group_size; ic++) {
    const VEC4_T in_texel = texelFetch(t_in, ivec3(out_l, c_start + ic, n), 0);
    const T w = t_weight[w_base + ic];
    sum = fma(VEC4_T(w), in_texel, sum);
  }

  sum += VEC4_T(T(t_bias[out_c]));
  imageStore(t_out, ivec3(out_l, out_c, n), op(sum, VEC4_T(out_min), VEC4_T(out_max)));
}
