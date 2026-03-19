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
#define T ${texel_load_component_type(DTYPE, STORAGE)}

$if STORAGE == "buffer":
  #define BUFFER
$if HAS_BIAS:
  #define HAS_BIAS

${define_required_extensions(STORAGE, DTYPE)}

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight", DTYPE, STORAGE, is_scalar_array=False)}
$if HAS_BIAS:
  ${layout_declare_tensor(B, "r", "t_bias", DTYPE, STORAGE, is_scalar_array=False)}

// in_sizes: {L_in, C, N, 1} in WHCN order
${layout_declare_ubo(B, "ivec4", "in_sizes")}
// out_sizes: {L_out, C, N, 1} in WHCN order
${layout_declare_ubo(B, "ivec4", "out_sizes")}

layout(push_constant) uniform restrict Block {
  int kernel_size;
  int stride;
  int padding;
  int dilation;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Thread mapping: X = C/4, Y = L_out, Z = N
// Each thread computes 4 output channels at one spatial position.
// Depthwise: each channel has its own filter, so 4 channels can be computed
// independently with element-wise vec4 FMA.

void main() {
  const int c4 = int(gl_GlobalInvocationID.x);
  const int l_out = int(gl_GlobalInvocationID.y);
  const int n = int(gl_GlobalInvocationID.z);

  const int L_in = in_sizes.x;
  const int C = in_sizes.y;
  const int C4 = div_up_4(C);
  const int L_out = out_sizes.x;

  if (c4 >= C4 || l_out >= L_out) {
    return;
  }

  VEC4_T sum = VEC4_T(0);

  for (int k = 0; k < kernel_size; k++) {
    const int l_in = l_out * stride - padding + k * dilation;
    if (l_in >= 0 && l_in < L_in) {
#ifdef BUFFER
      const VEC4_T in_val = t_in[(n * L_in + l_in) * C4 + c4];
      const VEC4_T w_val = t_weight[k * C4 + c4];
#else
      const VEC4_T in_val = texelFetch(t_in, ivec3(l_in, c4, n), 0);
      const VEC4_T w_val = texelFetch(t_weight, ivec3(k, 0, c4), 0);
#endif
      sum = fma(w_val, in_val, sum);
    }
  }

#ifdef HAS_BIAS
#ifdef BUFFER
  sum += t_bias[c4];
#else
  sum += texelFetch(t_bias, ivec3(c4, 0, 0), 0);
#endif
#endif

#ifdef BUFFER
  t_out[(n * L_out + l_out) * C4 + c4] = sum;
#else
  imageStore(t_out, ivec3(l_out, c4, n), sum);
#endif
}
