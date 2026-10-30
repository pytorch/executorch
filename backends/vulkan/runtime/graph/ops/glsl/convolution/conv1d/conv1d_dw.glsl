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
  #define SCALAR_BUFFER
$if HAS_BIAS:
  #define HAS_BIAS

${define_required_extensions(STORAGE, DTYPE)}

layout(std430) buffer;

#include "common.glslh"

$if STORAGE == "buffer":
  ${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE, is_scalar_array=True)}
  ${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE, is_scalar_array=True)}
  ${layout_declare_tensor(B, "r", "t_weight", DTYPE, STORAGE, is_scalar_array=True)}
$else:
  ${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE, is_scalar_array=False)}
  ${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE, is_scalar_array=False)}
  ${layout_declare_tensor(B, "r", "t_weight", DTYPE, STORAGE, is_scalar_array=False)}
$if HAS_BIAS:
  $if STORAGE == "buffer":
    ${layout_declare_tensor(B, "r", "t_bias", DTYPE, STORAGE, is_scalar_array=True)}
  $else:
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
  float output_min;
  float output_max;
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
      const int in_base = (n * L_in + l_in) * C + c4 * 4;
      T in_s0 = t_in[in_base];
      T in_s1 = (c4 * 4 + 1 < C) ? t_in[in_base + 1] : T(0);
      T in_s2 = (c4 * 4 + 2 < C) ? t_in[in_base + 2] : T(0);
      T in_s3 = (c4 * 4 + 3 < C) ? t_in[in_base + 3] : T(0);
      const VEC4_T in_val = VEC4_T(in_s0, in_s1, in_s2, in_s3);

      const int w_base = k * C + c4 * 4;
      T w_s0 = t_weight[w_base];
      T w_s1 = (c4 * 4 + 1 < C) ? t_weight[w_base + 1] : T(0);
      T w_s2 = (c4 * 4 + 2 < C) ? t_weight[w_base + 2] : T(0);
      T w_s3 = (c4 * 4 + 3 < C) ? t_weight[w_base + 3] : T(0);
      const VEC4_T w_val = VEC4_T(w_s0, w_s1, w_s2, w_s3);
#else
      const VEC4_T in_val = texelFetch(t_in, ivec3(l_in, c4, n), 0);
      const VEC4_T w_val = texelFetch(t_weight, ivec3(k, 0, c4), 0);
#endif
      sum = fma(w_val, in_val, sum);
    }
  }

#ifdef HAS_BIAS
#ifdef BUFFER
  const int bias_base = c4 * 4;
  T b0 = t_bias[bias_base];
  T b1 = (bias_base + 1 < C) ? t_bias[bias_base + 1] : T(0);
  T b2 = (bias_base + 2 < C) ? t_bias[bias_base + 2] : T(0);
  T b3 = (bias_base + 3 < C) ? t_bias[bias_base + 3] : T(0);
  sum += VEC4_T(b0, b1, b2, b3);
#else
  sum += texelFetch(t_bias, ivec3(c4, 0, 0), 0);
#endif
#endif

  sum = clamp(sum, VEC4_T(output_min), VEC4_T(output_max));

#ifdef BUFFER
  const int out_base = (n * L_out + l_out) * C + c4 * 4;
  t_out[out_base] = sum.x;
  if (c4 * 4 + 1 < C) t_out[out_base + 1] = sum.y;
  if (c4 * 4 + 2 < C) t_out[out_base + 2] = sum.z;
  if (c4 * 4 + 3 < C) t_out[out_base + 3] = sum.w;
#else
  imageStore(t_out, ivec3(l_out, c4, n), sum);
#endif
}
