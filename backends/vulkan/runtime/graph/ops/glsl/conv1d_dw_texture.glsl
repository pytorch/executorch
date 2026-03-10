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
${layout_declare_ubo(B, "int", "kernel_size", "int", "stride", "int", "padding", "int", "dilation")}
${layout_declare_ubo(B, "int", "in_length")}
${layout_declare_ubo(B, "float", "out_min", "float", "out_max")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Depthwise 1D convolution for width-packed TEXTURE_3D tensors.
 *
 * Each invocation computes one output texel which contains up to 4 adjacent
 * width (length) positions for a single channel c at batch n.
 *
 * Tensor layout: [N, C, L] stored as width-packed texture3d where
 *   texture x = L / 4 (packed width texel index), y = C, z = N.
 *
 * For depthwise conv: groups == C_in == C_out, so each output channel c reads
 * only from input channel c. Weight shape is [C, 1, K] stored as a contiguous
 * buffer: t_weight[c * kernel_size + k].
 */
void main() {
  const int out_l = int(gl_GlobalInvocationID.x);
  const int out_c = int(gl_GlobalInvocationID.y);
  const int n = int(gl_GlobalInvocationID.z);

  if (out_l >= out_limits.x || out_c >= out_limits.y || n >= out_limits.z) {
    return;
  }

  // out_l is a texel index, each texel holds 4 width positions.
  // The 4 logical output positions are: base_l, base_l+1, base_l+2, base_l+3.
  const int base_out_l = out_l * 4;
  const int w_base = out_c * kernel_size;

  VEC4_T sum = VEC4_T(0);

  for (int k = 0; k < kernel_size; k++) {
    const T w = t_weight[w_base + k];

    // For each of the 4 packed width lanes, compute the input position and
    // accumulate. All 4 lanes share the same weight scalar w.
    const ivec4 in_l = ivec4(
        base_out_l * stride - padding + k * dilation,
        (base_out_l + 1) * stride - padding + k * dilation,
        (base_out_l + 2) * stride - padding + k * dilation,
        (base_out_l + 3) * stride - padding + k * dilation);

    // Each lane reads from a potentially different input texel.
    const ivec4 in_texel_idx = in_l >> 2; // divide by 4
    const ivec4 in_lane = in_l & 3;      // mod 4

    for (int lane = 0; lane < 4; lane++) {
      if (in_l[lane] >= 0 && in_l[lane] < in_length) {
        const VEC4_T in_texel =
            texelFetch(t_in, ivec3(in_texel_idx[lane], out_c, n), 0);
        sum[lane] += w * in_texel[in_lane[lane]];
      }
    }
  }

  sum += VEC4_T(T(t_bias[out_c]));
  imageStore(
      t_out,
      ivec3(out_l, out_c, n),
      op(sum, VEC4_T(out_min), VEC4_T(out_max)));
}
