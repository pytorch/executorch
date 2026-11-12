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

#include "indexing_utils.h"

layout(std430) buffer;

${layout_declare_tensor(0, "w", "t_out", DTYPE, "texture3d")}
${layout_declare_tensor(1, "r", "t_in", DTYPE, "texture3d")}
${layout_declare_tensor(2, "r", "t_kernel", DTYPE, "texture2d")}
${layout_declare_tensor(3, "r", "t_bias", DTYPE, "texture2d")}
${layout_declare_tensor(4, "r", "t_residual", DTYPE, "texture3d")}
${layout_declare_tensor(5, "r", "t_scale", DTYPE, "texture2d")}
${layout_declare_ubo(6, "ivec3", "out_limits")}
${layout_declare_ubo(7, "ivec4", "in_sizes")}
${layout_declare_ubo(8, "ivec2", "kernel_size", "ivec2", "stride", "ivec2", "padding", "ivec2", "dilation")}
${layout_declare_ubo(9, "ivec2", "overlay_region", "int", "in_group_size")}
${layout_declare_ubo(10, "float", "activation_a", "float", "activation_b", "float", "residual_alpha")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "bool", "has_activation", "false")}
${layout_declare_spec_const(C, "int", "activation_op", "0")}
${layout_declare_spec_const(C, "bool", "has_residual", "false")}
${layout_declare_spec_const(C, "bool", "has_scale", "false")}
${layout_declare_spec_const(C, "int", "residual_op", "0")}

#include "conv2d_epilogue.glslh"

/*
 * Computes a 2D transpose convolution. Each shader invocation calculates the
 * output at a single output location. For details, refer to conv2d.glsl which
 * uses a similar approach.
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, out_limits))) {
    return;
  }

  ivec2 ipos = pos.xy + padding;

  const ivec2 start = max(
      ivec2(0),
      ivec2(ceil((vec2(ipos) - kernel_size + 1) / vec2(stride))));
  const ivec2 end =
      min(ivec2(in_sizes.xy),
          ivec2(floor(vec2(ipos) / vec2(stride))) + 1);

  const int ic = in_group_size;
  const int kx_stride = ic * (stride.x - 1);

  int ky_start = overlay_region.y - 1 - (ipos.y - stride.y * start.y) + pos.z * kernel_size.y;
  int kx_start = (overlay_region.x - 1 - (ipos.x - stride.x * start.x)) * ic;

  VEC4_T sum = texelFetch(t_bias, ivec2(pos.z, 0), 0);
  for (int y = start.y, ky = ky_start; y < end.y; ++y, ky += stride.y) {
    for (int x = start.x, kx = kx_start; x < end.x; ++x, kx += kx_stride) {
      for (int z4 = 0; z4 < ic / 4; ++z4, kx += 4) {
        const VEC4_T in_texel = texelFetch(t_in, ivec3(x, y, z4), 0);
        const ivec4 kxs = kx + ivec4(0, 1, 2, 3);

        sum = fma(in_texel.xxxx, texelFetch(t_kernel, ivec2(kxs.x, ky), 0), sum);
        sum = fma(in_texel.yyyy, texelFetch(t_kernel, ivec2(kxs.y, ky), 0), sum);
        sum = fma(in_texel.zzzz, texelFetch(t_kernel, ivec2(kxs.z, ky), 0), sum);
        sum = fma(in_texel.wwww, texelFetch(t_kernel, ivec2(kxs.w, ky), 0), sum);
      }
    }
  }

  const vec4 scale =
      has_scale ? texelFetch(t_scale, ivec2(pos.z, 0), 0) : vec4(1);
  imageStore(t_out, pos, apply_epilogue(vec4(sum), pos, scale));
}
