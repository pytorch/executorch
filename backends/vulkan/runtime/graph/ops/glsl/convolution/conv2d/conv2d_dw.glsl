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

${layout_declare_tensor(0, "w", "t_out", DTYPE, "texture3d")}
${layout_declare_tensor(1, "r", "t_in", DTYPE, "texture3d")}
${layout_declare_tensor(2, "r", "t_kernel", DTYPE, "texture2d")}
${layout_declare_tensor(3, "r", "t_bias", DTYPE, "texture2d")}
${layout_declare_ubo(4, "ivec3", "out_limits")}
${layout_declare_ubo(5, "ivec4", "in_sizes")}
${layout_declare_ubo(6, "ivec2", "kernel_size", "ivec2", "stride", "ivec2", "padding", "ivec2", "dilation")}
${layout_declare_ubo(7, "ivec2", "overlay_region", "int", "in_group_size")}
${layout_declare_ubo(8, "float", "out_min", "float", "out_max")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#include "dispatch.glslh"

${layout_declare_spec_const(C, "int", "ngroups", "1")}

/*
 * Computes a depthwise convolution. Each shader invocation calculates the
 * output at a single output location.
 */
void main() {
  const uint linear_idx = linear_idx_from_gid();
  const uint div_by_x = linear_idx / out_limits.x;
  const ivec3 pos = ivec3(
    linear_idx % out_limits.x,
    div_by_x % out_limits.y,
    div_by_x / out_limits.y);

  if (pos.z >= out_limits.z) {
    return;
  }

  // Compute the index of the top-left element of the overlay region. Negative
  // indices indicate that the top-left element is in a region added by padding.
  const ivec2 ipos = pos.xy * stride - padding;

  // Padding is constant 0, so taps that fall outside the input contribute
  // nothing and are skipped. The loops still cover the full kernel extent so
  // that kx (the tap index into the flattened filter row) stays aligned with
  // the tap being read: clipping the loop bounds to in_sizes instead leaves kx
  // short by the clipped taps, and every following row of the filter then
  // reads the wrong weights for any output whose window crosses the right or
  // bottom edge.
  VEC4_T sum = texelFetch(t_bias, ivec2(pos.z, 0), 0);
  int kx = 0;
  for (int ky = 0; ky < kernel_size.y; ++ky) {
    const int y = ipos.y + ky * dilation.y;
    for (int i = 0; i < kernel_size.x; ++i, ++kx) {
      const int x = ipos.x + i * dilation.x;
      if (any(lessThan(ivec2(x, y), ivec2(0))) ||
          any(greaterThanEqual(ivec2(x, y), in_sizes.xy))) {
        continue;
      }
      // The weight kernel was rearranged such that every NxN filter is
      // flattened to fit in one row. Each filter was then stacked on top of
      // each other vertically.
      const VEC4_T in_texel = texelFetch(t_in, ivec3(x, y, pos.z), 0);
      sum = fma(in_texel, texelFetch(t_kernel, ivec2(kx, pos.z), 0), sum);
    }
  }

  imageStore(t_out, pos, op(sum, out_min, out_max));
}
