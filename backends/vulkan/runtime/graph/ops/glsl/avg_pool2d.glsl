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

${layout_declare_tensor(0, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(1, "r", "t_in", DTYPE, STORAGE)}
${layout_declare_ubo(2, "ivec3", "out_limits")}
${layout_declare_ubo(3, "ivec4", "in_sizes")}
${layout_declare_ubo(4, "ivec2", "kernel_size", "ivec2", "stride", "ivec2", "padding", "ivec2", "dilation")}
${layout_declare_ubo(5, "int", "divisor_override", "int", "count_include_pad")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, out_limits))) {
    return;
  }

  const ivec2 ipos = pos.xy * stride - padding;

  const ivec2 start = max(ivec2(0), ipos);
  const ivec2 end = min(ipos + kernel_size, ivec2(in_sizes.xy));

  VEC4_T sum = VEC4_T(0);
  for (int y = start.y; y < end.y; ++y) {
    for (int x = start.x; x < end.x; ++x) {
      sum += texelFetch(t_in, ivec3(x, y, pos.z), 0);
    }
  }

  int div;
  if (divisor_override > 0) {
    div = divisor_override;
  } else if (count_include_pad > 0) {
    ivec2 empty = max(ipos + kernel_size - padding - ivec2(in_sizes.xy), ivec2(0));
    div = (kernel_size.y - empty.y) * (kernel_size.x - empty.x);
  } else {
    div = (end.y - start.y) * (end.x - start.x);
  }
  imageStore(t_out, pos, sum / div);
}
