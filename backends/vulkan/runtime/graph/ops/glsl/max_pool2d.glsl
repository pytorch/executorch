/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}
#define FLT_MIN -3.402823466e+38

#include "indexing_utils.h"

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(B, "w", "t_idx", "int", STORAGE)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE)}
${layout_declare_ubo(B, "ivec3", "out_limits")}
${layout_declare_ubo(B, "ivec4", "in_sizes")}
${layout_declare_ubo(B, "ivec2", "kernel_size", "ivec2", "stride", "ivec2", "padding", "ivec2", "dilation")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "write_indices", "1")}

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, out_limits))) {
    return;
  }

  const ivec2 ipos = pos.xy * stride - padding;

  const ivec2 start = ipos;
  const ivec2 end = ipos + kernel_size * dilation;

  vec4 out_texel = vec4(FLT_MIN);
  ivec4 idx_texel = ivec4(0);

  for (int y = start.y; y < end.y; y += dilation.y) {
    for (int x = start.x; x < end.x; x += dilation.x) {
      if ((x >= 0 && x < in_sizes.x) && (y >= 0 && y < in_sizes.y)) {
        const vec4 cur_texel = load_texel(t_in, ivec3(x, y, pos.z));

        // Set idx if value is greatest in the pool; else, keep the existing idx.
        ivec4 cur_idx = ivec4(x + int(in_sizes.x) * y);
        ivec4 mask = ivec4(greaterThan(cur_texel, out_texel));
        idx_texel = ivec4(mix(idx_texel, cur_idx, mask));

        out_texel = max(cur_texel, out_texel);
      }
    }
  }

  imageStore(t_out, pos, out_texel);
  if (write_indices > 0) {
    imageStore(t_idx, pos, idx_texel);
  }
}
