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

layout(std430) buffer;

#include "indexing_utils.h"

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_in", "int", STORAGE)}
${layout_declare_tensor(B, "r", "t_weight", DTYPE, STORAGE)}
${layout_declare_ubo(B, "ivec4", "sizes")}
${layout_declare_ubo(B, "ivec4", "out_axis_map")}
${layout_declare_ubo(B, "ivec4", "in_axis_map")}
${layout_declare_ubo(B, "ivec4", "weight_axis_map")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int packed_dim = C_DIM;

void main() {
  const ivec3 out_lpos = ivec3(gl_GlobalInvocationID);
  const ivec4 out_tidx = lpos_to_tidx(out_lpos, sizes, out_axis_map.w, packed_dim);
  if (any(greaterThanEqual(out_tidx, sizes))) {
    return;
  }
  VEC4_T out_texel;

  // Consider optimizing via W-packing format for t_in and t_weight.
  for (int i = 0; i < 4; ++i) {
    // Read input tensor for embedding index.
    const ivec3 in_pos = lpos_to_pos(ivec3(out_tidx.y, out_tidx.z * 4 + i, out_tidx.w / 4), in_axis_map);
    const int in_texel_elem = load_texel(t_in, in_pos)[out_tidx.w % 4];

    // Read weight tensor for embedding.
    const ivec3 weight_pos = lpos_to_pos(ivec3(out_tidx.x, in_texel_elem, 0), weight_axis_map);
    out_texel[i] = load_texel(t_weight, weight_pos).x;
  }

  imageStore(t_out, lpos_to_pos(out_lpos, out_axis_map), out_texel);
}
