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

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_in", "int", STORAGE)}
${layout_declare_tensor(B, "r", "t_weight", DTYPE, STORAGE)}
${layout_declare_ubo(B, "ivec4", "sizes")}

#include "indexing_utils.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "out_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 out_axis_map = unhash_axis_map(out_layout);
const lowp int packed_dim = unhash_packed_dim(out_layout);

${layout_declare_spec_const(C, "int", "in_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 in_axis_map = unhash_axis_map(in_layout);

${layout_declare_spec_const(C, "int", "weight_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 weight_axis_map = unhash_axis_map(weight_layout);

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
    const ivec3 in_lpos = ivec3(out_tidx.y, out_tidx.z * 4 + i, out_tidx.w / 4);
    const int in_texel_elem = load_texel_lpos(t_in, in_lpos, in_axis_map)[out_tidx.w % 4];

    // Read weight tensor for embedding, it is height-packed.
    const ivec3 weight_lpos = ivec3(out_tidx.x, in_texel_elem / 4, 0);
    out_texel[i] = load_texel_lpos(t_weight, weight_lpos, weight_axis_map)[in_texel_elem % 4];
  }

  write_texel_lpos(t_out, out_lpos, out_texel, out_axis_map);
}
