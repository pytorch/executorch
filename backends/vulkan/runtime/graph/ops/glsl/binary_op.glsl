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

#define op(X, Y, A) ${OPERATOR}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_other", DTYPE, STORAGE)}

#include "broadcasting_utils.h"
#include "indexing_utils.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "out_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 out_axis_map = unhash_axis_map(out_layout);
const lowp int packed_dim = unhash_packed_dim(out_layout);

${layout_declare_spec_const(C, "int", "in_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 in_axis_map = unhash_axis_map(in_layout);

${layout_declare_spec_const(C, "int", "other_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 other_axis_map = unhash_axis_map(other_layout);

layout(push_constant) uniform restrict Block {
  ivec4 out_sizes;
  ivec4 in_sizes;
  ivec4 other_sizes;
  ivec2 broadcast_params;
  float alpha;
};

void main() {
  const ivec3 lpos = ivec3(gl_GlobalInvocationID);
  const ivec4 tidx = lpos_to_tidx(lpos, out_sizes, out_axis_map.w, packed_dim);

  if (any(greaterThanEqual(tidx, out_sizes))) {
    return;
  }

  // broadcast on logical sizes
  ivec4 in_idx = broadcast_indices(tidx, in_sizes);
  VEC4_T in_texel = VEC4_T(load_texel(
    t_in,
    // read axis mapped texel
    tidx_to_pos(in_idx, in_sizes, in_axis_map, packed_dim)));

  // broadcast on logical sizes
  ivec4 other_idx = broadcast_indices(tidx, other_sizes);
  VEC4_T other_texel = VEC4_T(load_texel(
    t_other,
    // read axis mapped texel
    tidx_to_pos(other_idx, other_sizes, other_axis_map, packed_dim)));

  // Check boolean broadcast flags; we use ivec2 instead of bvec2 for alignment.
  if (broadcast_params.x > 0) {
    in_texel = in_texel.xxxx;
  }
  if (broadcast_params.y > 0) {
    other_texel = other_texel.xxxx;
  }

  write_texel_lpos(
    t_out,
    lpos,
    VEC4_T(op(in_texel, other_texel, alpha)),
    out_axis_map);
}
