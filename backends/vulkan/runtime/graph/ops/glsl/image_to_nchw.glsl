/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define BUF_T ${buffer_scalar_type(DTYPE)}
#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}

${define_active_storage_type(STORAGE)}

${define_required_extensions(DTYPE)}

layout(std430) buffer;

${layout_declare_buffer(B, "w", "buf_out", DTYPE)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE)}

$if USE_PUSH_CONST:
  layout(push_constant) uniform restrict Block {
    ivec4 sizes;
  $if not TO_STAGING:
    ivec4 buf_strides;
  };
$else:
  ${layout_declare_ubo(B, "ivec4", "sizes")}
  $if not TO_STAGING:
    ${layout_declare_ubo(B, "ivec4", "buf_strides")}

#include "indexing_utils.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "t_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 axis_map = unhash_axis_map(t_layout);
const lowp int packed_dim = unhash_packed_dim(t_layout);

void write_out_texel(VEC4_T texel, ivec4 tidx) {
  $if TO_STAGING:
    const ivec4 buf_indices = tidx_to_nchwi(tidx, sizes, packed_dim);
  $else:
    const ivec4 buf_indices = tidx_to_4bufi(tidx, buf_strides, packed_dim);

  if (tidx[packed_dim] < sizes[packed_dim]) {
    buf_out[buf_indices.x] = BUF_T(texel.x);
  }
  if (tidx[packed_dim] + 1 < sizes[packed_dim]) {
    buf_out[buf_indices.y] = BUF_T(texel.y);
  }
  if (tidx[packed_dim] + 2 < sizes[packed_dim]) {
    buf_out[buf_indices.z] = BUF_T(texel.z);
  }
  if (tidx[packed_dim] + 3 < sizes[packed_dim]) {
    buf_out[buf_indices.w] = BUF_T(texel.w);
  }
}

void main() {
  const ivec3 lpos = ivec3(gl_GlobalInvocationID);
  const ivec4 tidx = lpos_to_tidx(lpos, sizes, axis_map.w, packed_dim);

  if (any(greaterThanEqual(tidx, sizes))) {
    return;
  }

  const VEC4_T intex = load_texel(t_in, lpos_to_pos(lpos, axis_map));
  write_out_texel(intex, tidx);
}
