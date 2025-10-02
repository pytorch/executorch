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
#define SCALAR_T ${texel_load_component_type(DTYPE, STORAGE)}

${define_active_storage_type(STORAGE)}

${define_required_extensions(DTYPE)}
${define_required_extensions(BUF_DTYPE)}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_buffer(B, "r", "buf_in", BUF_DTYPE)}

$if USE_PUSH_CONST:
  layout(push_constant) uniform restrict Block {
    ivec4 sizes;
  $if not FROM_STAGING:
    ivec4 buf_strides;
  };
$else:
  ${layout_declare_ubo(B, "ivec4", "sizes")}
  $if not FROM_STAGING:
    ${layout_declare_ubo(B, "ivec4", "buf_strides")}

#include "indexing_utils.h"

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "t_layout", "DEFAULT_LAYOUT")}
${layout_declare_spec_const(C, "int", "transpose_hw", "0")}

const lowp ivec4 axis_map = unhash_axis_map(t_layout);
const lowp int packed_dim = unhash_packed_dim(t_layout);

VEC4_T read_texel(ivec4 tidx) {
  ivec4 tidx_to_use = tidx;
  ivec4 sizes_to_use = sizes;
  int packed_dim_to_use = packed_dim;
  if (transpose_hw == 1) {
    sizes_to_use.xy = sizes_to_use.yx;
    tidx_to_use.xy = tidx.yx;

    if (packed_dim == 1) {
      packed_dim_to_use = 0;
    }
    if (packed_dim == 0) {
      packed_dim_to_use = 1;
    }
  }

  $if FROM_STAGING:
    const ivec4 buf_indices = tidx_to_nchwi(tidx_to_use, sizes_to_use, packed_dim_to_use);
  $else:
    const ivec4 buf_indices = tidx_to_4bufi(tidx_to_use, buf_strides, packed_dim_to_use);

  VEC4_T texel = VEC4_T(0);
  if (tidx[packed_dim] < sizes[packed_dim]) {
    texel.x = SCALAR_T(buf_in[buf_indices.x]);
  }
  if (tidx[packed_dim] + 1 < sizes[packed_dim]) {
    texel.y = SCALAR_T(buf_in[buf_indices.y]);
  }
  if (tidx[packed_dim] + 2 < sizes[packed_dim]) {
    texel.z = SCALAR_T(buf_in[buf_indices.z]);
  }
  if (tidx[packed_dim] + 3 < sizes[packed_dim]) {
    texel.w = SCALAR_T(buf_in[buf_indices.w]);
  }
  return texel;
}

void main() {
  const ivec3 lpos = ivec3(gl_GlobalInvocationID);
  const ivec4 tidx = lpos_to_tidx(lpos, sizes, axis_map.w, packed_dim);
  if (any(greaterThanEqual(tidx, sizes))) {
    return;
  }

  $if DTYPE == "double" and DTYPE == "int64":
    VEC4_T texel = read_texel(tidx);
    write_texel(t_out, lpos_to_pos(lpos, axis_map), texel);
  $else:
    write_texel(t_out, lpos_to_pos(lpos, axis_map), read_texel(tidx));
}
