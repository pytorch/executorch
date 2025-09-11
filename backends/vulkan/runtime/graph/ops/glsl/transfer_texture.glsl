/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}
#define UBO_PARAMS ${UBO_PARAMS}

#define VEC4_T ${texel_type(DTYPE)}
#define T ${buffer_scalar_type(DTYPE)}

${define_active_storage_type("texture3d")}
${define_required_extensions(DTYPE)}

layout(std430) buffer;

#include "indexing_utils.h"

${layout_declare_tensor(B, "w", "t_out", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_in", DTYPE, "texture3d")}

$if UBO_PARAMS:
  $if OP_NAME == "slice":
    ${layout_declare_ubo(B, "int", "start")}
    ${layout_declare_ubo(B, "int", "step")}

  $if OP_NAME == "select":
    ${layout_declare_ubo(B, "int", "index")}

layout(push_constant) uniform restrict Block {
  ivec4 out_sizes;
  ivec4 in_sizes;
  int selected_dim;
  $if not UBO_PARAMS:
    $if OP_NAME == "slice":
      int start;
      int step;

    $if OP_NAME == "select":
      int index;
};

${layout_declare_spec_const(C, "int", "out_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 out_axis_map = unhash_axis_map(out_layout);
const lowp int out_packed_dim = unhash_packed_dim(out_layout);

${layout_declare_spec_const(C, "int", "in_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 in_axis_map = unhash_axis_map(in_layout);
const lowp int in_packed_dim = unhash_packed_dim(in_layout);

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#include "${OP_NAME}.glslh"

void main() {
  const ivec3 lpos = ivec3(gl_GlobalInvocationID);
  ivec4 out_tidx = lpos_to_tidx(lpos, out_sizes, out_axis_map.w, out_packed_dim);

  if (any(greaterThanEqual(out_tidx, out_sizes))) {
    return;
  }

  if (can_use_fast_path()) {
    ivec4 in_tidx = out_tidx_to_in_tidx(out_tidx);
    ivec3 in_pos = tidx_to_pos(in_tidx, in_sizes, in_axis_map, in_packed_dim);
    VEC4_T in_texel = VEC4_T(load_texel(t_in, in_pos));

    write_texel_lpos(t_out, lpos, in_texel, out_axis_map);
  }
  else {
    VEC4_T out_texel = VEC4_T(0);
    for (int texel_i = 0; texel_i < 4; ++texel_i) {
      ivec4 in_tidx = out_tidx_to_in_tidx(out_tidx);
      ivec3 in_pos = tidx_to_pos(in_tidx, in_sizes, in_axis_map, in_packed_dim);
      int element_idx = in_tidx[in_packed_dim] % 4;

      VEC4_T in_texel = VEC4_T(load_texel(t_in, in_pos));
      T selected_value = T(in_texel[element_idx]);

      out_texel[texel_i] = selected_value;

      out_tidx[out_packed_dim]++;
    }

    write_texel_lpos(t_out, lpos, out_texel, out_axis_map);
  }
}
