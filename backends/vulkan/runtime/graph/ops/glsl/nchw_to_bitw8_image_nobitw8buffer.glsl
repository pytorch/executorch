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

${define_active_storage_type(STORAGE)}

#include "indexing_utils.h"

layout(std430) buffer;

#extension GL_EXT_control_flow_attributes : require

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_buffer(B, "r", "nchw_in", "int")}

$if USE_PUSH_CONST:
  layout(push_constant) uniform restrict Block {
    ivec4 sizes;
  };
$else:
  ${layout_declare_ubo(B, "ivec4", "sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "t_layout", "DEFAULT_LAYOUT")}
${layout_declare_spec_const(C, "int", "transpose_hw", "0")}

const lowp ivec4 axis_map = unhash_axis_map(t_layout);
const lowp int packed_dim = unhash_packed_dim(t_layout);

/*
 * Extends sign of int8
 */
int extend_sign(int x) {
  return x | mix(0, 0xFFFFFF00, x >= (1 << 7));
}

ivec4 read_texel(ivec4 tidx) {
  const ivec4 tidx_to_use = ivec4(mix(tidx.xy, tidx.yx, bvec2(transpose_hw == 1)), tidx.zw);
  const ivec4 sizes_to_use = ivec4(mix(sizes.xy, sizes.yx, bvec2(transpose_hw == 1)), sizes.zw);
  const int packed_dim_to_use = mix(packed_dim, packed_dim ^ transpose_hw, packed_dim < 2);

  const ivec4 buf_indices = tidx_to_nchwi(
      tidx_to_use, sizes_to_use, packed_dim_to_use);

  const int mask = (1 << 8) - 1;

  ivec4 out_tex = ivec4(0);

  [[unroll]] for (int i = 0; i < 4; ++i) {
    if (tidx[packed_dim] + i < sizes[packed_dim]) {
      const int in_texel = nchw_in[buf_indices[i] >> 2];
      int extracted_val = (in_texel >> (8 * (buf_indices[i] & 3))) & mask;
      extracted_val = extend_sign(extracted_val);
      out_tex[i] = extracted_val;
    }
  }

  return out_tex;
}

void main() {
  const ivec3 lpos = ivec3(gl_GlobalInvocationID);
  ivec4 tidx = lpos_to_tidx(lpos, sizes, axis_map.w, packed_dim);

  if (any(greaterThanEqual(tidx, sizes))) {
    return;
  }

  write_texel(t_out, lpos_to_pos(lpos, axis_map), VEC4_T(read_texel(tidx)));
}
