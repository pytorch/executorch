// where.glsl

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
#define T ${buffer_scalar_type(DTYPE)}
#define COND_T ${buffer_scalar_type("bool")}

${define_active_storage_type(STORAGE)}
${define_required_extensions(DTYPE)}
${define_required_extensions("bool")}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_condition", "bool", STORAGE)}
${layout_declare_tensor(B, "r", "t_self", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_other", DTYPE, STORAGE)}


#include "indexing_utils.h"

$if STORAGE == "buffer":
  ${layout_declare_ubo(B, "int", "out_numl")}
  ${layout_declare_ubo(B, "ivec4", "out_strides")}
  ${layout_declare_ubo(B, "ivec4", "cond_strides")}
  ${layout_declare_ubo(B, "ivec4", "self_strides")}
  ${layout_declare_ubo(B, "ivec4", "other_strides")}

  ${layout_declare_spec_const(C, "int", "out_packed_dim", "DEFAULT_LAYOUT")}
  ${layout_declare_spec_const(C, "int", "cond_packed_dim", "DEFAULT_LAYOUT")}
  ${layout_declare_spec_const(C, "int", "self_packed_dim", "DEFAULT_LAYOUT")}
  ${layout_declare_spec_const(C, "int", "other_packed_dim", "DEFAULT_LAYOUT")}
$else:
  ${layout_declare_ubo(B, "ivec3", "out_limits")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#ifdef USING_BUFFER

void main() {
  int out_bufi = int(gl_GlobalInvocationID.x);
  // ivec4 tidx = ivec4(gl_GlobalInvocationID, 0);
  // int out_bufi = tidx_to_bufi(tidx, out_strides);
  // int cond_bufi = tidx_to_bufi(tidx, cond_strides);
  // int self_bufi = tidx_to_bufi(tidx, self_strides);
  // int other_bufi = tidx_to_bufi(tidx, other_strides);
  if (out_bufi >= out_numl) {
    return;
  }

  const ivec4 out_tidx = bufi_to_tidx(out_bufi, out_strides, out_packed_dim);
  out_bufi = tidx_to_bufi(out_tidx, out_strides);

  const ivec4 cond_tidx = bufi_to_tidx(out_bufi, out_strides, out_packed_dim);
  const int cond_bufi = tidx_to_bufi(cond_tidx, cond_strides);

  const ivec4 self_tidx = bufi_to_tidx(out_bufi, out_strides, out_packed_dim);
  const int self_bufi = tidx_to_bufi(self_tidx, self_strides);

  const ivec4 other_tidx = bufi_to_tidx(out_bufi, out_strides, out_packed_dim);
  const int other_bufi = tidx_to_bufi(other_tidx, other_strides);

  COND_T cond = t_condition[cond_bufi] ;
  T v_self = t_self[self_bufi];
  T v_other = t_other[other_bufi];

  if (cond > 0) {
    t_out[out_bufi] = v_self;
  } else {
    t_out[out_bufi] = v_other;
  }
}

#else // !USING_BUFFER

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);


  if (any(greaterThanEqual(pos, out_limits))) {
    return;
  }

  vec4 cond = load_texel(t_condition, pos);
  VEC4_T selftex = load_texel(t_self, pos);
  VEC4_T othertex = load_texel(t_other, pos);

  VEC4_T outtex;

  for (int idx = 0; idx < 4; ++idx) {
    if (cond[idx] == 1) {
      outtex[idx] = selftex[idx];
    } else {
      outtex[idx] = othertex[idx];
    }
  }
  write_texel(t_out, pos, outtex);
}
 #endif // !USING_BUFFER
