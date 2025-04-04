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
#define FLOAT_T ${buffer_scalar_type(DTYPE)}

${define_active_storage_type(STORAGE)}

${define_required_extensions(DTYPE)}
$if STORAGE == "buffer":
  ${define_required_extensions("int8")}

#include "indexing_utils.h"

layout(std430) buffer;

${layout_declare_tensor(0, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(1, "r", "t_mat1", DTYPE, STORAGE)}
${layout_declare_tensor(2, "r", "t_qmat2", "int8", STORAGE)}
${layout_declare_tensor(3, "r", "t_scales", DTYPE, STORAGE)}

$if STORAGE == "buffer":
  ${layout_declare_ubo(4, "ivec4", "out_sizes")}
  ${layout_declare_ubo(5, "ivec4", "out_strides")}
  ${layout_declare_ubo(6, "int", "out_numel")}
  ${layout_declare_ubo(7, "ivec4", "mat1_sizes")}
  ${layout_declare_ubo(8, "ivec4", "mat1_strides")}
  ${layout_declare_ubo(9, "ivec4", "qmat2_strides")}
  ${layout_declare_ubo(10, "ivec4", "scales_strides")}
$else:
  ${layout_declare_ubo(4, "ivec3", "out_limits")}
  ${layout_declare_ubo(5, "ivec4", "mat1_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// This header file must be defined after the layout descriptors have been
// declared because the functions in the header assume some variables have been
// declared as layout descriptors.

#ifdef USING_BUFFER

#ifndef FLOAT_T
#define FLOAT_T float
#endif

void main() {
  const int out_bufi = int(gl_GlobalInvocationID.x);
  if (out_bufi >= out_numel) {
    return;
  }

  const ivec4 out_tidx = bufi_to_tidx(out_bufi, out_strides, 0);

  const FLOAT_T scale = t_scales[out_tidx.x];

  FLOAT_T outval = FLOAT_T(0.0);

  int mat1_offset = out_tidx.y * mat1_strides.y + out_tidx.z * qmat2_strides.z;
  int qmat2_offset = out_tidx.x;

  // TODO(ssjia): optimize memory access pattern by traversing mat1 x in inner loop
  for (int i = 0; i < mat1_sizes.x; i++) {
    const FLOAT_T mat1_val = t_mat1[mat1_offset];
    const FLOAT_T mat2_val = FLOAT_T(t_qmat2[qmat2_offset]);

    outval += mat1_val * mat2_val;

    mat1_offset++;
    qmat2_offset += qmat2_strides.y;
  }

  t_out[out_bufi] = outval * scale;
}

#else // USING_TEXTURE

#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require

void main() {
  const u16vec2 out_pos = u16vec2(
    gl_GlobalInvocationID.x,
    gl_GlobalInvocationID.y);

  if (out_pos.x >= out_limits.x || out_pos.y >= out_limits.y) {
    return;
  }

  const uint16_t qmat2_pos_x = out_pos.x;

  VEC4_T outtex = VEC4_T(0);

  const VEC4_T scales = load_texel(t_scales,  u16vec3(out_pos.x, 0, 0));

  VEC4_T mat1_tex;
  VEC4_T mat2_tex[4];
  for (
    uint16_t i = uint16_t(0), x = uint16_t(0);
    i < uint16_t(mat1_sizes.x);
    i += uint16_t(4), x++)
  {
    mat1_tex = load_texel(t_mat1, u16vec3(x, out_pos.y, 0));

    mat2_tex[0] = load_texel(t_qmat2, u16vec3(out_pos.x, i, 0));
    mat2_tex[1] = load_texel(t_qmat2, u16vec3(out_pos.x, i + uint16_t(1), 0));
    mat2_tex[2] = load_texel(t_qmat2, u16vec3(out_pos.x, i + uint16_t(2), 0));
    mat2_tex[3] = load_texel(t_qmat2, u16vec3(out_pos.x, i + uint16_t(3), 0));

    outtex += mat1_tex.x * mat2_tex[0] + mat1_tex.y * mat2_tex[1] + mat1_tex.z * mat2_tex[2] + mat1_tex.w * mat2_tex[3];
  }

  outtex *= scales;
  write_texel(t_out, u16vec3(out_pos, 0), outtex);
}

#endif
