/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_load_type(DTYPE, "buffer")}

$if DTYPE == "half":
  #extension GL_EXT_shader_16bit_storage : require
  #extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
$elif DTYPE == "int8":
  #extension GL_EXT_shader_8bit_storage : require
  #extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
$elif DTYPE == "uint8":
  #extension GL_EXT_shader_8bit_storage : require
  #extension GL_EXT_shader_explicit_arithmetic_types_uint8 : require

layout(std430) buffer;

${layout_declare_tensor(0, "rw", "buffer_in", DTYPE, "buffer")}
${layout_declare_ubo(1, "int", "ntexels")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const float scalar = 2.0;

void main() {
  const int t_id = ivec3(gl_GlobalInvocationID).x;
  if (t_id >= ntexels) {
    return;
  }

  buffer_in[t_id] = buffer_in[t_id] + VEC4_T(scalar);// buffer_in[t_id];
}
