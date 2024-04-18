/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define VEC4_T ${buffer_gvec_type(DTYPE, 4)}

#include "indexing_utils.h"

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

layout(set = 0, binding = 0) buffer  PRECISION restrict writeonly Buffer {
  VEC4_T data[];
}
buffer_in;

layout(set = 0, binding = 1) uniform PRECISION restrict Params {
  int len;
}
params;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const int i = ivec3(gl_GlobalInvocationID).x;

  const int base = 4 * i;
  if (base < params.len) {
    buffer_in.data[i] = VEC4_T(base, base + 1, base + 2, base + 3);
  }
}
