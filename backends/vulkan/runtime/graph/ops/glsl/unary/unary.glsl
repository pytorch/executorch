/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions(STORAGE, DTYPE)}

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}
#define T ${buffer_scalar_type(DTYPE)}

// Buffer ops compute in the parameter type. Runtime bounds of int32 clamps stay
// int so that large values remain exact.
$if DYNAMIC_PARAMS:
  #define PARAM_T ${texel_load_component_type(DTYPE, "texture3d")}
$else:
  #define PARAM_T float
$if STORAGE == "buffer":
  #define IN_T PARAM_T
  #define OUT_T T
$else:
  #define IN_T VEC4_T
  #define OUT_T VEC4_T
$if DTYPE in ["half", "float"]:
  #define UNARY_OP_IS_FLOAT
$elif DTYPE == "uint8":
  #define UNARY_OP_IS_BITWISE

${define_active_storage_type(STORAGE)}

#include "indexing_utils.h"

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE)}

$if DYNAMIC_PARAMS:
  ${layout_declare_ubo(B, "uint", "minimum")}
  ${layout_declare_ubo(B, "uint", "maximum")}

layout(push_constant) uniform restrict Block {
$if STORAGE == "buffer":
  int numel;
$else:
  ivec4 out_limits;
$if DYNAMIC_PARAMS:
  ivec2 bounds_are_int;
$else:
  float minimum;
  float maximum;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "op_type", "0")}

#include "dispatch.glslh"
#include "unary_op.glslh"

$if DYNAMIC_PARAMS:
  float decode_bound(const uint value, const int is_int) {
    return is_int != 0 ? float(int(value)) : uintBitsToFloat(value);
  }

  #define PARAM_A PARAM_T(decode_bound(minimum, bounds_are_int.x))
  #define PARAM_B PARAM_T(decode_bound(maximum, bounds_are_int.y))
$else:
  #define PARAM_A minimum
  #define PARAM_B maximum

#ifdef USING_BUFFER

void main() {
  const int i = int(linear_idx_from_gid());
  if (i >= numel) {
    return;
  }

  t_out[i] = apply_unary_op(IN_T(t_in[i]), PARAM_A, PARAM_B);
}

#else

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, out_limits.xyz))) {
    return;
  }

  VEC4_T in_texel = texelFetch(t_in, pos, 0);
  imageStore(t_out, pos, apply_unary_op(in_texel, PARAM_A, PARAM_B));
}

#endif
