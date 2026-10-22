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

#define op(X, A, B) ${OPERATOR}

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

#include "dispatch.glslh"
#include "activations.h"

$if DYNAMIC_PARAMS:
  float decode_bound(const uint value, const int is_int) {
    return is_int != 0 ? float(int(value)) : uintBitsToFloat(value);
  }

#ifdef USING_BUFFER

void main() {
  const int i = int(linear_idx_from_gid());
  if (i >= numel) {
    return;
  }

$if DYNAMIC_PARAMS:
  const T in_val = T(t_in[i]);
  const T minimum_val = T(decode_bound(minimum, bounds_are_int.x));
  const T maximum_val = T(decode_bound(maximum, bounds_are_int.y));
  t_out[i] = T(op(in_val, minimum_val, maximum_val));
$else:
  const float in_val = float(t_in[i]);
  t_out[i] = T(op(in_val, minimum, maximum));
}

#else

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, out_limits.xyz))) {
    return;
  }

  VEC4_T in_texel = texelFetch(t_in, pos, 0);
$if DYNAMIC_PARAMS:
  const VEC4_T minimum_val = VEC4_T(decode_bound(minimum, bounds_are_int.x));
  const VEC4_T maximum_val = VEC4_T(decode_bound(maximum, bounds_are_int.y));
  imageStore(t_out, pos, op(in_texel, minimum_val, maximum_val));
$else:
  imageStore(t_out, pos, VEC4_T(op(in_texel, minimum, maximum)));
}

#endif
