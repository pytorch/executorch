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

#define op(X, A, B) ${OPERATOR}

${define_active_storage_type(STORAGE)}

$if BINARY_OP != "A":
  #define binary_op(A, B) ${BINARY_OP}

#include "indexing_utils.h"

${define_required_extensions(DTYPE)}

layout(std430) buffer;

${layout_declare_tensor(0, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(1, "r", "t_in", DTYPE, STORAGE)}
$if BINARY_OP != "A":
  ${layout_declare_tensor(2, "r", "t_other", DTYPE, STORAGE)}

layout(push_constant) uniform restrict Block {
$if STORAGE == "buffer":
  int numel;
$else:
  ivec4 out_limits;
float minimum;
float maximum;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#include "activations.h"

#ifdef USING_BUFFER

void main() {
  const int i = int(gl_GlobalInvocationID.x);
  if (i >= numel) {
    return;
  }

  float in_val = float(t_in[i]);
  t_out[i] = T(op(in_val, minimum, maximum));
}

#else

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, out_limits.xyz))) {
    return;
  }

  VEC4_T in_texel = texelFetch(t_in, pos, 0);

$if BINARY_OP == "A":
  imageStore(t_out, pos, VEC4_T(op(in_texel, minimum, maximum)));
$else:
  imageStore(t_out, pos, VEC4_T(binary_op(op(in_texel, minimum, maximum), texelFetch(t_other, pos, 0))));
}

#endif
