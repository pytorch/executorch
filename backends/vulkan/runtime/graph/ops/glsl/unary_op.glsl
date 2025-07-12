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

#include "indexing_utils.h"

${define_required_extensions(DTYPE)}

layout(std430) buffer;

${layout_declare_tensor(0, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(1, "r", "t_in", DTYPE, STORAGE)}
$if STORAGE == "buffer":
  ${layout_declare_ubo(2, "int", "numel")}
$else:
  ${layout_declare_ubo(2, "ivec3", "out_limits")}
${layout_declare_ubo(3, "float", "minimum")}
${layout_declare_ubo(4, "float", "maximum")}

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

  if (any(greaterThanEqual(pos, out_limits))) {
    return;
  }

  VEC4_T in_texel = texelFetch(t_in, pos, 0);
  imageStore(t_out, pos, VEC4_T(op(in_texel, minimum, maximum)));
}

#endif
