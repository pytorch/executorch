/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define BUF_T ${buffer_scalar_type(DTYPE)}
#define VEC4_T ${texel_type(DTYPE)}

${define_active_storage_type(STORAGE)}
${define_required_extensions(DTYPE)}
${define_required_extensions(SCALAR_VALUE_TYPE)}

#include "indexing_utils.h"

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_ubo(B, buffer_scalar_type(SCALAR_VALUE_TYPE), "scalar_value")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#ifdef USING_BUFFER

void main() {
  const int i = int(gl_GlobalInvocationID.x);

  if (i > 0) {
    return;
  }

  t_out[i] = BUF_T(scalar_value);
}

# else // !USING_BUFFER

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  // Scalar tensor is a special case where the packed dim is always 1.
  if (any(greaterThanEqual(pos, ivec3(1)))) {
    return;
  }

  VEC4_T outtex = VEC4_T(scalar_value);
  write_texel(t_out, pos, outtex);
}

#endif // !USING_BUFFER
