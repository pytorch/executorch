/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#include "broadcasting_utils.h"
#include "indexing_utils.h"

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}
${define_active_storage_type(STORAGE)}

layout(std430) buffer;

${layout_declare_tensor(0, "w", "image_out", DTYPE, STORAGE)}
${layout_declare_tensor(1, "r", "image_in", DTYPE, STORAGE)}
${layout_declare_ubo(2, "ivec3", "out_limits")}
${layout_declare_ubo(3, "ivec4", "sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, out_limits))) {
    return;
  }

  VEC4_T in_texel = texelFetch(image_in, pos, 0);
  imageStore(image_out, pos, in_texel);
}
