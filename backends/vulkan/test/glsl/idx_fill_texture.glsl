/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_type(DTYPE)}

#include "indexing_utils.h"

layout(std430) buffer;

${layout_declare_tensor(0, "w", "image_out", DTYPE, "texture3d")}
${layout_declare_ubo(1, "ivec4", "sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int packed_dim = C_DIM;
layout(constant_id = 4) const int offset = 10;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const ivec4 idx = to_tensor_idx(pos, sizes, packed_dim);

  if (any(greaterThanEqual(idx, sizes))) {
    return;
  }

  const ivec4 buf_indices = get_texel_nchw_buffer_ixs(idx, sizes, packed_dim);
  VEC4_T texel = VEC4_T(buf_indices) + offset;
  imageStore(image_out, pos, texel);
}
