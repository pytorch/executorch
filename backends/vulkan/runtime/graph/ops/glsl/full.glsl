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

#define POS ${get_pos[NDIM]("pos")}

#include "indexing_utils.h"

layout(std430) buffer;

layout(set = 0, binding = 0, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly ${IMAGE_T[NDIM][DTYPE]} image_out;

layout(set = 0, binding = 1) uniform PRECISION restrict Sizes {
  ivec4 sizes;
};

layout(set = 0, binding = 2) uniform PRECISION restrict FillVal {
  float fill_value;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int packed_dim = C_DIM;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const ivec4 idx = to_tensor_idx(pos, sizes, packed_dim);

  if (any(greaterThanEqual(idx, sizes))) {
    return;
  }

  VEC4_T outtex = VEC4_T(fill_value);
  const int packed_dim_size = sizes[packed_dim];
  int packed_idx = idx[packed_dim];

  if (packed_idx + 3 >= packed_dim_size) {
    ivec4 packed_ind = ivec4(packed_idx) + ivec4(0, 1, 2, 3);
    VEC4_T valid_idx = VEC4_T(lessThan(packed_ind, ivec4(packed_dim_size)));
    outtex = outtex * valid_idx;
  }

  imageStore(image_out, POS, outtex);
}
