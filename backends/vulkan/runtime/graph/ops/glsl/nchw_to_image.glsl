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
#define SCALAR_T ${texel_component_type(DTYPE)}

#include "indexing_utils.h"

$if DTYPE == "half":
  #extension GL_EXT_shader_16bit_storage : require

layout(std430) buffer;

layout(set = 0, binding = 0, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly ${IMAGE_T[NDIM][DTYPE]} image_out;
layout(set = 0, binding = 1) buffer  PRECISION restrict readonly Buffer {
  BUF_T buffer_in[];
};

layout(set = 0, binding = 2) uniform PRECISION restrict Sizes {
  ivec4 sizes;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int packed_dim = C_DIM;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const ivec4 idx = to_tensor_idx(pos, sizes, packed_dim);

  if (any(greaterThanEqual(idx, sizes))) {
    return;
  }

  const ivec4 buf_indices = get_texel_nchw_buffer_ixs(idx, sizes, packed_dim);

  const int packed_dim_size = sizes[packed_dim];
  int packed_idx = idx[packed_dim];

  VEC4_T texel = VEC4_T(0);
  if (packed_idx < packed_dim_size) {
    texel.x = SCALAR_T(buffer_in[buf_indices.x]);
  }
  if (packed_idx + 1 < packed_dim_size) {
    texel.y = SCALAR_T(buffer_in[buf_indices.y]);
  }
  if (packed_idx + 2 < packed_dim_size) {
    texel.z = SCALAR_T(buffer_in[buf_indices.z]);
  }
  if (packed_idx + 3 < packed_dim_size) {
    texel.w = SCALAR_T(buffer_in[buf_indices.w]);
  }

  imageStore(image_out, ${get_pos[NDIM]("pos")}, texel);
}
