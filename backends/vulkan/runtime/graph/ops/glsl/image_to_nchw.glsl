/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#include "indexing_utils.h"

layout(std430) buffer;

layout(set = 0, binding = 0) uniform PRECISION ${SAMPLER_T[NDIM][DTYPE]} image_in;
layout(set = 0, binding = 1) buffer PRECISION restrict writeonly Buffer {
  ${T[DTYPE]} data[];
}
buffer_out;

layout(set = 0, binding = 2) uniform PRECISION restrict GpuSizes {
  ivec4 data;
}
gpu_sizes;

layout(set = 0, binding = 3) uniform PRECISION restrict CpuSizes {
  ivec4 data;
}
cpu_sizes;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const ivec4 coord = POS_TO_COORD_${PACKING}(pos, gpu_sizes.data);

  if (any(greaterThanEqual(coord, gpu_sizes.data))) {
    return;
  }

  const ${VEC4_T[DTYPE]} intex = texelFetch(image_in, ${GET_POS[NDIM]("pos")}, 0);

  const int base_index = COORD_TO_BUFFER_IDX(coord, cpu_sizes.data);
  const ivec4 buf_indices =
      base_index + ivec4(0, 1, 2, 3) * STRIDE_${PACKING}(cpu_sizes.data);

  const int packed_dim_size = PACKED_DIM_${PACKING}(cpu_sizes.data);
  int packed_coord = PACKED_DIM_${PACKING}(coord);

  if (packed_coord < packed_dim_size) {
    buffer_out.data[buf_indices.x] = intex.x;
  }
  if (packed_coord + 1 < packed_dim_size) {
    buffer_out.data[buf_indices.y] = intex.y;
  }
  if (packed_coord + 2 < packed_dim_size) {
    buffer_out.data[buf_indices.z] = intex.z;
  }
  if (packed_coord + 3 < packed_dim_size) {
    buffer_out.data[buf_indices.w] = intex.w;
  }
}
