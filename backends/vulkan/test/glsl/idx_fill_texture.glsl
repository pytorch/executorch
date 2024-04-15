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

layout(set = 0, binding = 0, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly ${IMAGE_T[NDIM][DTYPE]} image_out;

layout(set = 0, binding = 1) uniform PRECISION restrict GpuSizes {
  ivec4 data;
}
gpu_sizes;

layout(set = 0, binding = 2) uniform PRECISION restrict CpuSizes {
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

  const int base_index = COORD_TO_BUFFER_IDX(coord, cpu_sizes.data);
  const ivec4 buf_indices =
      base_index + ivec4(0, 1, 2, 3) * PLANE_SIZE_${PACKING}(gpu_sizes.data);

  VEC4_T texel = VEC4_T(buf_indices);

  imageStore(image_out, ${get_pos[NDIM]("pos")}, texel);
}
