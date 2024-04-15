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

#define to_tensor_idx to_tensor_idx_${PACKING}
#define get_packed_dim get_packed_dim_${PACKING}

#include "broadcasting_utils.h"
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

layout(set = 0, binding = 3) uniform PRECISION restrict FillVal {
  float data;
}
fill_value;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const ivec4 idx = to_tensor_idx(pos, gpu_sizes.data);

  if (any(greaterThanEqual(idx, gpu_sizes.data))) {
    return;
  }

  VEC4_T outtex = VEC4_T(fill_value.data);
  const int packed_dim_size = get_packed_dim(cpu_sizes.data);
  int packed_idx = get_packed_dim(idx);

  if (packed_idx + 3 >= packed_dim_size) {
    ivec4 packed_ind = ivec4(packed_idx) + ivec4(0, 1, 2, 3);
    VEC4_T valid_idx = VEC4_T(lessThan(packed_ind, ivec4(packed_dim_size)));
    outtex = outtex * valid_idx;
  }

  imageStore(image_out, ${get_pos[NDIM]("pos")}, outtex);
}
