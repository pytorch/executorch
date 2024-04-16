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

#define to_tensor_idx to_tensor_idx_${PACKING}
#define get_packed_dim get_packed_dim_${PACKING}
#define get_packed_stride get_packed_stride_${PACKING}

#include "indexing_utils.h"

layout(std430) buffer;

layout(set = 0, binding = 0) uniform PRECISION ${SAMPLER_T[NDIM][DTYPE]} image_in;
layout(set = 0, binding = 1) buffer PRECISION restrict writeonly Buffer {
  BUF_T data[];
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
  const ivec4 idx = to_tensor_idx(pos, gpu_sizes.data);

  if (any(greaterThanEqual(idx, gpu_sizes.data))) {
    return;
  }

  const VEC4_T intex = texelFetch(image_in, ${get_pos[NDIM]("pos")}, 0);

  const int base_index = to_buffer_i(idx, cpu_sizes.data);
  const ivec4 buf_indices =
      base_index + ivec4(0, 1, 2, 3) * get_packed_stride(cpu_sizes.data);

  const int packed_dim_size = get_packed_dim(cpu_sizes.data);
  int packed_idx = get_packed_dim(idx);

  if (packed_idx < packed_dim_size) {
    buffer_out.data[buf_indices.x] = BUF_T(intex.x);
  }
  if (packed_idx + 1 < packed_dim_size) {
    buffer_out.data[buf_indices.y] = BUF_T(intex.y);
  }
  if (packed_idx + 2 < packed_dim_size) {
    buffer_out.data[buf_indices.z] = BUF_T(intex.z);
  }
  if (packed_idx + 3 < packed_dim_size) {
    buffer_out.data[buf_indices.w] = BUF_T(intex.w);
  }
}
