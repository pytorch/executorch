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

#extension GL_EXT_shader_16bit_storage : enable

layout(std430) buffer;

#define BUF_T ${buffer_scalar_type(DTYPE)}
#define VEC4_T ${texel_type(DTYPE)}
#define SCALAR_T ${texel_component_type(DTYPE)}

#define pos_to_coord pos_to_coord_${PACKING}
#define coord_to_pos coord_to_pos${PACKING}
#define get_packed_dim get_packed_dim_${PACKING}
#define get_packed_stride get_packed_stride_${PACKING}

layout(set = 0, binding = 0, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly ${IMAGE_T[ND][DTYPE]} image_out;
layout(set = 0, binding = 1) buffer  PRECISION restrict readonly Buffer {
  BUF_T data[];
}
buffer_in;

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
  const ivec4 coord = pos_to_coord(pos, gpu_sizes.data);

  if (any(greaterThanEqual(coord, gpu_sizes.data))) {
    return;
  }

  const int base_index = coord_to_buffer_idx(coord, cpu_sizes.data);
  const ivec4 buf_indices =
      base_index + ivec4(0, 1, 2, 3) * get_packed_stride(cpu_sizes.data);

  SCALAR_T val_x = SCALAR_T(buffer_in.data[buf_indices.x]);
  SCALAR_T val_y = SCALAR_T(buffer_in.data[buf_indices.y]);
  SCALAR_T val_z = SCALAR_T(buffer_in.data[buf_indices.z]);
  SCALAR_T val_w = SCALAR_T(buffer_in.data[buf_indices.w]);

  VEC4_T texel = VEC4_T(val_x, val_y, val_z, val_w);

  const int packed_dim_size = get_packed_dim(cpu_sizes.data);
  int packed_coord = get_packed_dim(coord);

  if (packed_coord + 3 >= packed_dim_size) {
    ivec4 packed_ind = ivec4(packed_coord) + ivec4(0, 1, 2, 3);
    VEC4_T valid_idx = VEC4_T(lessThan(packed_ind, ivec4(packed_dim_size)));
    texel = texel * valid_idx;
  }

  imageStore(image_out, ${get_pos[ND]("pos")}, texel);
}
