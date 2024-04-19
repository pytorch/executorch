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

layout(std430) buffer;

#include "indexing_utils.h"

layout(set = 0, binding = 0, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly ${IMAGE_T[NDIM][DTYPE]} image_out;
layout(set = 0, binding = 1) uniform PRECISION sampler3D image_in;

#define VEC4_T ${texel_type(DTYPE)}

#define to_texture_pos_elem to_texture_pos_elem_${PACKING}

layout(set = 0, binding = 2) uniform PRECISION restrict OutGpuSizes {
  ivec4 out_gpu_sizes;
};

layout(set = 0, binding = 3) uniform PRECISION restrict OutCpuSizes {
  ivec4 out_cpu_sizes;
};

layout(set = 0, binding = 4) uniform PRECISION restrict InGpuSizes {
  ivec4 in_gpu_sizes;
};

layout(set = 0, binding = 5) uniform PRECISION restrict InCpuSizes {
  ivec4 in_cpu_sizes;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int packed_dim = 2;

void main() {
	const ivec3 out_pos = ivec3(gl_GlobalInvocationID);
	const ivec4 out_tensor_idx = to_tensor_idx(out_pos, out_gpu_sizes, packed_dim);

  if (all(greaterThanEqual(out_tensor_idx, out_gpu_sizes))) {
    return;
  }

  // Assume there is a virtual continous buffer in nchw format. From the output
  // pos, we first calculate the index in the virual buffer, and then calculate
  // the input position from the indx.

  const uint base_index = to_nchw_i(out_tensor_idx, out_cpu_sizes);
  const uvec4 buf_indices =
    base_index + ivec4(0, 1, 2, 3) * get_nchw_stride(out_cpu_sizes, packed_dim);

  VEC4_T value;
  // Need to look up the 4 values in the output texel separately.
  for (int i=0; i<4; i++) {
    ivec4 user_coor = from_buffer_i(buf_indices[i], in_cpu_sizes);

    ivec4 in_pos_elem = to_texture_elem_pos(user_coor, in_gpu_sizes, packed_dim);
    //ivec4 in_pos_elem = to_texture_pos_elem(user_coor, in_gpu_sizes);

    VEC4_T intex = VEC4_T(texelFetch(image_in, in_pos_elem.xyz, 0));

    value[i] = intex[in_pos_elem.w];
  }

  imageStore(image_out, out_pos, value);
}
