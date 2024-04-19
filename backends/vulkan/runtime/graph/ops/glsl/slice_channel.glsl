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
#define to_texture_pos_elem to_texture_pos_elem_${PACKING}
#define get_packed_stride get_packed_stride_${PACKING}


layout(std430) buffer;

#include "indexing_utils.h"

layout(set = 0, binding = 0, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly ${IMAGE_T[NDIM][DTYPE]} image_out;
layout(set = 0, binding = 1) uniform PRECISION sampler3D image_in;

layout(set = 0, binding = 2) uniform PRECISION restrict OutSizes {
  uvec4 data;
}
out_sizes;

layout(set = 0, binding = 3) uniform PRECISION restrict OutCpuSizes {
  uvec4 out_cpu_sizes;
};

layout(set = 0, binding = 4) uniform PRECISION restrict InGpuSizes {
  uvec4 in_gpu_sizes;
};

layout(set = 0, binding = 5) uniform PRECISION restrict SliceArg {
  int offset;
  int step;
}
slice_arg;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 out_pos = ivec3(gl_GlobalInvocationID);
  
  const ivec4 idx = to_tensor_idx_C_packed(out_pos, out_sizes.data);

  if (any(greaterThanEqual(idx, out_sizes.data))) {
    return;
  }

  // We map the output pos using the buffer index.  For each index in the texel,
  // we calculate the source whcn-coordinate amended with offset-ed channel
  // value.  Then we calculate the actual texture position from the
  // whcn-coordinate.

  const uint base_index = to_buffer_i(idx, out_cpu_sizes);
  uvec4 buf_indices =
    base_index + ivec4(0, 1, 2, 3) * get_packed_stride(out_cpu_sizes);
 
  vec4 outex;
  for (int i=0;i<4;i++) {
      ivec4 user_coor = from_buffer_i(buf_indices[i], out_cpu_sizes);
 
      int in_channel = user_coor.z;

      ivec4 in_user_coor = user_coor;
      in_user_coor.z = slice_arg.offset + in_channel * slice_arg.step;

      ivec4 in_pow_elem = to_texture_pos_elem_C_packed(
        in_user_coor,
        in_gpu_sizes);

      vec4 v = texelFetch(image_in, in_pow_elem.xyz, 0);

      outex[i] = v[in_pow_elem.w];
  }
  imageStore(image_out, out_pos, outex);
}
