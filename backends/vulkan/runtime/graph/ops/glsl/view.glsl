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

layout(set = 0, binding = 2) uniform PRECISION restrict OutSizes {
  ivec4 out_sizes;
};

layout(set = 0, binding = 3) uniform PRECISION restrict InSizes {
  ivec4 in_sizes;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int packed_dim = C_DIM;

void main() {
	const ivec3 out_pos = ivec3(gl_GlobalInvocationID);
	const ivec4 out_tensor_idx = to_tensor_idx(out_pos, out_sizes, packed_dim);

  if (all(greaterThanEqual(out_tensor_idx, out_sizes))) {
    return;
  }

  // Assume there is a virtual continous buffer in nchw format. From the output
  // pos, we first calculate the index in the virual buffer, and then calculate
  // the input position from the indx.
  const ivec4 buf_indices = get_texel_nchw_buffer_ixs(out_tensor_idx, out_sizes, packed_dim);

  VEC4_T value;
  // Need to look up the 4 values in the output texel separately.
  for (int i =0 ; i < 4; i++) {
    ivec4 user_coor = from_nchw_buffer_i(buf_indices[i], in_sizes);
    ivec4 in_pos_elem = to_texture_elem_pos(user_coor, in_sizes, packed_dim);
    VEC4_T intex = texelFetch(image_in, in_pos_elem.xyz, 0);
    value[i] = intex[in_pos_elem.w];
  }

  imageStore(image_out, out_pos, value);
}
