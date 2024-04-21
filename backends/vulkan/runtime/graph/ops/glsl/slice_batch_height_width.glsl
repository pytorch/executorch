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

layout(set = 0, binding = 2) uniform PRECISION restrict Sizes {
  ivec4 sizes;
};

layout(set = 0, binding = 3) uniform PRECISION restrict SliceArg {
  int dim;
  int offset;
  int step;
  // Used when dim=batch. Stride is the # of plances for each batch  value.
  int stride;
}
slice_arg;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int packed_dim = C_DIM;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (pos_out_of_bounds(pos, sizes, packed_dim)) {
    return;
  }

  ivec3 in_pos = pos;

  int index = pos[slice_arg.dim] / slice_arg.stride;
  int within_stride = pos[slice_arg.dim] % slice_arg.stride;

  in_pos[slice_arg.dim] = slice_arg.offset * slice_arg.stride + index * slice_arg.step *
    slice_arg.stride + within_stride;

  imageStore(image_out, pos, texelFetch(image_in, in_pos, 0));

}
