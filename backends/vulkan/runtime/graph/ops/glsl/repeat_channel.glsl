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

layout(set = 0, binding = 2) uniform PRECISION restrict RepeatArgs {
  // With input_size (n, c_i, h, w) and repeat r
  // out_size == (n, c_i * r, h, w)
  ivec4 out_sizes;
  ivec4 in_sizes;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int packed_dim = C_DIM;


void main() {
  const ivec3 out_pos = ivec3(gl_GlobalInvocationID);

  const ivec4 out_whcn = to_tensor_idx(out_pos, out_sizes, packed_dim);

  if (any(greaterThanEqual(out_whcn, out_sizes))) {
    return;
  }

  VEC4_T v;
  // Loop over the 4 elements in texel, calculate the corresponding elem, and
  // fetch. Not most efficient algorithm because likely we fetch same texel
  // multiple times in this loop.

  for (int i=0; i<4;i++) {
    ivec4 in_whcn = out_whcn;
    in_whcn.z = (out_whcn.z + i) % in_sizes.z;

    ivec4 in_elem_pos = to_texture_elem_pos(in_whcn, in_sizes, packed_dim);

    v[i] = VEC4_T(texelFetch(image_in, in_elem_pos.xyz, 0))[in_elem_pos.w];
  }

  imageStore(image_out, out_pos, v);
}
