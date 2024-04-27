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
layout(set = 0, binding = 1) uniform PRECISION sampler3D existing_out;
layout(set = 0, binding = 2) uniform PRECISION sampler3D image_in;

layout(set = 0, binding = 3) uniform PRECISION restrict CopyArgs {
  ivec4 out_sizes;
  ivec4 in_sizes;
  // Analogus to range variable in copy. It defines the # of channel being
  // copied.
  int channel_range;  
  int src_channel_offset;
  int dst_channel_offset;
  int unused; 
  // Operates on (x, y, z) extents. 
  ivec3 range;
  int unused1;
  ivec3 dst_offset;
  int unused2;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int packed_dim = C_DIM;

void main() {
  // Note: Unlike other shaders, the range is often not equal to the destination
  // texture extent.
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  if (any(greaterThanEqual(pos, range))) {
    return;
  }

  const ivec3 out_pos = pos + dst_offset;

  const ivec4 out_whcn = to_tensor_idx(out_pos, out_sizes, packed_dim);

  // First read the existing values to make sure the boundary values stay.
  VEC4_T v = VEC4_T(texelFetch(existing_out, out_pos, 0));

  for (int i=0; i<4; i++) {
    ivec4 in_whcn = out_whcn;

    in_whcn.z = out_whcn.z - dst_channel_offset + i;

    // Handle the partial update for begining of channel in an existing tensor.
    // If the source channel index is below zero or exceeds the range, we skip
    // updating the element to avoid overwriting existing data.
    if ((in_whcn.z < 0) || (in_whcn.z >= channel_range)) {
      continue;
    }

    // Readjust for the source offset.
    in_whcn.z = in_whcn.z + src_channel_offset;
    
    ivec4 in_elem_pos = to_texture_elem_pos(in_whcn, in_sizes, packed_dim);
    v[i] = VEC4_T(texelFetch(image_in, in_elem_pos.xyz, 0))[in_elem_pos.w];
  }

  imageStore(image_out, out_pos, v);
}
