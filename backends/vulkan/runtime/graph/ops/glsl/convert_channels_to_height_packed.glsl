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

$if DTYPE == "half":
  #extension GL_EXT_shader_16bit_storage : require

layout(std430) buffer;

layout(set = 0, binding = 0, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly image3D image_out;
layout(set = 0, binding = 1) uniform PRECISION ${SAMPLER_T[NDIM][DTYPE]} image_in;

layout(set = 0, binding = 2) uniform PRECISION restrict Sizes {
  ivec4 sizes;
};

layout(set = 0, binding = 3) uniform PRECISION restrict OutLimits {
  ivec3 out_limits;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, out_limits))) {
    return;
  }

  int src_w = pos.x;
  int src_base_h = pos.y * 4;

  int num_c = sizes.y;

  int src_c = pos.z % num_c;
  int src_n = pos.z / num_c;

  // Fetch the 4 elements from the channel-packed tensor
  ivec4 src_pos0 = get_channel_packed_pos_from_index(
    ivec4(src_n, src_c, src_base_h, src_w),
    sizes);

  ivec4 src_pos1 = get_channel_packed_pos_from_index(
    ivec4(src_n, src_c, src_base_h + 1, src_w),
    sizes);

  ivec4 src_pos2 = get_channel_packed_pos_from_index(
    ivec4(src_n, src_c, src_base_h + 2, src_w),
    sizes);

  ivec4 src_pos3 = get_channel_packed_pos_from_index(
    ivec4(src_n, src_c, src_base_h + 3, src_w),
    sizes);

  vec4 t0 = texelFetch(image_in, src_pos0.xyz, 0);
  vec4 t1 = texelFetch(image_in, src_pos1.xyz, 0);
  vec4 t2 = texelFetch(image_in, src_pos2.xyz, 0);
  vec4 t3 = texelFetch(image_in, src_pos3.xyz, 0);

  vec4 out_t = vec4(
    t0[src_pos0.w],
    t1[src_pos1.w],
    t2[src_pos2.w],
    t3[src_pos3.w]);

  imageStore(image_out, pos, out_t);
}
