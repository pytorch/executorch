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
#define to_texture_pos to_texture_pos_${PACKING}

#define op(X, Y, A) ${OPERATOR}

#include "broadcasting_utils.h"
#include "indexing_utils.h"

layout(std430) buffer;

layout(set = 0, binding = 0, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly ${IMAGE_T[NDIM][DTYPE]} image_out;
layout(set = 0, binding = 1) uniform PRECISION sampler3D image_in;
layout(set = 0, binding = 2) uniform PRECISION sampler3D image_other;

layout(set = 0, binding = 3) uniform PRECISION restrict OutSizes {
  ivec4 data;
}
out_sizes;

layout(set = 0, binding = 4) uniform PRECISION restrict InSizes {
  ivec4 data;
}
in_sizes;

layout(set = 0, binding = 5) uniform PRECISION restrict OtherSizes {
  ivec4 data;
}
other_sizes;

layout(set = 0, binding = 6) uniform PRECISION restrict BroadcastParams {
  ivec2 data;
}
broadcast_params;

layout(set = 0, binding = 7) uniform PRECISION restrict Alpha {
  float data;
}
alpha;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const ivec4 idx = to_tensor_idx(pos, out_sizes.data);

  if (any(greaterThanEqual(idx, out_sizes.data))) {
    return;
  }

  ivec4 in_idx = broadcast_indices(idx, in_sizes.data);
  VEC4_T in_texel = VEC4_T(texelFetch(
    image_in,
    to_texture_pos(in_idx, in_sizes.data),
    0));

  ivec4 other_idx = broadcast_indices(idx, other_sizes.data);
  VEC4_T other_texel = VEC4_T(texelFetch(
    image_other,
    to_texture_pos(other_idx, other_sizes.data),
    0));

  // Check boolean broadcast flags; we use ivec2 instead of bvec2 for alignment.
  if (broadcast_params.data.x > 0) {
    in_texel = in_texel.xxxx;
  }
  if (broadcast_params.data.y > 0) {
    other_texel = other_texel.xxxx;
  }

  imageStore(image_out, pos, VEC4_T(op(in_texel, other_texel, alpha.data)));
}
