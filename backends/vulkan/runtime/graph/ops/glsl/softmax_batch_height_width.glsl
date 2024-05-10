/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define op1(X) ${OPERATOR1}

#define op2(X, Y) ${OPERATOR2}

#include "indexing_utils.h"
#include "softmax.h"

layout(std430) buffer;

layout(set = 0, binding = 0, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly ${IMAGE_T[NDIM][DTYPE]} image_out;
layout(set = 0, binding = 1) uniform PRECISION sampler3D image_in;

layout(set = 0, binding = 2) uniform PRECISION restrict Extents {
  ivec3 extents;
};

layout(set = 0, binding = 3) uniform PRECISION restrict Sizes {
  ivec4 sizes;
};

layout(set = 0, binding = 4) uniform PRECISION restrict Params {
  // x in_dim
  // y softmax_dim
  ivec2 dims;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * This shader can compute softmax along batch, height, and width.
 */

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  // `early_exit` is the global workgroup position-based condition for unnecessary invocations to exit.
  ivec4 early_exit = get_early_exit(sizes, dims.x, dims.y);

  if (!all(lessThan(pos, early_exit.xyz))) {
    return;
  }

  // `input_dim_stride` is the stride to include elements along the softmax dimension calculation.
  ivec4 input_dim_stride = get_input_dim_stride(dims.x, dims.y, sizes.z);

  // Calculate the denominator for the whole dimension.
  // For numerical stability to avoid floating point overflow,
  // we leverage the translation invariance of the softmax function,
  // subtracting every element along input_dim_stride by
  // the maximum element along input_dim_stride.
  // find the maximum element
  vec4 max_element = texelFetch(image_in, pos, 0);
  ivec3 cand_pos = pos + input_dim_stride.xyz;
  while (all(lessThan(cand_pos, extents.xyz))) {
    max_element = max(texelFetch(image_in, cand_pos, 0), max_element);
    cand_pos += input_dim_stride.xyz;
  }
  // Calculate the denominator along the direction of input_dim_stride.
  cand_pos = pos;
  vec4 denominator = vec4(0, 0, 0, 0);
  while (all(lessThan(cand_pos, extents.xyz))) {
    denominator += exp(texelFetch(image_in, cand_pos, 0) - max_element);
    cand_pos += input_dim_stride.xyz;
  }
  // Calculate every final element along the direction of input_dim_stride.
  cand_pos = pos;
  while (all(lessThan(cand_pos, extents.xyz))) {
    const vec4 numerator = op1(texelFetch(image_in, cand_pos, 0) - max_element);
    imageStore(image_out, cand_pos, op2(numerator, denominator));
    cand_pos += input_dim_stride.xyz;
  }
}
