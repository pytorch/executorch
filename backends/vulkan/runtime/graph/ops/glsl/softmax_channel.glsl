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

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  // `early_exit` is the global workgroup position-based condition for unnecessary invocations to exit.
  ivec4 early_exit = get_early_exit(sizes, dims.x, dims.y);

  // how "wide" a batch is in terms of z. Only have one invocation per batch,
  // as one batch width has elements from every channel in-memory.
  if (!all(lessThan(pos, early_exit.xyz))) {
    return;
  }

  const int b_stride = int(ceil(sizes.z / 4.0));
  const ivec3 src_pos = ivec3(pos.x, pos.y, pos.z * b_stride);
  // tail case, padded zeros in memory if tensor's channel dim % 4 != 0
  uint tail_case_size = sizes.z % 4;
  if (tail_case_size == 0) {
    tail_case_size = 4;
  }
  // Calculate the denominator for the whole dimension.
  // For numerical stability to avoid floating point overflow,
  // we leverage the translation invariance of the softmax function,
  // subtracting every element along channel by the maximum element along
  // channel. find the maximum element
  float max_element = texelFetch(image_in, src_pos, 0)[0];
  for (int c = 0; c < b_stride - 1; c++) {
    const vec4 c_texel =
        texelFetch(image_in, ivec3(src_pos.x, src_pos.y, src_pos.z + c), 0);
    for (int t = 0; t < 4; t++) {
      if (c_texel[t] > max_element) {
        max_element = c_texel[t];
      }
    }
  }
  vec4 c_texel = texelFetch(
      image_in, ivec3(src_pos.x, src_pos.y, src_pos.z + b_stride - 1), 0);
  for (int t = 0; t < tail_case_size; t++) {
    if (c_texel[t] > max_element) {
      max_element = c_texel[t];
    }
  }
  // Calculate the denominator.
  float denominator = 0;
  for (int c = 0; c < b_stride - 1; c++) {
    const vec4 c_texel =
        texelFetch(image_in, ivec3(src_pos.x, src_pos.y, src_pos.z + c), 0);
    for (int t = 0; t < 4; t++) {
      denominator += exp(c_texel[t] - max_element);
    }
  }
  c_texel = texelFetch(
      image_in, ivec3(src_pos.x, src_pos.y, src_pos.z + b_stride - 1), 0);
  for (int t = 0; t < tail_case_size; t++) {
    denominator += exp(c_texel[t] - max_element);
  }
  // Calculate every final channel element.
  for (int c = 0; c < b_stride; c++) {
    const ivec3 dst_pos = ivec3(src_pos.x, src_pos.y, src_pos.z + c);
    const vec4 numerator = op1(texelFetch(image_in, dst_pos, 0) - max_element);
    imageStore(image_out, dst_pos, op2(numerator, denominator));
  }
}
