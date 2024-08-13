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

#define op(X, A, B) ${OPERATOR}

#include "indexing_utils.h"

layout(std430) buffer;

layout(set = 0, binding = 0, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly ${IMAGE_T[NDIM][DTYPE]} image_out;
layout(set = 0, binding = 1) uniform PRECISION sampler3D image_in;
layout(set = 0, binding = 2) uniform PRECISION sampler3D kernel_in;
layout(set = 0, binding = 3) uniform PRECISION sampler3D bias_in;

layout(set = 0, binding = 4) uniform PRECISION restrict OutLimits {
  ivec3 out_limits;
};

layout(set = 0, binding = 5) uniform PRECISION restrict InSizes {
  ivec4 in_sizes;
};

layout(set = 0, binding = 6) uniform PRECISION restrict Params {
  int kernel_size;
  int stride;
  int padding;
  int dilation;
  int in_group_size;
  int out_group_size;
};

layout(set = 0, binding = 7) uniform PRECISION restrict OutputParams {
  float out_min;
  float out_max;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Let us define
//
// input = (N, in_C, in_L),
// output = (N, out_C, out_L),
// groups = G,
// kernel = K,
//
// which results in shapes
//
// weight = (out_C, in_C / G, K),
// bias = (out_C,).
//
// This implementation performs out_C shader invocations, where each invocation
// calculates the rolling kernel of the length dimension for each batch, i.e.,
// computes out_L * N results.
//
// Note that we can rewrite this implementation as out_L * out_C * ceil(N / 4)
// shader invocations, where each invocation computes 1 result. But that
// performs worse.
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);

  if (any(greaterThanEqual(pos, out_limits))) {
    return;
  }

  int in_length = in_sizes.x;
  int batch_size = in_sizes.z;

  // "out_c" is the output's channel index where we write our result.
  // Across shader invocations, this is the only value that varies.
  int out_c = pos.y;
  vec4 bias = texelFetch(bias_in, ivec3(out_c, 0, 0), 0);

  // "in_c" tracks the input's channel start index.
  // We iterate over the input group that corresponds to the output group.
  int c_start = (out_c / out_group_size) * in_group_size;
  int c_end = c_start + in_group_size;

  // "in_l" tracks the input's length start index for our input-kernel overlay
  // region.
  int l_start = -padding;
  int l_end = in_length + padding - dilation * (kernel_size - 1);

  // Since the input/output tensors are channel-packed, which is along the
  // batch dimension, we can batch-read/write four elements at a time.
  for (int n = 0; n < batch_size; n += 4) {
    // "out_l" tracks the output's length index where we write our result.
    int out_l = 0;

    for (int in_l = l_start; in_l < l_end; in_l += stride, ++out_l) {
      vec4 sum = vec4(0);

      for (int in_c = c_start; in_c < c_end; ++in_c) {
        // "k" tracks the kernel's index for our input-kernel computation.
        // It reads out-of-bound zeros, but trying to avoid them complicates
        // for-loop conditions, which results in worse performance.
        for (int k = 0; k < kernel_size; k += 4) {
          // Since the weight tensor is width-packed, which is along the length
          // dimension, we can batch-read four elements at a time.
          const ivec3 w_pos = ivec3(k / 4, in_c % in_group_size, out_c);
          const vec4 weight = texelFetch(kernel_in, w_pos, 0);

          const ivec3 in_pos_0 = ivec3(in_l + k * dilation, in_c, n / 4);
          sum = fma(weight.xxxx, texelFetch(image_in, in_pos_0, 0), sum);

          const ivec3 in_pos_1 = ivec3(in_l + (k+1) * dilation, in_c, n / 4);
          sum = fma(weight.yyyy, texelFetch(image_in, in_pos_1, 0), sum);

          const ivec3 in_pos_2 = ivec3(in_l + (k+2) * dilation, in_c, n / 4);
          sum = fma(weight.zzzz, texelFetch(image_in, in_pos_2, 0), sum);

          const ivec3 in_pos_3 = ivec3(in_l + (k+3) * dilation, in_c, n / 4);
          sum = fma(weight.wwww, texelFetch(image_in, in_pos_3, 0), sum);
        }
      }

      ivec3 out_pos = ivec3(out_l, out_c, n / 4);
      imageStore(image_out, out_pos, op(sum + bias.x, out_min, out_max));
    }
  }
}
