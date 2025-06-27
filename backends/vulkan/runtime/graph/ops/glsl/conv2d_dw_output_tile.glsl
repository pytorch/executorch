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

#define TILE_SIZE ${TILE_SIZE}

#define BATCH_SIZE_X ${BATCH_SIZE_X}

#define BATCH_SIZE_Y ${BATCH_SIZE_Y}

#define LOCAL_WG_SIZE 64

#define op(X, A, B) ${OPERATOR}

#include "indexing_utils.h"

layout(std430) buffer;

${layout_declare_tensor(0, "w", "t_out", DTYPE, "texture3d")}
${layout_declare_tensor(1, "r", "t_in", DTYPE, "texture3d")}
${layout_declare_tensor(2, "r", "t_kernel", DTYPE, "texture2d")}
${layout_declare_tensor(3, "r", "t_bias", DTYPE, "texture2d")}

layout(push_constant) uniform restrict Block {
  ivec4 out_limits;
  ivec4 in_sizes;
  ivec2 kernel_size;
  ivec2 stride;
  ivec2 padding;
  ivec2 dilation;
  ivec2 overlay_region;
  int in_group_size;
  int dummy_padding;
  float out_min;
  float out_max;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Computes a depthwise convolution. Each shader invocation calculates the
 * output at a single output location.
 */

void main() {
  // x and y are divided by batch size to determine 3d position
  // since work size is calculated by x * ((y + B_Y - 1) / B_Y) * z
  const ivec2 out_limits_xy_scaled = (out_limits.xy + ivec2(BATCH_SIZE_X, BATCH_SIZE_Y) - 1) / ivec2(BATCH_SIZE_X, BATCH_SIZE_Y);

  const uint div_by_x = gl_GlobalInvocationID.x / out_limits_xy_scaled.x;
  ivec3 pos = ivec3(
    gl_GlobalInvocationID.x % out_limits_xy_scaled.x,
    div_by_x,
    gl_GlobalInvocationID.y);

  // do not process if top pixel does not fit within the output range
  if (pos.y >= out_limits_xy_scaled.y || pos.z >= out_limits.z) {
    return;
  }

  // scale pos.xy by batch sizes, because that's the top pixel to be processed
  pos.x *= BATCH_SIZE_X;
  pos.y *= BATCH_SIZE_Y;

  // Compute the index of the top-left element of the overlay region. Negative
  // indices indicate that the top-left element is in a region added by padding.
  const ivec2 ipos = pos.xy * stride - padding;

  // Compute the start and end of the input indices to load. Padding is assumed
  // to be constant 0 padding, so any reads from the padding region is skipped.
  const ivec2 start = ipos;

  // sum outputs
  VEC4_T sum[BATCH_SIZE_Y * BATCH_SIZE_X];

  for (int i = 0; i < BATCH_SIZE_Y * BATCH_SIZE_X; i++) {
    sum[i] = VEC4_T(0);
  }

  // array to store input texels
  VEC4_T in_texels[TILE_SIZE + BATCH_SIZE_X - 1];

  // array to store kernel data of previous y
  VEC4_T prev_kernel_line[TILE_SIZE];

  int kx = 0;
  for (int y = start.y, i = 0; i < TILE_SIZE + BATCH_SIZE_Y - 1; y += dilation.y, i++) {
    for (int x = start.x, j = 0; j < TILE_SIZE + BATCH_SIZE_X - 1; x += dilation.x, j++) {
      in_texels[j] = texelFetch(t_in, ivec3(x, y, pos.z), 0);
    }

    // from 2nd iteration onwards accumulate dot product in 2nd sum
    // based on kernel line data fetched in previous iteration and input texel from this iteration
    if (i > 0) {
      for (int j = 0; j < TILE_SIZE; j++) {
        for (int s = 0; s < BATCH_SIZE_X; s++) {
          sum[BATCH_SIZE_X + s] = fma(in_texels[j + s], prev_kernel_line[j], sum[BATCH_SIZE_X + s]);
        }
      }
    }

    // accumulate dot product in 1st sum only until tile size
    if (i < TILE_SIZE) {
      for (int j = 0; j < TILE_SIZE; j++, kx++) {
        prev_kernel_line[j] = texelFetch(t_kernel, ivec2(kx, pos.z), 0);
        for (int s = 0; s < BATCH_SIZE_X; s++) {
          sum[s] = fma(in_texels[j + s], prev_kernel_line[j], sum[s]);
        }
      }
    }
  }

  const VEC4_T bias = texelFetch(t_bias, ivec2(pos.z, 0), 0);
  for (int y = 0; y < BATCH_SIZE_Y; y++) {
    for (int x = 0; x < BATCH_SIZE_X; x++) {
      const ivec3 out_pos = ivec3(pos.x + x, pos.y + y, pos.z);
      if (all(lessThan(out_pos.xy, out_limits.xy))) {
        imageStore(t_out, out_pos, op(sum[y * BATCH_SIZE_X + x] + bias, out_min, out_max));
      }
    }
  }
}
