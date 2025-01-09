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

#define BATCH_SIZE_Y ${BATCH_SIZE_Y}

#define op(X, A, B) ${OPERATOR}

#include "indexing_utils_u16.h"

layout(std430) buffer;

${layout_declare_tensor(0, "w", "t_out", DTYPE, "texture3d")}
${layout_declare_tensor(1, "r", "t_in", DTYPE, "texture3d")}
${layout_declare_tensor(2, "r", "t_kernel", DTYPE, "texture2d")}
${layout_declare_tensor(3, "r", "t_bias", DTYPE, "texture2d")}
${layout_declare_ubo(4, "ivec3", "out_limits")}
${layout_declare_ubo(5, "ivec4", "in_sizes")}
${layout_declare_ubo(6, "ivec2", "kernel_size", "ivec2", "stride", "ivec2", "padding", "ivec2", "dilation")}
${layout_declare_ubo(7, "ivec2", "overlay_region", "int", "in_group_size")}
${layout_declare_ubo(8, "float", "out_min", "float", "out_max")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * Computes a depthwise convolution. Each shader invocation calculates the
 * output at a single output location.
 */
void main() {
  // y divided up by batch size is used to determine 3d position
  // since work size is calculated by x * ((y + B_Y - 1) / B_Y) * z
  const int out_limits_y_scaled = (out_limits.y + BATCH_SIZE_Y - 1) / BATCH_SIZE_Y;

  u16vec3 pos = idx_to_u16pos_x_wise(gl_GlobalInvocationID.x, out_limits.x, out_limits_y_scaled);

  // scale pos.y by batch size, because that's the top pixel to be processed
  pos.y *= uint16_t(BATCH_SIZE_Y);

  // do not process if top pixel does not fit within the output range
  if (any(greaterThanEqual(u16vec3(pos.x, pos.y, pos.z), out_limits))) {
    return;
  }

  // Compute the index of the top-left element of the overlay region. Negative
  // indices indicate that the top-left element is in a region added by padding.
  const u16vec2 ipos = pos.xy * u16vec2(stride) - u16vec2(padding);

  // Compute the start and end of the input indices to load. Padding is assumed
  // to be constant 0 padding, so any reads from the padding region is skipped.
  const u16vec2 start = ipos;
  const u16vec2 end = ipos + u16vec2(overlay_region.xy);

  // sum outputs
  VEC4_T sum[BATCH_SIZE_Y];

  sum[0] = texelFetch(t_bias, u16vec2(pos.z, 0), 0);
  for (int i = 1; i < BATCH_SIZE_Y; i++) {
    sum[i] = sum[0];
  }

  // array to store input texels
  VEC4_T in_texels[TILE_SIZE];

  // array to store kernel data of previous y
  VEC4_T prev_kernel_line[TILE_SIZE];

  uint16_t kx = uint16_t(0);
  for (uint16_t y = start.y, i = uint16_t(0); i < uint16_t(TILE_SIZE + BATCH_SIZE_Y - 1); y += uint16_t(dilation.y), i++) {
    for (uint16_t x = start.x, j = uint16_t(0); j < uint16_t(TILE_SIZE); x += uint16_t(dilation.x), j++) {
      in_texels[int(j)] = texelFetch(t_in, u16vec3(x, y, pos.z), 0);
    }

    // from 2nd iteration onwards accumulate dot product in 2nd sum
    // based on kernel line data fetched in previous iteration and input texel from this iteration
    if (i > uint16_t(0)) {
      for (uint16_t j = uint16_t(0); j < uint16_t(TILE_SIZE); j++) {
        sum[1] = fma(in_texels[int(j)], prev_kernel_line[int(j)], sum[1]);
      }
    }

    // accumulate dot product in 1st sum only until tile size
    if (i < uint16_t(TILE_SIZE)) {
      for (uint16_t j = uint16_t(0); j < uint16_t(TILE_SIZE); j++, kx++) {
        prev_kernel_line[int(j)] = texelFetch(t_kernel, u16vec2(kx, pos.z), 0);
        sum[0] = fma(in_texels[int(j)], prev_kernel_line[int(j)], sum[0]);
      }
    }
  }

  for (int i = 0; i < BATCH_SIZE_Y; i++) {
    if (any(greaterThanEqual(u16vec3(pos.x, pos.y + i, pos.z), out_limits))) {
      continue;
    }
    imageStore(t_out, u16vec3(pos.x, pos.y + i, pos.z), op(sum[i], out_min, out_max));
  }
}
