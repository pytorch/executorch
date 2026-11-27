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

${layout_declare_tensor(0, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_tensor(1, "r", "t_in", DTYPE, STORAGE)}
${layout_declare_tensor(2, "r", "t_idx", "int", STORAGE)}
${layout_declare_ubo(3, "ivec4", "sizes")}
${layout_declare_ubo(4, "ivec4", "in_sizes")}
${layout_declare_ubo(5, "int", "gpu_dim")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int packed_dim = C_DIM;

void main() {
  const ivec3 out_pos = ivec3(gl_GlobalInvocationID);

  if (pos_out_of_bounds(out_pos, sizes, packed_dim)) {
    return;
  }

  // Selecting along the batch dim steps over the z axis in units of channel
  // texels, because the batch and channel dims share that axis. The width and
  // height dims each have an axis to themselves, so they step by one. The
  // stride is read from in_sizes rather than baked in at build time so that it
  // follows a resize that changes the channel count.
  const int stride = gpu_dim == 2 ? ((in_sizes.z + 3) / 4) : 1;

  const int out_idx = out_pos[gpu_dim] / stride;
  const int within_stride = out_pos[gpu_dim] % stride;
  const int in_idx = texelFetch(t_idx, ivec3(out_idx, 0, 0), 0).x;

  ivec3 in_pos = out_pos;
  in_pos[gpu_dim] = in_idx * stride + within_stride;

  imageStore(t_out, out_pos, texelFetch(t_in, in_pos, 0));
}
