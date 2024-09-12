/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define BUF_T ${buffer_scalar_type(DTYPE)}
#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}

${define_active_storage_type(STORAGE)}

#include "indexing_utils.h"

${define_required_extensions(DTYPE)}

layout(std430) buffer;

${layout_declare_buffer(B, "w", "nchw_out", DTYPE)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, STORAGE)}
${layout_declare_ubo(B, "ivec4", "sizes")}
${layout_declare_ubo(B, "ivec4", "axis_map")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int packed_dim = C_DIM;

void write_out_texel(VEC4_T texel, ivec4 tensor_idx) {
  const ivec4 buf_indices = get_texel_nchw_buffer_ixs(
      tensor_idx,
      sizes,
      packed_dim);

  if (tensor_idx[packed_dim] < sizes[packed_dim]) {
    nchw_out[buf_indices.x] = BUF_T(texel.x);
  }
  if (tensor_idx[packed_dim] + 1 < sizes[packed_dim]) {
    nchw_out[buf_indices.y] = BUF_T(texel.y);
  }
  if (tensor_idx[packed_dim] + 2 < sizes[packed_dim]) {
    nchw_out[buf_indices.z] = BUF_T(texel.z);
  }
  if (tensor_idx[packed_dim] + 3 < sizes[packed_dim]) {
    nchw_out[buf_indices.w] = BUF_T(texel.w);
  }
}

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const ivec4 tensor_idx = to_tensor_idx(pos, sizes, axis_map, packed_dim);

  if (any(greaterThanEqual(tensor_idx, sizes))) {
    return;
  }

  const VEC4_T intex = load_texel(t_in, pos);
  write_out_texel(intex, tensor_idx);
}
