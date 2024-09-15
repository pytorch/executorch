/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}
#define SCALAR_T ${texel_load_component_type(DTYPE, STORAGE)}

${define_active_storage_type(STORAGE)}

#include "indexing_utils.h"

${define_required_extensions(DTYPE)}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "t_out", DTYPE, STORAGE)}
${layout_declare_buffer(B, "r", "nchw_in", DTYPE)}
${layout_declare_ubo(B, "ivec4", "sizes")}
${layout_declare_ubo(B, "ivec4", "axis_map")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int packed_dim = C_DIM;

VEC4_T read_texel(ivec4 tensor_idx) {
  const ivec4 buf_indices = get_texel_nchw_buffer_ixs(
      tensor_idx,
      sizes,
      packed_dim);

  VEC4_T texel = VEC4_T(0);
  if (tensor_idx[packed_dim] < sizes[packed_dim]) {
    texel.x = SCALAR_T(nchw_in[buf_indices.x]);
  }
  if (tensor_idx[packed_dim] + 1 < sizes[packed_dim]) {
    texel.y = SCALAR_T(nchw_in[buf_indices.y]);
  }
  if (tensor_idx[packed_dim] + 2 < sizes[packed_dim]) {
    texel.z = SCALAR_T(nchw_in[buf_indices.z]);
  }
  if (tensor_idx[packed_dim] + 3 < sizes[packed_dim]) {
    texel.w = SCALAR_T(nchw_in[buf_indices.w]);
  }
  return texel;
}

void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const ivec4 tensor_idx = to_tensor_idx(pos, sizes, axis_map, packed_dim);
  if (any(greaterThanEqual(tensor_idx, sizes))) {
    return;
  }

  write_texel(t_out, pos, read_texel(tensor_idx));
}
