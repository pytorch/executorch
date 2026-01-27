/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions("texture3d", DTYPE)}
${define_explicit_type_extensions(DTYPE)}

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_type(DTYPE)}
#define T ${buffer_scalar_type(DTYPE)}

${define_active_storage_type("texture3d")}

layout(std430) buffer;

#include "indexing_utils.h"

${layout_declare_tensor(B, "w", "t_out", DTYPE, "texture3d")}
${layout_declare_tensor(B, "r", "t_in", DTYPE, "texture3d")}

layout(push_constant) uniform restrict Block {
  ivec4 out_sizes;
  ivec4 in_sizes;
  ivec4 permute_dims; // Permutation mapping: permute_dims[i] = j means output dim i comes from input dim j
};

${layout_declare_spec_const(C, "int", "out_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 out_axis_map = unhash_axis_map(out_layout);
const lowp int out_packed_dim = unhash_packed_dim(out_layout);

${layout_declare_spec_const(C, "int", "in_layout", "DEFAULT_LAYOUT")}
const lowp ivec4 in_axis_map = unhash_axis_map(in_layout);
const lowp int in_packed_dim = unhash_packed_dim(in_layout);

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Convert output tensor index to input tensor index based on permutation
ivec4 out_tidx_to_in_tidx(const ivec4 out_tidx) {
  ivec4 in_tidx;

  // Apply the permutation mapping: in_tidx[permute_dims[i]] = out_tidx[i]
  in_tidx[permute_dims.x] = out_tidx.x;
  in_tidx[permute_dims.y] = out_tidx.y;
  in_tidx[permute_dims.z] = out_tidx.z;
  in_tidx[permute_dims.w] = out_tidx.w;

  return in_tidx;
}

// Check if we can use the fast path where texels from the input tensor can be
// copied directly into the output tensor. This occurs when the packed dimension
// is preserved in the permutation, i.e. reading a texel from the output tensor
// produces 4 texels along the same dimension as reading a texel from the input
// tensor.
bool can_use_fast_path() {
  // Fast path is possible when the packed dimension is preserved in the permutation
  // This means permute_dims[out_packed_dim] == in_packed_dim
  return permute_dims[out_packed_dim] == in_packed_dim;
}

void main() {
  const ivec3 lpos = ivec3(gl_GlobalInvocationID);
  ivec4 out_tidx = lpos_to_tidx(lpos, out_sizes, out_axis_map.w, out_packed_dim);

  if (any(greaterThanEqual(out_tidx, out_sizes))) {
    return;
  }

  if (can_use_fast_path()) {
    // Fast path: packed dimension is preserved, so we can copy texels directly
    ivec4 in_tidx = out_tidx_to_in_tidx(out_tidx);
    ivec3 in_pos = tidx_to_pos(in_tidx, in_sizes, in_axis_map, in_packed_dim);
    VEC4_T in_texel = VEC4_T(load_texel(t_in, in_pos));

    write_texel_lpos(t_out, lpos, in_texel, out_axis_map);
  }
  else {
    // Slow path: packed dimension is not preserved, so each element of the
    // output texel may be "sourced" from a different texel in the input tensor.
    // Therefore each output texel element is processed individually.
    VEC4_T out_texel = VEC4_T(0);

    for (int texel_i = 0; texel_i < 4; ++texel_i) {
      ivec4 in_tidx = out_tidx_to_in_tidx(out_tidx);
      ivec3 in_pos = tidx_to_pos(in_tidx, in_sizes, in_axis_map, in_packed_dim);
      int element_idx = in_tidx[in_packed_dim] % 4;

      VEC4_T in_texel = VEC4_T(load_texel(t_in, in_pos));
      T selected_value = T(in_texel[element_idx]);

      out_texel[texel_i] = selected_value;

      out_tidx[out_packed_dim]++;
    }

    write_texel_lpos(t_out, lpos, out_texel, out_axis_map);
  }
}
