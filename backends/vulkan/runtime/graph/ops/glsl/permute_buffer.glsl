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
#define T ${buffer_scalar_type(DTYPE)}

${define_active_storage_type("buffer")}
${define_required_extensions(DTYPE)}

layout(std430) buffer;

#include "indexing_utils.h"

${layout_declare_tensor(B, "w", "t_out", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_in", DTYPE, "buffer")}

${layout_declare_ubo(B, "ivec4", "in_sizes")}
${layout_declare_ubo(B, "ivec4", "out_strides")}
${layout_declare_ubo(B, "int", "out_numel")}

layout(push_constant) uniform restrict Block {
  ivec4 in_strides;
  ivec4 permute_dims; // Permutation mapping: permute_dims[i] = j means output dim i comes from input dim j
};

${layout_declare_spec_const(C, "int", "out_layout", "DEFAULT_LAYOUT")}
${layout_declare_spec_const(C, "int", "in_layout", "DEFAULT_LAYOUT")}

const lowp ivec4 out_dim_order = unhash_dim_order(out_layout);

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

void main() {
  const int out_bufi = ivec3(gl_GlobalInvocationID).x;
  if (out_bufi >= out_numel) {
    return;
  }

  // Convert buffer index to tensor index for output
  const ivec4 out_tidx = bufi_to_tidx(out_bufi, out_strides, out_dim_order);

  // Convert output tensor index to input tensor index using permutation
  const ivec4 in_tidx = out_tidx_to_in_tidx(out_tidx);

  // Convert input tensor index back to buffer index
  const int in_bufi = tidx_to_bufi(in_tidx, in_strides);

  // Copy data from input to output
  t_out[out_bufi] = t_in[in_bufi];
}
