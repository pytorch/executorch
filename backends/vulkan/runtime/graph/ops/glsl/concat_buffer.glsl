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

$for i in range(NUM_INPUTS):
  ${layout_declare_tensor(B, "r", "t_in" + str(i + 1), DTYPE, "buffer")}

${layout_declare_ubo(B, "int", "concat_dim")}

${layout_declare_ubo(B, "ivec4", "out_sizes")}
${layout_declare_ubo(B, "ivec4", "out_strides")}

$for i in range(NUM_INPUTS):
  ${layout_declare_ubo(B, "ivec4", "in" + str(i+1) + "_sizes")}
  ${layout_declare_ubo(B, "ivec4", "in" + str(i+1) + "_strides")}

${layout_declare_ubo(B, "int", "out_numel")}

${layout_declare_spec_const(C, "int", "out_packed_dim", "DEFAULT_LAYOUT")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const int out_bufi = ivec3(gl_GlobalInvocationID).x;
  if (out_bufi >= out_numel) {
    return;
  }

  // Convert buffer linear index to 4-D tensor index for output
  const ivec4 out_tidx = bufi_to_tidx(out_bufi, out_strides, out_packed_dim);

  // Determine which input tensor to read from
  ivec4 in_tidx = out_tidx;

  $for i in range(NUM_INPUTS):
    // Check if the index at the concat dim is within bounds of the input tensor
    // If so, read from that input tensor and write to output
    if (in_tidx[concat_dim] < in${i+1}_sizes[concat_dim]) {
      int in_bufi = tidx_to_bufi(in_tidx, in${i+1}_strides);
      t_out[out_bufi] = t_in${i+1}[in_bufi];
      return;
    }
    // otherwise, decrement the index at the concat dim
    else {
      in_tidx[concat_dim] -= in${i+1}_sizes[concat_dim];
    }
}
