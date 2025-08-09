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

${layout_declare_tensor(B, "rw", "t_out", DTYPE, "buffer")}

$for i in range(NUM_INPUTS):
  ${layout_declare_tensor(B, "r", "t_inp" + str(i), DTYPE, "buffer")}

${layout_declare_tensor(B, "r", "t_concat_offset", "int", "buffer")}

${layout_declare_ubo(B, "int", "concat_dim")}

${layout_declare_ubo(B, "ivec4", "out_sizes")}
${layout_declare_ubo(B, "ivec4", "out_strides")}

$for i in range(NUM_INPUTS):
  ${layout_declare_ubo(B, "ivec4", "inp" + str(i) + "_sizes")}
  ${layout_declare_ubo(B, "ivec4", "inp" + str(i) + "_strides")}

${layout_declare_ubo(B, "int", "out_numel")}

${layout_declare_spec_const(C, "int", "out_layout", "DEFAULT_LAYOUT")}

const lowp ivec4 out_dim_order = unhash_dim_order(out_layout);

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#define NUM_INPUTS ${NUM_INPUTS}

#include "concat_utils.glslh"

/*
 * This shader template concatenates up to NUM_INPUT input tensors to the
 * output tensor along the concat_dim. Elements from the input tensor will
 * be inserted along the output's concat_dim starting at concat_offset.
 */
void main() {
  const int tid = ivec3(gl_GlobalInvocationID).x;

  // The 1-3 input tensors are interpreted as one concatenated tensor ("volume")
  // along the concat_dim for the purposes of tensor indexing. Each thread is
  // responsible for reading one item from this volume and writing it to the
  // appropriate output location.
  ivec4 inp_volume_sizes = out_sizes;
  inp_volume_sizes[concat_dim] = total_concat_dim_numel();

  // Account for 0 size input tensors
  if (any(lessThanEqual(inp_volume_sizes, ivec4(0)))) {
    return;
  }

  ivec4 inp_volume_tidx = nchwi_to_tidx(tid, inp_volume_sizes);

  // bounds check
  if (any(greaterThanEqual(inp_volume_tidx, inp_volume_sizes))) {
    return;
  }

  int concat_offset = t_concat_offset[0];

  ivec4 out_tidx = inp_volume_tidx;
  out_tidx[concat_dim] += concat_offset;

  const uint out_bufi = tidx_to_bufi(out_tidx, out_strides);

  // Go through the list of input tensors, and find which input this output
  // element should be read from.
  $for i in range(NUM_INPUTS):
    if (inp_volume_tidx[concat_dim] < inp${i}_sizes[concat_dim]) {
      int inp_bufi = tidx_to_bufi(inp_volume_tidx, inp${i}_strides);
      t_out[out_bufi] = t_inp${i}[inp_bufi];
      return;
    }
    else {
      inp_volume_tidx[concat_dim] -= inp${i}_sizes[concat_dim];
    }
}
