/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions("buffer", DTYPE)}

#define PRECISION ${PRECISION}

#define T ${buffer_scalar_type(DTYPE)}

${define_active_storage_type("buffer")}

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_tensor(B, "rw", "t_out", DTYPE, "buffer")}

$for i in range(NUM_INPUTS):
  ${layout_declare_tensor(B, "r", "t_inp" + str(i), DTYPE, "buffer")}

${layout_declare_tensor(B, "r", "t_concat_offset", "int", "buffer")}

${layout_declare_ubo(B, "BufferMetadata", "outp")}

$for i in range(NUM_INPUTS):
  ${layout_declare_ubo(B, "BufferMetadata", "inp" + str(i) + "p")}

${layout_declare_spec_const(C, "int", "concat_dim", "0")}
${layout_declare_spec_const(C, "int", "out_layout", "CONTIG_LAYOUT_INT")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#define NUM_INPUTS ${NUM_INPUTS}

/*
 * This shader template concatenates up to NUM_INPUT input tensors to the
 * output tensor along the concat_dim. Elements from the input tensor will
 * be inserted along the output's concat_dim starting at concat_offset.
 */
void main() {
  const int tid = int(gl_GlobalInvocationID.x);

  // The 1-3 input tensors are interpreted as one concatenated tensor ("volume")
  // along the concat_dim for the purposes of tensor indexing. Each thread is
  // responsible for reading one item from this volume and writing it to the
  // appropriate output location.

  // Compute inp_volume_sizes from output sizes, replacing concat_dim with the
  // sum of all input sizes along that dimension.
  ivec4 inp_volume_sizes = ivec4(outp.sizes[0]);
  int total_concat_dim = 0;
  $for i in range(NUM_INPUTS):
    total_concat_dim += int(safe_idx(inp${i}p.sizes[0], concat_dim));
  safe_set(inp_volume_sizes, concat_dim, total_concat_dim);

  // Account for 0 size input tensors
  if (any(lessThanEqual(inp_volume_sizes, ivec4(0)))) {
    return;
  }

  // Decompose flat index into 4D tensor index (contiguous WHCN layout)
  TensorIndex4D inp_volume_tidx;
  {
    int remaining = tid;
    const int div_x = remaining / inp_volume_sizes.x;
    const int div_xy = div_x / inp_volume_sizes.y;
    inp_volume_tidx.data = ivec4(
        remaining % inp_volume_sizes.x,
        div_x % inp_volume_sizes.y,
        div_xy % inp_volume_sizes.z,
        div_xy / inp_volume_sizes.z);
  }

  // bounds check
  if (any(greaterThanEqual(inp_volume_tidx.data, inp_volume_sizes))) {
    return;
  }

  int concat_offset = t_concat_offset[0];

  TensorIndex4D out_tidx = inp_volume_tidx;
  out_tidx.data[concat_dim] += concat_offset;

  const uint out_bufi = tensor4d_idx_to_linear_idx(outp, out_tidx);

  // Go through the list of input tensors, and find which input this output
  // element should be read from.
  $for i in range(NUM_INPUTS):
    if (inp_volume_tidx.data[concat_dim] < int(safe_idx(inp${i}p.sizes[0], concat_dim))) {
      uint inp_bufi = tensor4d_idx_to_linear_idx(inp${i}p, inp_volume_tidx);
      t_out[out_bufi] = t_inp${i}[inp_bufi];
      return;
    }
    else {
      inp_volume_tidx.data[concat_dim] -= int(safe_idx(inp${i}p.sizes[0], concat_dim));
    }
}
