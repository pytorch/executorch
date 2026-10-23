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

#include "dispatch.glslh"

#define NUM_INPUTS ${NUM_INPUTS}

/*
 * This shader template concatenates up to NUM_INPUT input tensors to the
 * output tensor along the concat_dim. Elements from the input tensor will
 * be inserted along the output's concat_dim starting at concat_offset.
 * Supports arbitrary dimensions up to 8D using BufferMetadata.
 */
void main() {
  const int tid = int(linear_idx_from_gid());

  // The 1-3 input tensors are interpreted as one concatenated tensor ("volume")
  // along the concat_dim for the purposes of tensor indexing. Each thread is
  // responsible for reading one item from this volume and writing it to the
  // appropriate output location.

  // Compute total size along concat_dim
  int total_concat_dim = 0;
  $for i in range(NUM_INPUTS):
    total_concat_dim += int(size_at(inp${i}p, concat_dim));

  if (total_concat_dim <= 0) {
    return;
  }

  // Decompose flat index into N-D tensor index (contiguous WHCN layout)
  TensorIndex inp_volume_tidx;
  initialize(inp_volume_tidx);
  int remaining = tid;
  const int n = int_ndim(outp);
  for (int d = 0; d < n; ++d) {
    int dim_size = (d == concat_dim) ? total_concat_dim : int(size_at(outp, d));
    if (dim_size <= 0) {
      return;
    }
    inp_volume_tidx.data[div_4(d)][mod_4(d)] = remaining % dim_size;
    remaining = remaining / dim_size;
  }

  // Bounds check: if remaining > 0, tid is outside the input volume
  if (remaining > 0) {
    return;
  }

  int concat_offset = t_concat_offset[0];

  TensorIndex out_tidx = inp_volume_tidx;
  out_tidx.data[div_4(concat_dim)][mod_4(concat_dim)] += uint(concat_offset);

  const uint out_bufi = tensor_idx_to_linear_idx(outp, out_tidx);
  if (out_of_bounds(out_bufi, outp)) {
    return;
  }

  // Go through the list of input tensors, and find which input this output
  // element should be read from.
  $for i in range(NUM_INPUTS):
    uint in_dim_size_${i} = size_at(inp${i}p, concat_dim);
    if (idx_at(inp_volume_tidx, concat_dim) < in_dim_size_${i}) {
      uint inp_bufi = tensor_idx_to_linear_idx(inp${i}p, inp_volume_tidx);
      t_out[out_bufi] = t_inp${i}[inp_bufi];
      return;
    }
    else {
      inp_volume_tidx.data[div_4(concat_dim)][mod_4(concat_dim)] -= in_dim_size_${i};
    }
}
