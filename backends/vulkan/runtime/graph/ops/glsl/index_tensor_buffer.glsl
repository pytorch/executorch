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

${layout_declare_tensor(B, "w", "t_out", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_self", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_index", "int", "buffer")}

${layout_declare_ubo(B, "BufferMetadata", "outp")}
${layout_declare_ubo(B, "BufferMetadata", "inp")}
${layout_declare_ubo(B, "BufferMetadata", "index")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Implements aten.index.Tensor for the case where self is 1D and there is
// exactly one index tensor. Each output element is:
//   output[...] = self[index[...]]

void main() {
  const uint out_bufi = gl_GlobalInvocationID.x;
  if (out_of_bounds(out_bufi, outp)) {
    return;
  }

  // Convert output buffer index to tensor index
  TensorIndex out_tidx = linear_idx_to_tensor_idx(outp, out_bufi);

  // aten.index.Tensor with a single index tensor gathers along dim 0. This
  // index space is WHCN-ordered -- axis 0 is the LAST pytorch dim -- so
  // pytorch dim 0 is the highest axis, and self's trailing dims map 1:1 onto
  // out's. With a 1-D index, rank(out) == rank(self).
  const uint gather_axis = ndim(outp) - 1;
  const uint gather_pos = idx_at(out_tidx, gather_axis);

  // The index tensor is 1-D, so only its axis 0 (W) is populated. Indexing it
  // with out's full coordinate is equivalent only when out is itself 1-D.
  TensorIndex index_tidx;
  initialize(index_tidx);
  index_tidx.data[0][0] = gather_pos;
  const uint index_bufi = tensor_idx_to_linear_idx(index, index_tidx);
  const int idx = t_index[index_bufi];

  // self shares out's trailing coordinates; only the gathered axis differs.
  TensorIndex self_tidx = out_tidx;
  self_tidx.data[div_4(gather_axis)][mod_4(gather_axis)] = uint(idx);
  const uint self_bufi = tensor_idx_to_linear_idx(inp, self_tidx);

  t_out[out_bufi] = t_self[self_bufi];
}
