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

  // Read the index value at the same tensor position
  const uint index_bufi = tensor_idx_to_linear_idx(index, out_tidx);
  const int idx = t_index[index_bufi];

  // Construct a tensor index for the 1D self tensor.
  // In WHCN ordering, a 1D tensor has its elements along dim 0 (width).
  TensorIndex self_tidx;
  self_tidx.data[0] = uvec4(uint(idx), 0, 0, 0);
  self_tidx.data[1] = uvec4(0);
  const uint self_bufi = tensor_idx_to_linear_idx(inp, self_tidx);

  t_out[out_bufi] = t_self[self_bufi];
}
