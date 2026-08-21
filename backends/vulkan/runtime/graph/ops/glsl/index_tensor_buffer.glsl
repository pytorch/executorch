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

  const uint self_rank = ndim(inp);
  const uint index_rank = ndim(index);
  // WHCN order places self's trailing axes before the index axes.
  const uint index_axis_offset = self_rank - 1;

  uint index_bufi = 0;
  for (uint d = 0; d < index_rank; ++d) {
    index_bufi +=
        stride_at(index, d) * idx_at(out_tidx, index_axis_offset + d);
  }
  const int idx = t_index[index_bufi];

  uint self_bufi = stride_at(inp, self_rank - 1) * uint(idx);
  for (uint d = 0; d + 1 < self_rank; ++d) {
    self_bufi += stride_at(inp, d) * idx_at(out_tidx, d);
  }

  t_out[out_bufi] = t_self[self_bufi];
}
