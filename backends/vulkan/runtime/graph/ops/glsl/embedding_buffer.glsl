/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define T ${buffer_scalar_type(DTYPE)}

${define_active_storage_type("buffer")}
${define_required_extensions(DTYPE)}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

#define DEBUG_MODE
#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_indices", "int", "buffer")}
${layout_declare_tensor(B, "r", "t_weight", DTYPE, "buffer")}

${layout_declare_ubo(B, "BufferMetadata", "outp")}
${layout_declare_ubo(B, "BufferMetadata", "indices")}
${layout_declare_ubo(B, "BufferMetadata", "weight")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

TensorIndex out_tidx_to_indices_tidx(const TensorIndex out_tidx) {
  TensorIndex indices_tidx;
  int d = 0;
  // First half of the index
  [[unroll]] for (uint d = 0; d < ndim(indices); ++d) {
    indices_tidx.data[div_4(d)][mod_4(d)] = idx_at(out_tidx, d + 1);
  }
  [[unroll]] for (uint d = ndim(indices); d < DIMLIMIT; ++d) {
    indices_tidx.data[div_4(d)][mod_4(d)] = 0;
  }
  return indices_tidx;
}

int load_embedding_idx(const TensorIndex indices_tidx) {
  const uint bufi = tensor_idx_to_linear_idx(indices, indices_tidx);
  return t_indices[bufi];
}

T load_weight_elem(const int embedding_idx, const uint dim_idx) {
  uint bufi = uint(embedding_idx) * width(weight) + dim_idx;
  return t_weight[bufi];
}

void main() {
  const uint out_bufi = gl_GlobalInvocationID.x;
  if (out_of_bounds(out_bufi, outp)) {
    return;
  }

  TensorIndex out_tidx = linear_idx_to_tensor_idx(outp, out_bufi);
  TensorIndex indices_tidx = out_tidx_to_indices_tidx(out_tidx);

  const uint bufi = tensor_idx_to_linear_idx(indices, indices_tidx);
  const int embedding_idx = load_embedding_idx(indices_tidx);

  t_out[out_bufi] = load_weight_elem(embedding_idx, x(out_tidx));
}
