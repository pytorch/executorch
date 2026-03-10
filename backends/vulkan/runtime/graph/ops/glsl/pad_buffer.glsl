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

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_in", DTYPE, "buffer")}

${layout_declare_ubo(B, "BufferMetadata", "out_meta")}
${layout_declare_ubo(B, "BufferMetadata", "in_meta")}
${layout_declare_ubo(B, "ivec4", "pad_per_dim")}
${layout_declare_ubo(B, "float", "fill_value")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint out_bufi = gl_GlobalInvocationID.x;
  if (out_bufi >= numel(out_meta)) {
    return;
  }

  TensorIndex out_tidx = linear_idx_to_tensor_idx(out_meta, out_bufi);

  // Subtract pad offsets per dimension to get input tensor index.
  // Unsigned underflow (when output index < pad offset) wraps to a large
  // value that fails the out_of_bounds check below.
  TensorIndex in_tidx = out_tidx;
  [[unroll]] for (int d = 0; d < 4; d++) {
    in_tidx.data[0][d] -= uint(pad_per_dim[d]);
  }

  if (out_of_bounds(in_tidx, in_meta)) {
    t_out[out_bufi] = T(fill_value);
    return;
  }

  const uint in_bufi = tensor_idx_to_linear_idx(in_meta, in_tidx);
  t_out[out_bufi] = t_in[in_bufi];
}
