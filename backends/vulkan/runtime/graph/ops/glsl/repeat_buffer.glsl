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
${layout_declare_tensor(B, "r", "t_in", DTYPE, "buffer")}

${layout_declare_ubo(B, "BufferMetadata", "out_meta")}
${layout_declare_ubo(B, "BufferMetadata", "in_meta")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint out_bufi = gl_GlobalInvocationID.x;
  if (out_of_bounds(out_bufi, out_meta)) {
    return;
  }

  TensorIndex out_tidx = linear_idx_to_tensor_idx(out_meta, out_bufi);

  TensorIndex in_tidx;
  initialize(in_tidx);

  const int n = int_ndim(out_meta);
  for (int d = 0; d < n; d++) {
    in_tidx.data[div_4(d)][mod_4(d)] =
        idx_at(out_tidx, d) % size_at(in_meta, d);
  }

  const uint in_bufi = tensor_idx_to_linear_idx(in_meta, in_tidx);

  t_out[out_bufi] = t_in[in_bufi];
}
