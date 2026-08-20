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

${layout_declare_spec_const(C, "int", "unfold_dim", "0")}
${layout_declare_spec_const(C, "int", "step", "1")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint out_bufi = gl_GlobalInvocationID.x;
  if (out_of_bounds(out_bufi, out_meta)) {
    return;
  }

  const TensorIndex out_tidx = linear_idx_to_tensor_idx(out_meta, out_bufi);
  TensorIndex in_tidx;
  initialize(in_tidx);

  for (int d = 0; d < int_ndim(in_meta); ++d) {
    in_tidx.data[div_4(d)][mod_4(d)] = idx_at(out_tidx, d + 1);
  }
  in_tidx.data[div_4(unfold_dim)][mod_4(unfold_dim)] =
      idx_at(out_tidx, unfold_dim + 1) * step + idx_at(out_tidx, 0);

  const uint in_bufi = tensor_idx_to_linear_idx(in_meta, in_tidx);
  t_out[out_bufi] = t_in[in_bufi];
}
