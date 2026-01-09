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

#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_outp", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_inp", DTYPE, "buffer")}

${layout_declare_ubo(B, "BufferMetadata", "outp")}
${layout_declare_ubo(B, "BufferMetadata", "inp")}

${layout_declare_spec_const(C, "int", "outp_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "inp_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "permute_order", "0")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

TensorIndex permute(TensorIndex tidx) {
  TensorIndex new_tidx = tidx;

  new_tidx.data[0][0] = idx_at(tidx, extract_4b(permute_order, 0));
  new_tidx.data[0][1] = idx_at(tidx, extract_4b(permute_order, 1));
  new_tidx.data[0][2] = idx_at(tidx, extract_4b(permute_order, 2));
  new_tidx.data[0][3] = idx_at(tidx, extract_4b(permute_order, 3));

  new_tidx.data[1][0] = idx_at(tidx, extract_4b(permute_order, 4));
  new_tidx.data[1][1] = idx_at(tidx, extract_4b(permute_order, 5));
  new_tidx.data[1][2] = idx_at(tidx, extract_4b(permute_order, 6));
  new_tidx.data[1][3] = idx_at(tidx, extract_4b(permute_order, 7));

  return new_tidx;
}

void main() {
  const uint inp_bufi = gl_GlobalInvocationID.x;
  if (inp_bufi >= numel(inp)) {
    return;
  }

  TensorIndex inp_tidx = linear_idx_to_tensor_idx(inp, inp_bufi, inp_layout);
  TensorIndex outp_tidx = permute(inp_tidx);
  const uint outp_bufi = tensor_idx_to_linear_idx(outp, outp_tidx);

  // Copy data from input to output
  t_outp[outp_bufi] = t_inp[inp_bufi];
}
