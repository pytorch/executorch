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

#define VEC4_T ${texel_type(DTYPE)}
#define T ${buffer_scalar_type(DTYPE)}

${define_active_storage_type("buffer")}

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

#include "dispatch.glslh"

int inverse_permute_dim(const int inp_dim) {
  [[unroll]] for (int d = 0; d < DIMLIMIT; ++d) {
    if (extract_4b(permute_order, d) == inp_dim) {
      return d;
    }
  }
  return inp_dim;
}

TensorIndex unpermute(const TensorIndex outp_tidx) {
  TensorIndex inp_tidx;
  inp_tidx.data[0] = uvec4(
      idx_at(outp_tidx, inverse_permute_dim(0)),
      idx_at(outp_tidx, inverse_permute_dim(1)),
      idx_at(outp_tidx, inverse_permute_dim(2)),
      idx_at(outp_tidx, inverse_permute_dim(3)));
  inp_tidx.data[1] = uvec4(
      idx_at(outp_tidx, inverse_permute_dim(4)),
      idx_at(outp_tidx, inverse_permute_dim(5)),
      idx_at(outp_tidx, inverse_permute_dim(6)),
      idx_at(outp_tidx, inverse_permute_dim(7)));
  return inp_tidx;
}

void main() {
  const uint outp_bufi = linear_idx_from_gid();
  if (outp_bufi >= numel(outp)) {
    return;
  }

  const TensorIndex outp_tidx =
      linear_idx_to_tensor_idx(outp, outp_bufi, outp_layout);
  if (out_of_bounds(outp_tidx, outp)) {
    t_outp[outp_bufi] = T(0);
    return;
  }

  const TensorIndex inp_tidx = unpermute(outp_tidx);
  const uint inp_bufi = tensor_idx_to_linear_idx(inp, inp_tidx);

  // Copy data from input to output
  t_outp[outp_bufi] = t_inp[inp_bufi];
}
