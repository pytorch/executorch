/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions(STORAGE, DTYPE)}

#define PRECISION ${PRECISION}

#define T ${buffer_scalar_type(DTYPE)}

${define_active_storage_type(STORAGE)}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_outp", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "t_inp", DTYPE, STORAGE)}

${layout_declare_ubo(B, "BufferMetadata", "outp")}
${layout_declare_ubo(B, "BufferMetadata", "inp")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "outp_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "inp_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "upscale_factor", "1")}

/*
 * pixel_shuffle: rearranges (N, C*r*r, H, W) -> (N, C, H*r, W*r).
 *
 * For output element at NCHW index (n, c, h_out, w_out):
 *   h_in = h_out / r
 *   w_in = w_out / r
 *   c_in = c * r * r + (h_out % r) * r + (w_out % r)
 *
 * The W, H, C dims correspond to NCHW indices [3], [2], [1] when ndim == 4
 * (ndim - 1, ndim - 2, ndim - 3 in general). We use NCHW dim numbering so the
 * mapping is independent of the tensor's memory layout.
 */
void main() {
  const uint outp_bufi = gl_GlobalInvocationID.x;
  if (outp_bufi >= numel(outp)) {
    return;
  }

  TensorIndex outp_tidx = linear_idx_to_tensor_idx(outp, outp_bufi);

  // NCHW dim indices for W (last), H (second-last), C (third-last).
  const int nd = int_ndim(outp);
  const int w_dim_nchw = nd - 1;
  const int h_dim_nchw = nd - 2;
  const int c_dim_nchw = nd - 3;

  // Convert NCHW dim index to WHCN dim index expected by indexing helpers,
  // where the "dim" parameter is ndim - 1 - nchw_dim (i.e. logical ordering
  // matching strides). The tidx.data array stores indices in WHCN order:
  // tidx.data[0] = W, tidx.data[1] = H, tidx.data[2] = C, tidx.data[3] = N.
  const int r = upscale_factor;

  const uint w_out = idx_at(outp_tidx, 0);
  const uint h_out = idx_at(outp_tidx, 1);
  const uint c_out = idx_at(outp_tidx, 2);

  const uint w_in = w_out / uint(r);
  const uint h_in = h_out / uint(r);
  const uint c_in = c_out * uint(r) * uint(r) +
      (h_out % uint(r)) * uint(r) + (w_out % uint(r));

  TensorIndex inp_tidx = outp_tidx;
  inp_tidx.data[0][0] = w_in;
  inp_tidx.data[0][1] = h_in;
  inp_tidx.data[0][2] = c_in;

  const uint inp_bufi = tensor_idx_to_linear_idx(inp, inp_tidx);

  t_outp[outp_bufi] = t_inp[inp_bufi];
}
