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

${define_required_extensions(DTYPE)}

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_outp", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_inp", DTYPE, "buffer")}

${layout_declare_ubo(B, "BufferMetadata", "outp")}
${layout_declare_ubo(B, "BufferMetadata", "inp")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const uint outp_bufi = gl_GlobalInvocationID.x;
  if (outp_bufi >= numel(outp)) {
    return;
  }

  TensorIndex outp_tidx;
  linear_idx_to_tensor_idx(outp, outp_bufi, outp_tidx);

  // Map output tensor index to input tensor index by taking modulo
  // with input tensor sizes for each dimension
  TensorIndex inp_tidx = outp_tidx;
  for (int d = 0; d < ndim(inp); ++d) {
    uint inp_size = size_at(inp, d);
    uint outp_idx = idx_at(outp_tidx, d);
    inp_tidx.data[div_4(d)][mod_4(d)] = outp_idx % inp_size;
  }

  const uint inp_bufi = tensor_idx_to_linear_idx(inp, inp_tidx);
  // Copy data from input to output
  t_outp[outp_bufi] = t_inp[inp_bufi];
}
