/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

${define_active_storage_type("buffer")}

layout(std430) buffer;

#include "indexing.glslh"

// Output buffer: packed int8x4 values (each int32 contains 4 packed int8)
${layout_declare_tensor(B, "w", "t_outp", "int", "buffer")}
// Input staging buffer: raw int8 data interpreted as int32 for device compat
${layout_declare_tensor(B, "r", "nchw_in", "int", "buffer")}

// Metadata for output tensor
${layout_declare_ubo(B, "BufferMetadata", "outp")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "outp_layout", "CONTIG_LAYOUT_INT")}

void main() {
  const uint texel_idx = gl_GlobalInvocationID.x;
  const uint num_texels = numel(outp) / 4;
  if (texel_idx >= num_texels) {
    return;
  }

  const int inner_dim = get_packed_dim(outp_layout);
  const int outer_dim = get_outer_packed_dim(outp_layout);

  const TensorIndex4D tidx =
      texel_idx_to_tensor4d_idx(outp, texel_idx, outp_layout);

  // Bounds check on outer dimension
  if (tidx.data[outer_dim] >= int(outp.sizes[0][outer_dim])) {
    return;
  }

  // Tensor sizes in WHCN order for NCHW contiguous index computation
  const uint W = outp.sizes[0][0];
  const uint H = outp.sizes[0][1];
  const uint C = outp.sizes[0][2];

  // Pack 4 int8 values along inner dimension into one int32
  int packed = 0;
  [[unroll]] for (int i = 0; i < 4; ++i) {
    const int elem_inner = tidx.data[inner_dim] + i;
    if (elem_inner < int(outp.sizes[0][inner_dim])) {
      // Build element coordinates
      ivec4 elem = tidx.data;
      elem[inner_dim] = elem_inner;

      // Compute NCHW contiguous index: w + h*W + c*H*W + n*C*H*W
      const uint nchw_idx = uint(elem[0]) + uint(elem[1]) * W +
                            uint(elem[2]) * H * W + uint(elem[3]) * C * H * W;

      // Read int8 from staging buffer (each int32 contains 4 bytes)
      const uint int_idx = nchw_idx >> 2;
      const uint byte_pos = nchw_idx & 3;
      const int staging_val = nchw_in[int_idx];
      const int byte_val = (staging_val >> (byte_pos * 8)) & 0xFF;

      packed |= (byte_val << (i * 8));
    }
  }

  t_outp[texel_idx] = packed;
}
