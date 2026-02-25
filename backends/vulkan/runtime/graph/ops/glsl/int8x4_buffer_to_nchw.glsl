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

// Output staging buffer: raw int8 data interpreted as int32 for device compat
${layout_declare_tensor(B, "w", "nchw_out", "int", "buffer")}
// Input buffer: packed int8x4 values (each int32 contains 4 packed int8)
${layout_declare_tensor(B, "r", "t_inp", "int", "buffer")}

// Metadata for input tensor
${layout_declare_ubo(B, "BufferMetadata", "inp")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "inp_layout", "CONTIG_LAYOUT_INT")}

void main() {
  // One thread per output int32 in the NCHW staging buffer.
  // Each output int32 holds 4 consecutive NCHW bytes.
  const uint out_int32_idx = gl_GlobalInvocationID.x;

  const uint W = inp.sizes[0][0];
  const uint H = inp.sizes[0][1];
  const uint C = inp.sizes[0][2];
  const uint N = inp.sizes[0][3];
  const uint total_numel = W * H * C * N;
  const uint num_out_int32s = (total_numel + 3u) / 4u;

  if (out_int32_idx >= num_out_int32s) {
    return;
  }

  int output_int32 = 0;
  [[unroll]] for (int j = 0; j < 4; ++j) {
    const uint nchw_idx = out_int32_idx * 4u + uint(j);
    if (nchw_idx >= total_numel) {
      break;
    }

    // Convert NCHW linear index to tensor4D (WHCN) coordinates.
    const uint w = nchw_idx % W;
    const uint h = (nchw_idx / W) % H;
    const uint c = (nchw_idx / (W * H)) % C;
    const uint n = nchw_idx / (W * H * C);

    TensorIndex4D tidx;
    tidx.data = ivec4(int(w), int(h), int(c), int(n));

    // tensor4d_idx_to_buf_idx returns a linear element index where
    // element_index / 4 is the int32 slot and element_index % 4 is the byte
    // position within that int32. This matches the packing order used by
    // nchw_to_int8x4_buffer when writing to the int8x4 buffer.
    const int elem_buf_idx = tensor4d_idx_to_buf_idx(inp, tidx, inp_layout);
    const int int8_val =
        (t_inp[elem_buf_idx / 4] >> ((elem_buf_idx % 4) * 8)) & 0xFF;

    output_int32 |= (int8_val << (j * 8));
  }

  nchw_out[out_int32_idx] = output_int32;
}
