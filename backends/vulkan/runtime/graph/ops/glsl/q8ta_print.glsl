/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#extension GL_EXT_debug_printf : enable

#define PRECISION ${PRECISION}

${define_active_storage_type("buffer")}

#define DEBUG_MODE

layout(std430) buffer;

#include "indexing.glslh"

// Input buffer: packed int8x4 values
${layout_declare_tensor(B, "r", "t_inp", "int", "buffer")}

// Metadata for input tensor
${layout_declare_ubo(B, "BufferMetadata", "inp")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "inp_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "inp_block_config", "0")}

layout(push_constant) uniform restrict Block {
  int value_ref;
};

// Load a single int8 value from the buffer at the given tensor index
int get_value(TensorIndex4D tidx) {
  const int buf_idx = tensor4d_idx_to_buf_idx(inp, tidx, inp_layout);
  const int packed = t_inp[div_4(buf_idx)];
  const ivec4 vals = unpack_int8x4(packed);
  return vals[mod_4(buf_idx)];
}

void main() {
  const int N = int(min(50, size_at(inp, 0))); // width
  const int M = int(min(50, size_at(inp, 1))); // height

  debugPrintfEXT(
      "\\n[q8ta_print] value_ref=%d, sizes=(%u, %u, %u, %u), printing %dx%d plane at c=0, n=0",
      value_ref,
      size_at(inp, 0), size_at(inp, 1), size_at(inp, 2), size_at(inp, 3),
      N, M);

  // Print NxM plane of values at (w=0..N-1, h=0..M-1, c=0, n=0)
  for (int y = 0; y < M; y++) {
    if (y == 0) {
      debugPrintfEXT("\\n[[");
    } else {
      debugPrintfEXT("\\n [");
    }

    for (int x = 0; x < N; x++) {
      TensorIndex4D t;
      t.data = ivec4(x, y, 0, 0);
      int v = get_value(t);
      if (x < N - 1) {
        debugPrintfEXT("%4d, ", v);
      } else {
        debugPrintfEXT("%4d", v);
      }
    }

    if (y == M - 1) {
      debugPrintfEXT("]]\\n");
    } else {
      debugPrintfEXT("]");
    }
  }
}
