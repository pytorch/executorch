/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions("buffer", DTYPE)}

#extension GL_EXT_debug_printf : enable

#define PRECISION ${PRECISION}

${define_active_storage_type("buffer")}

#define DEBUG_MODE

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_tensor(B, "r", "t_inp", DTYPE, "buffer")}

${layout_declare_ubo(B, "BufferMetadata", "inp")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "inp_layout", "CONTIG_LAYOUT_INT")}

layout(push_constant) uniform restrict Block {
  int value_ref;
};

void main() {
  const int W = int(min(5, size_at(inp, 0)));
  const int H = int(min(5, size_at(inp, 1)));

  debugPrintfEXT(
      "\\n[print_buffer] value_ref=%d, sizes=(%u, %u, %u, %u), printing %dx%d plane at c=0, n=0",
      value_ref,
      size_at(inp, 0), size_at(inp, 1), size_at(inp, 2), size_at(inp, 3),
      W, H);

  for (int y = 0; y < H; y++) {
    if (y == 0) {
      debugPrintfEXT("\\n[[");
    } else {
      debugPrintfEXT("\\n [");
    }

    for (int x = 0; x < W; x++) {
      TensorIndex4D t;
      t.data = ivec4(x, y, 0, 0);
      int buf_idx = tensor4d_idx_to_buf_idx(inp, t, inp_layout);
      float v = float(t_inp[buf_idx]);
      if (x < W - 1) {
        debugPrintfEXT("%8.4f, ", v);
      } else {
        debugPrintfEXT("%8.4f", v);
      }
    }

    if (y == H - 1) {
      debugPrintfEXT("]]\\n");
    } else {
      debugPrintfEXT("]");
    }
  }
}
