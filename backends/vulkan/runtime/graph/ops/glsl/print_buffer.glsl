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
  const uint buf_idx = gl_GlobalInvocationID.x;
  if (buf_idx >= inp.ndim_numel.y) {
    return;
  }

  float v = float(t_inp[buf_idx]);
  if (abs(v) > 1e5) {
    TensorIndex tidx = linear_idx_to_tensor_idx(inp, buf_idx);
    debugPrintfEXT(
        "[print_buffer] value_ref=%d, sizes=(%u, %u, %u, %u), idx=(%d, %d, %d, %d), value=%f\\n",
        value_ref,
        size_at(inp, 0), size_at(inp, 1), size_at(inp, 2), size_at(inp, 3),
        tidx.data[0].x, tidx.data[0].y, tidx.data[0].z, tidx.data[0].w,
        v);
  }
}
