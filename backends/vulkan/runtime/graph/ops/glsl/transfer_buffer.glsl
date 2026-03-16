/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions("buffer", DTYPE)}
#extension GL_EXT_control_flow_attributes : require

#define PRECISION ${PRECISION}
#define UBO_PARAMS ${UBO_PARAMS}

#define T ${buffer_scalar_type(DTYPE)}

${define_active_storage_type("buffer")}

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_in", DTYPE, "buffer")}

${layout_declare_ubo(B, "BufferMetadata", "outp")}
${layout_declare_ubo(B, "BufferMetadata", "inp")}

$if UBO_PARAMS:
  $if OP_NAME == "slice":
    ${layout_declare_ubo(B, "int", "start")}
    ${layout_declare_ubo(B, "int", "step")}

  $if OP_NAME == "select":
    ${layout_declare_ubo(B, "int", "index")}

layout(push_constant) uniform restrict Block {
  int selected_dim;
  $if not UBO_PARAMS:
    $if OP_NAME == "slice":
      int start;
      int step;

    $if OP_NAME == "select":
      int index;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#include "${OP_NAME}.glslh"

void main() {
  const uint out_bufi = gl_GlobalInvocationID.x;
  if (out_of_bounds(out_bufi, outp)) {
    return;
  }

  TensorIndex out_tidx = linear_idx_to_tensor_idx(outp, out_bufi);
  TensorIndex in_tidx = out_tidx_to_in_tidx(out_tidx);

  const uint in_bufi = tensor_idx_to_linear_idx(inp, in_tidx);
  t_out[out_bufi] = t_in[in_bufi];
}
