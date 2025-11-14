/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define T ${buffer_scalar_type(DTYPE)}

${define_active_storage_type("buffer")}
${define_required_extensions(DTYPE)}

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_out", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_input", DTYPE, "buffer")}
${layout_declare_tensor(B, "r", "t_index", "int", "buffer")}

${layout_declare_ubo(B, "BufferMetadata", "outp")}
${layout_declare_ubo(B, "BufferMetadata", "inp")}
${layout_declare_ubo(B, "BufferMetadata", "index")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int gather_dim = 0;

void main() {
  const uint out_bufi = gl_GlobalInvocationID.x;
  if (out_of_bounds(out_bufi, outp)) {
    return;
  }

  TensorIndex out_tidx = linear_idx_to_tensor_idx(outp, out_bufi);

  // Load the index value at the same position in the index tensor
  const uint index_bufi = tensor_idx_to_linear_idx(index, out_tidx);
  const int gather_idx = t_index[index_bufi];

  // Construct the input tensor index by replacing the gather dimension
  // with the gathered index value
  TensorIndex input_tidx = out_tidx;
  input_tidx.data[div_4(gather_dim)][mod_4(gather_dim)] = gather_idx;

  // Load from input tensor and store to output
  const uint input_bufi = tensor_idx_to_linear_idx(inp, input_tidx);

  t_out[out_bufi] = t_input[input_bufi];
}
