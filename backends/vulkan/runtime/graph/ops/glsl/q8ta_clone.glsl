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

// Output buffer: packed int8x4 values
${layout_declare_tensor(B, "w", "t_outp", "int", "buffer")}
// Input buffer: packed int8x4 values
${layout_declare_tensor(B, "r", "t_inp", "int", "buffer")}

// Metadata for output tensor
${layout_declare_ubo(B, "BufferMetadata", "outp")}
// Metadata for input tensor
${layout_declare_ubo(B, "BufferMetadata", "inp")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "inp_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "outp_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "inp_block_config", "0")}
${layout_declare_spec_const(C, "int", "outp_block_config", "0")}

#include "block_indexing.glslh"
#include "block_int8x4_load.glslh"
#include "block_int8x4_store.glslh"

// Generate loading functions for t_inp buffer
define_load_int8x4_buffer_fns(t_inp)

// Generate storing functions for t_outp buffer
define_store_int8x4_buffer_fns(t_outp)

void main() {
  TensorIndex4D tidx;

  // Buffer storage: use linear dispatch
  const uint contig_block_idx = gl_GlobalInvocationID.x;
  tidx = contiguous_block_idx_to_tensor4d_idx_with_block_config(
      inp, contig_block_idx, inp_block_config);

  if (out_of_bounds(tidx, inp)) {
    return;
  }

  // Load int8x4 block from input using the thread's block index
  const int inp_block_outer_dim = get_block_outer_dim(inp_block_config);
  ivec4 int8_block = load_int8x4_block_from_t_inp(
      inp, tidx, inp_layout, inp_block_outer_dim);

  // If input and output have different block configs (different packed dims),
  // transpose the block to match output's layout
  if (inp_block_config != outp_block_config) {
    int8_block = transpose_int8x4_block(int8_block);
  }

  // Store values to output buffer using output's block config
  const int outp_block_outer_dim = get_block_outer_dim(outp_block_config);
  store_int8x4_block_to_t_outp(
      outp, tidx, outp_layout, outp_block_outer_dim, int8_block);
}
