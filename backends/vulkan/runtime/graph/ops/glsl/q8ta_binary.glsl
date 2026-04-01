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

#define op(X, Y) ${OPERATOR}

layout(std430) buffer;

#include "indexing.glslh"
#include "common.glslh"
#include "block_indexing.glslh"
#include "block_int8x4_load.glslh"
#include "block_int8x4_store.glslh"

// Output buffer: packed int8x4 values
${layout_declare_tensor(B, "w", "t_out", "int", "buffer")}
// Input buffers: packed int8x4 values
${layout_declare_tensor(B, "r", "t_in_a", "int", "buffer")}
${layout_declare_tensor(B, "r", "t_in_b", "int", "buffer")}

// Metadata for output and input tensors
${layout_declare_ubo(B, "BufferMetadata", "out_meta")}
${layout_declare_ubo(B, "BufferMetadata", "in_a_meta")}
${layout_declare_ubo(B, "BufferMetadata", "in_b_meta")}

layout(push_constant) uniform restrict Block {
  float input_a_scale;
  int input_a_zp;
  float input_b_scale;
  int input_b_zp;
  float output_inv_scale;
  int output_zp;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "out_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "in_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "other_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "block_config", "0")}

// Generate loading functions for input buffers
define_load_int8x4_buffer_fns(t_in_a)
define_load_int8x4_buffer_fns(t_in_b)

// Generate storing functions for output buffer
define_store_int8x4_buffer_fns(t_out)

void main() {
  // Buffer storage: use linear dispatch
  const uint contig_block_idx = gl_GlobalInvocationID.x;
  TensorIndex4D tidx = contiguous_block_idx_to_tensor4d_idx_with_block_config(
      out_meta, contig_block_idx, block_config);

  if (out_of_bounds(tidx, out_meta)) {
    return;
  }

  const int block_outer_dim = get_block_outer_dim(block_config);

  // Load int8x4 blocks from both inputs
  ivec4 in_block_a = load_int8x4_block_from_t_in_a(
      in_a_meta, tidx, in_layout, block_outer_dim);
  ivec4 in_block_b = load_int8x4_block_from_t_in_b(
      in_b_meta, tidx, other_layout, block_outer_dim);

  ivec4 out_block;

  for (int row = 0; row < 4; row++) {
    vec4 in_texel_a = unpack_and_dequantize(
        in_block_a[row], input_a_scale, input_a_zp);
    vec4 in_texel_b = unpack_and_dequantize(
        in_block_b[row], input_b_scale, input_b_zp);

    vec4 out_texel = op(in_texel_a, in_texel_b);
    out_block[row] = quantize_and_pack(out_texel, output_inv_scale, output_zp);
  }

  // Store to output buffer
  store_int8x4_block_to_t_out(
      out_meta, tidx, out_layout, block_outer_dim, out_block);
}
