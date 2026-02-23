/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions(INPUT_STORAGE, DTYPE)}

#define PRECISION ${PRECISION}
#define T ${buffer_scalar_type(DTYPE)}
$if INPUT_STORAGE == "texture3d":
  #define VEC4_T ${texel_load_type(DTYPE, "texture3d")}

$if INPUT_STORAGE == "buffer":
  ${define_active_storage_type("buffer")}
$else:
  ${define_active_storage_type("texture3d")}

layout(std430) buffer;

#include "indexing.glslh"

// Output buffer: quantized int32 values (each int32 contains 4 packed int8s)
${layout_declare_tensor(B, "w", "t_outp", "int", "buffer")}
// Input: floating point values (buffer or texture)
${layout_declare_tensor(B, "r", "t_inp", DTYPE, INPUT_STORAGE)}

// Metadata for output tensor (quantized int8x4) - always buffer
${layout_declare_ubo(B, "BufferMetadata", "outp")}
// Metadata for input tensor (floating point) - buffer or texture
$if INPUT_STORAGE == "buffer":
  ${layout_declare_ubo(B, "BufferMetadata", "inp")}
$else:
  ${layout_declare_ubo(B, "TextureMetadata", "inp")}

layout(push_constant) uniform restrict Block {
  float inv_scale;
  int zp;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "inp_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "outp_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "inp_block_config", "0")}
${layout_declare_spec_const(C, "int", "outp_block_config", "0")}

#include "block_indexing.glslh"
#include "block_load.glslh"
$if INPUT_STORAGE == "buffer":
  // Generate loading functions for t_inp buffer
  define_load_buffer_fns(t_inp)
$else:
  // Generate loading functions for t_inp texture
  define_load_texture_fns(t_inp)
#include "block_int8x4_store.glslh"

// Generate storing functions for t_outp buffer
define_store_int8x4_buffer_fns(t_outp)

ivec4 quantize_fp_block(
    const mat4 block, const float inv_scale, const int zp) {
  ivec4 result;
  [[unroll]] for (int i = 0; i < 4; ++i) {
    // Quantize: round(val * inv_scale) + zp, clamped to [-128, 127]
    ivec4 quantized = ivec4(round(block[i] * inv_scale)) + zp;
    quantized = clamp(quantized, -128, 127);
    // Pack 4 int8 values into one int32
    result[i] = ((quantized[0] & 0xFF) << 0) |
                ((quantized[1] & 0xFF) << 8) |
                ((quantized[2] & 0xFF) << 16) |
                ((quantized[3] & 0xFF) << 24);
  }
  return result;
}

void main() {
  TensorIndex4D tidx;

#ifdef USING_BUFFER
  // Buffer storage: use linear dispatch
  const uint contig_block_idx = gl_GlobalInvocationID.x;
  tidx = contiguous_block_idx_to_tensor4d_idx_with_block_config(
      inp, contig_block_idx, inp_block_config);
#else
  // Texture storage: use 3D extents dispatch
  const uvec3 thread_idx = gl_GlobalInvocationID;
  tidx = block_idx_3d_to_tensor4d_idx_with_block_config(
      inp, thread_idx, inp_block_config);
#endif

  if (out_of_bounds(tidx, inp)) {
    return;
  }

  // Load FP block from input using the thread's block index
  const int inp_block_outer_dim = get_block_outer_dim(inp_block_config);
  mat4 fp_block = load_fp_block_from_t_inp(
      inp, tidx, inp_layout, inp_block_outer_dim);

  // If input and output have different block configs (different packed dims),
  // transpose the block to match output's layout
  if (inp_block_config != outp_block_config) {
    fp_block = transpose(fp_block);
  }

  // Quantize the float block to int8 values
  ivec4 int8_block = quantize_fp_block(fp_block, inv_scale, zp);

  // Store quantized values to output buffer using output's block config
  const int outp_block_outer_dim = get_block_outer_dim(outp_block_config);
  store_int8x4_block_to_t_outp(
      outp, tidx, outp_layout, outp_block_outer_dim, int8_block);
}
