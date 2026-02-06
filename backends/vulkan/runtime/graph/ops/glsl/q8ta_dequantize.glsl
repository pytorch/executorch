/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions(OUTPUT_STORAGE, DTYPE)}

#define PRECISION ${PRECISION}
#define T ${buffer_scalar_type(DTYPE)}
$if OUTPUT_STORAGE == "texture3d":
  #define VEC4_T ${texel_load_type(DTYPE, "texture3d")}

$if OUTPUT_STORAGE == "buffer":
  ${define_active_storage_type("buffer")}
$else:
  ${define_active_storage_type("texture3d")}

layout(std430) buffer;

#include "indexing.glslh"

// Output: dequantized floating point values (buffer or texture)
${layout_declare_tensor(B, "w", "t_outp", DTYPE, OUTPUT_STORAGE)}
// Input buffer: quantized int32 values (each int32 contains 4 packed int8s)
${layout_declare_tensor(B, "r", "t_inp", "int", "buffer")}

// Metadata for output tensor (floating point) - buffer or texture
$if OUTPUT_STORAGE == "buffer":
  ${layout_declare_ubo(B, "BufferMetadata", "outp")}
$else:
  ${layout_declare_ubo(B, "TextureMetadata", "outp")}
// Metadata for input tensor (quantized int8x4) - always buffer
${layout_declare_ubo(B, "BufferMetadata", "inp")}

layout(push_constant) uniform restrict Block {
  float scale;
  int zp;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "outp_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "inp_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "outp_block_config", "0")}
${layout_declare_spec_const(C, "int", "inp_block_config", "0")}

#include "block_indexing.glslh"
#include "block_int8x4_load.glslh"
#include "block_store.glslh"

// Generate loading functions for t_inp buffer
define_load_int8x4_buffer_fns(t_inp)
// Generate storing functions for t_outp
$if OUTPUT_STORAGE == "buffer":
  define_store_buffer_fns(t_outp, T)
$else:
  define_store_texture_fns(t_outp, VEC4_T)

mat4 dequantize_int8x4_block(
    const ivec4 block, const float scale, const int zp) {
  mat4 result;
  [[unroll]] for (int i = 0; i < 4; ++i) {
    // Unpack 4 int8 values from packed int32
    int packed = block[i];
    ivec4 unpacked = ivec4(
        (packed >> 0) & 0xFF,
        (packed >> 8) & 0xFF,
        (packed >> 16) & 0xFF,
        (packed >> 24) & 0xFF);
    // Sign extend from 8-bit
    unpacked = (unpacked ^ 0x80) - 0x80;
    // Dequantize: (q - zp) * scale
    result[i] = vec4(unpacked - zp) * scale;
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

  // Load int8 block from input using the thread's block index
  const int inp_block_outer_dim = get_block_outer_dim(inp_block_config);
  ivec4 int8_block = load_int8x4_block_from_t_inp(
      inp, tidx, inp_layout, inp_block_outer_dim);

  // If input and output have different block configs (different packed dims),
  // transpose the block to match output's layout
  if (inp_block_config != outp_block_config) {
    int8_block = transpose_int8x4_block(int8_block);
  }

  // Dequantize the int8 block to float values
  mat4 fp_block = dequantize_int8x4_block(int8_block, scale, zp);

  // Store dequantized values to output buffer using output's block config
  const int outp_block_outer_dim = get_block_outer_dim(outp_block_config);
  store_fp_block_to_t_outp(
      outp, tidx, outp_layout, outp_block_outer_dim, fp_block);
}
