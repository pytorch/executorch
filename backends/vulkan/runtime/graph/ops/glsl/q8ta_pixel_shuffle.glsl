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

#extension GL_EXT_control_flow_attributes : require

layout(std430) buffer;

#include "indexing.glslh"
#include "block_int8x4_utils.glslh"

// Output buffer: packed int8x4 (each int32 = 4 packed int8 along packed_dim)
${layout_declare_tensor(B, "w", "t_outp", "int", "buffer")}
// Input buffer: packed int8x4 (each int32 = 4 packed int8 along packed_dim)
${layout_declare_tensor(B, "r", "t_inp", "int", "buffer")}

// Metadata for output and input tensors. Both are channels-packed int8x4
// buffers but may use different block layouts (e.g. PACKED_INT8_4W4C vs
// PACKED_INT8_4C).
${layout_declare_ubo(B, "BufferMetadata", "outp")}
${layout_declare_ubo(B, "BufferMetadata", "inp")}

layout(push_constant) uniform restrict Block {
  float scale_in;
  float inv_scale_out;
  int zp_in;
  int zp_out;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "inp_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "outp_layout", "CONTIG_LAYOUT_INT")}
// Set by the host when input and output quantization params match, making the
// shader a pure byte shuffle.
${layout_declare_spec_const(C, "int", "passthrough", "0")}

/*
 * PixelShuffle with upscale factor 2:
 *   out[n, c, 2h + dy, 2w + dx] = in[n, 4c + 2dy + dx, h, w]
 *
 * Each thread handles one block of 4 output channels at the 2x2 output pixels
 * produced by one input pixel. The 4 input words holding input channels
 * [16 * c4, 16 * c4 + 16) at that pixel form a 4x4 byte matrix where byte
 * (2dy + dx) of word i is output channel 4 * c4 + i at (2w + dx, 2h + dy). Its
 * transpose is the 4 output words, so every loaded and stored byte is used.
 *
 * Global x indexes the channel block, y the input width, and z the input row
 * and batch.
 */

// Requantize the 4 int8 lanes of a packed word.
int requantize_word(const int packed) {
  int result = 0;
  [[unroll]] for (int i = 0; i < 4; ++i) {
    const int q_in = (packed << (24 - 8 * i)) >> 24;
    const float dq = float(q_in - zp_in) * scale_in;
    const int q_out =
        clamp(int(round(dq * inv_scale_out)) + zp_out, -128, 127);
    result |= (q_out & 0xFF) << (8 * i);
  }
  return result;
}

void main() {
  const int c4 = int(gl_GlobalInvocationID.x);
  const int iw = int(gl_GlobalInvocationID.y);
  const int H_in = int(inp.sizes[0].y);
  const int ih = int(gl_GlobalInvocationID.z) % H_in;
  const int n = int(gl_GlobalInvocationID.z) / H_in;

  const int C_out = int(outp.sizes[0].z);
  if (4 * c4 >= C_out || iw >= int(inp.sizes[0].x) ||
      n >= int(inp.sizes[0].w)) {
    return;
  }

  TensorIndex4D inp_tidx;
  inp_tidx.data = ivec4(iw, ih, 16 * c4, n);
  const int inp_base = tensor4d_idx_to_texel_idx(inp, inp_tidx, inp_layout);
  // Word offset between consecutive channel blocks
  const int inp_c4_stride =
      int(inp.strides[0].z) * div_4(get_block_numel(inp_layout));

  ivec4 block = ivec4(0);
  [[unroll]] for (int i = 0; i < 4; ++i) {
    // Output channels past C_out are padding and stay zero.
    if (4 * c4 + i < C_out) {
      block[i] = t_inp[inp_base + i * inp_c4_stride];
      if (passthrough == 0) {
        block[i] = requantize_word(block[i]);
      }
    }
  }
  block = transpose_int8x4_block(block);

  [[unroll]] for (int i = 0; i < 4; ++i) {
    TensorIndex4D outp_tidx;
    outp_tidx.data = ivec4(2 * iw + (i & 1), 2 * ih + (i >> 1), 4 * c4, n);
    t_outp[tensor4d_idx_to_texel_idx(outp, outp_tidx, outp_layout)] = block[i];
  }
}
