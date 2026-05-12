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

// Output buffer: packed int8x4 (each int32 = 4 packed int8 along packed_dim)
${layout_declare_tensor(B, "w", "t_outp", "int", "buffer")}
// Input buffer: packed int8x4 (each int32 = 4 packed int8 along packed_dim)
${layout_declare_tensor(B, "r", "t_inp", "int", "buffer")}

// Metadata for output and input tensors. Both are int8x4 packed buffers but
// may use different block layouts (e.g. PACKED_INT8_4W vs PACKED_INT8_4W4C).
${layout_declare_ubo(B, "BufferMetadata", "outp")}
${layout_declare_ubo(B, "BufferMetadata", "inp")}

layout(push_constant) uniform restrict Block {
  float scale_in;
  float inv_scale_out;
  int zp_in;
  int zp_out;
  int upscale_factor;
  // Whether we can skip the requantize math and do a pure byte shuffle.
  // Set to 1 by the host when (scale_in == scale_out) && (zp_in == zp_out).
  int passthrough;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "inp_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "outp_layout", "CONTIG_LAYOUT_INT")}

/*
 * PixelShuffle map: out[n, c_out, oh, ow] =
 *     in[n, c_out * r * r + (oh % r) * r + (ow % r), oh / r, ow / r].
 *
 * Each thread produces one output int32 word (= 4 consecutive output channels
 * at one (n, oh, ow) spatial position). Channels are the packed dim and
 * packed_dim_block_size is 4, so writing one int word fills 4 channel lanes.
 *
 * The four channel lanes inside an output int come from four DIFFERENT input
 * words (channels spaced by r*r in the input), so each thread issues 4 input
 * loads. The (oh % r, ow % r) -> input lane mapping is fixed for a given
 * thread because all four output lanes share (oh, ow). Out-of-output-bounds
 * channel lanes (when C_out is not a multiple of 4) are zero-filled.
 *
 * Supported layouts: channels-packed family (PACKED_INT8_4W4C, PACKED_INT8_4C1W,
 * PACKED_INT8_CONV2D). Layout-aware byte indexing is handled by
 * tensor4d_idx_to_buf_idx (which consumes inp_layout / outp_layout).
 */

// Apply requantize to an int8 lane value.
int requantize_lane(const int q_in) {
  if (passthrough != 0) {
    return q_in;
  }
  // Requantize: round((q_in - zp_in) * scale_in * inv_scale_out) + zp_out,
  // clamped to int8.
  float dq = float(q_in - zp_in) * scale_in;
  int qv = int(round(dq * inv_scale_out)) + zp_out;
  qv = clamp(qv, -128, 127);
  return qv;
}

void main() {
  // Output sizes (WHCN order via meta.sizes[0])
  const int W_out = int(safe_idx(outp.sizes[0], 0));
  const int H_out = int(safe_idx(outp.sizes[0], 1));
  const int C_out = int(safe_idx(outp.sizes[0], 2));
  const int N = int(safe_idx(outp.sizes[0], 3));

  // Input sizes
  const int W_in = int(safe_idx(inp.sizes[0], 0));
  const int H_in = int(safe_idx(inp.sizes[0], 1));
  const int C_in = int(safe_idx(inp.sizes[0], 2));

  // One thread per output int32 word: word covers 4 consecutive channels
  // (along the packed dim) at one (n, oh, ow) spatial position.
  const int C_words = div_up_4(C_out);
  const int total_words = N * C_words * H_out * W_out;
  const int thread_idx = int(gl_GlobalInvocationID.x);
  if (thread_idx >= total_words) {
    return;
  }

  // Decode thread_idx in (W_out, H_out, C_words, N) order.
  const int ow = thread_idx % W_out;
  const int oh = (thread_idx / W_out) % H_out;
  const int c_word = (thread_idx / (W_out * H_out)) % C_words;
  const int n = thread_idx / (W_out * H_out * C_words);
  const int c_out_base = c_word * 4;

  const int r = upscale_factor;
  // (oh % r, ow % r) determines which input channel lane within the input
  // word group of size r*r — constant for all 4 output channel lanes here.
  const int offset = (oh % r) * r + (ow % r);
  const int ih = oh / r;
  const int iw = ow / r;

  const int c_in_first = c_out_base * r * r + offset;

  // Compute byte_idx for the first lane (i=0) via the layout-aware helper.
  TensorIndex4D inp_idx;
  inp_idx.data = ivec4(iw, ih, c_in_first, n);
  const int byte_idx_first = tensor4d_idx_to_buf_idx(inp, inp_idx, inp_layout);

  // byte_stride between successive c_in advances of r*r = inner_block_size = 4.
  // Each advance bumps the block-space C coord by 1, so byte_idx grows by
  // stride[inner_dim] * block_numel. Both factors are layout-only, no second
  // helper call needed. (Assumes r*r == inner_block_size == 4, enforced by the
  // C++ dispatch's r==2 and packed_dim_block_size==4 asserts.)
  const int byte_stride =
      int(stride_at(inp, get_packed_dim(inp_layout))) * get_block_numel(inp_layout);

  // lane is the byte position within an int32 word, which equals
  // (intra_block_idx % 4) since block_numel is a multiple of 4. And
  // intra_block_idx % 4 == inner_offset == c_in_first % 4 == offset.
  const int lane = offset;

  int packed_out = 0;
  [[unroll]] for (int i = 0; i < 4; ++i) {
    const int c_out_lane = c_out_base + i;
    int q_out = 0;
    if (c_out_lane < C_out) {
      const int c_in = c_in_first + i * r * r;
      int q_in;
      if (iw >= W_in || ih >= H_in || c_in >= C_in) {
        q_in = zp_in;
      } else {
        const int byte_idx = byte_idx_first + i * byte_stride;
        const int word_idx = div_4(byte_idx);
        const int packed = t_inp[word_idx];
        // Sign-extend from 8-bit
        q_in = ((packed >> (lane * 8)) << 24) >> 24;
      }
      q_out = requantize_lane(q_in);
    }
    packed_out |= (q_out & 0xFF) << (i * 8);
  }

  // Store the packed int directly. Output's packed dim is channels with
  // block size 4, so the byte index for c_out_base aligns to a word boundary.
  TensorIndex4D outp_idx;
  outp_idx.data = ivec4(ow, oh, c_out_base, n);
  const int outp_byte_idx = tensor4d_idx_to_buf_idx(outp, outp_idx, outp_layout);
  t_outp[div_4(outp_byte_idx)] = packed_out;
}
