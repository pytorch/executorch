/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Im2col transformation for FP32 / FP16 conv2d.
 *
 * One dispatch materializes a tile of OH_TILE output-height rows (full W_out
 * each) of the im2col matrix, starting at output-height row OH_OFFSET. The full
 * matrix has logical shape [M, K_total] where
 *   M       = H_out * W_out                 (number of output spatial positions)
 *   K_total = Kh * Kw * align_up_4(C_in)    (flattened receptive field)
 *
 * Tiling by output-height rows bounds the scratch tensor to a fixed byte budget
 * regardless of resolution: the scratch holds OH_TILE * W_out rows, not M. A
 * tile-local row m_local decodes to oh_local = m_local / W_out,
 * ow = m_local % W_out; the SOURCE spatial position uses the global output row
 * oh = OH_OFFSET + oh_local. Tiling by H rows (rather than flat M rows) keeps
 * this row->(oh,ow) decode exact for the spatial texture3d layout too. When
 * tiling is disabled the caller passes OH_OFFSET = 0 and OH_TILE = H_out.
 *
 * K layout (so a 4-tile in K — one vec4 — holds the same kernel position):
 *   K = (ki * Kw + kj) * Cin_padded + ci
 *
 * Three codegen'd storage variants of the output tensor:
 *   - texture2d, width-packed: texel at (k4, m_local) holds 4 K values for
 *     tile-local row m_local.  Extents = (K_total/4, OH_TILE * W_out).
 *   - texture3d, channels-packed: texel at (ow, oh_local, k4) holds 4 K values
 *     for output spatial position (OH_OFFSET + oh_local, ow).  Extents =
 *     (W_out, OH_TILE, K4).  Used as a fallback when M would exceed
 *     max_texture2d_dim.
 *   - buffer: vec4 at offset (m_local * K4 + k4), same K packing.
 *
 * The caller picks storage per device (Mali → buffer; others → texture2d
 * when its 2D extents fit, texture3d when its 3D extents fit, else buffer).
 */

#version 450 core

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_load_type(DTYPE, "texture3d")}

$if OUT_STORAGE == "buffer":
  #define OUTPUT_BUFFER
  #define VEC4_BUF_T ${texel_load_type(DTYPE, "buffer")}
$elif OUT_STORAGE == "texture3d":
  #define OUTPUT_TEXTURE3D

${define_required_extensions("texture3d", DTYPE)}
$if OUT_STORAGE == "buffer":
  ${define_required_extensions("buffer", DTYPE)}

layout(std430) buffer;

$if OUT_STORAGE == "buffer":
  ${layout_declare_tensor(B, "w", "t_out", DTYPE, "buffer", is_scalar_array=False)}
$else:
  ${layout_declare_tensor(B, "w", "t_out", DTYPE, OUT_STORAGE)}
${layout_declare_tensor(B, "r", "t_in", DTYPE, "texture3d")}

${layout_declare_ubo(B, "ivec4", "in_sizes")}

// Push constants are uploaded in 16-byte chunks (one ivec4 each) to comply
// with the per-entry size limit. All of these fields are shape-independent
// (they depend only on the conv kernel params and C_in), so they are safe to
// bake at build time even under dynamic shapes — W_out / H_out / M are derived
// at runtime from the refreshed in_sizes UBO below.
layout(push_constant) uniform restrict Block {
  ivec4 kernel_stride;  // (Kh, Kw, Sh, Sw)
  ivec4 padding_dil;    // (Ph, Pw, Dh, Dw)
  ivec4 dims;           // (Cin_padded, OH_OFFSET, OH_TILE, K4_total)
};

#define KERNEL_H   kernel_stride.x
#define KERNEL_W   kernel_stride.y
#define STRIDE_H   kernel_stride.z
#define STRIDE_W   kernel_stride.w
#define PADDING_H  padding_dil.x
#define PADDING_W  padding_dil.y
#define DILATION_H padding_dil.z
#define DILATION_W padding_dil.w
#define CIN_PADDED dims.x
#define OH_OFFSET  dims.y
#define OH_TILE    dims.z
#define K4_TOTAL   dims.w

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const int k4 = int(gl_GlobalInvocationID.x);
  // gl_GlobalInvocationID.y is the tile-local row m_local within this tile's
  // OH_TILE * W_out rows; it maps to the global output row via OH_OFFSET.
  const int m_local = int(gl_GlobalInvocationID.y);

  // Derive the spatial output extents from the (refreshed-on-resize) input
  // sizes UBO so the im2col mapping tracks dynamic input shapes. in_sizes is
  // (W_in, H_in, C_in, N). dilation == 1 is guaranteed by the C++ routing
  // heuristic, but the general formula is used for correctness.
  const int W_OUT =
      (in_sizes.x + 2 * PADDING_W - DILATION_W * (KERNEL_W - 1) - 1) / STRIDE_W +
      1;
  const int H_OUT =
      (in_sizes.y + 2 * PADDING_H - DILATION_H * (KERNEL_H - 1) - 1) / STRIDE_H +
      1;
  // Rows materialized by this tile (capped to the scratch extent).
  const int M_TILE = OH_TILE * W_OUT;

  if (k4 >= K4_TOTAL || m_local >= M_TILE) {
    return;
  }

  // Tile-local row m_local -> (oh_local, ow); global output row oh = OH_OFFSET +
  // oh_local. Rows past the real H_OUT (in a partial trailing tile, or when a
  // dynamic shape shrinks H_OUT below the build-time OH_OFFSET) write zeros.
  const int oh_local = m_local / W_OUT;
  const int ow       = m_local % W_OUT;
  const int oh       = OH_OFFSET + oh_local;

  const int k_start = k4 * 4;

  // K = (ki * Kw + kj) * Cin_padded + ci ; since Cin_padded % 4 == 0, all 4
  // K values in this texel share the same (ki, kj) and span 4 consecutive
  // ci values starting at ci_start.
  const int krow_idx = k_start / CIN_PADDED; // ki * Kw + kj
  const int ci_start = k_start % CIN_PADDED;
  const int kj       = krow_idx % KERNEL_W;
  const int ki       = krow_idx / KERNEL_W;
  const int ci_blk   = ci_start >> 2;        // ci_start / 4

  // Compute the input spatial position for this (oh, ow, ki, kj).
  const int ih = oh * STRIDE_H - PADDING_H + ki * DILATION_H;
  const int iw = ow * STRIDE_W - PADDING_W + kj * DILATION_W;

  VEC4_T out_texel = VEC4_T(0);
  if (oh < H_OUT && ih >= 0 && ih < in_sizes.y && iw >= 0 && iw < in_sizes.x) {
    out_texel = texelFetch(t_in, ivec3(iw, ih, ci_blk), 0);
  }

#if defined(OUTPUT_BUFFER)
  t_out[m_local * K4_TOTAL + k4] = VEC4_BUF_T(out_texel);
#elif defined(OUTPUT_TEXTURE3D)
  imageStore(t_out, ivec3(ow, oh_local, k4), out_texel);
#else
  imageStore(t_out, ivec2(k4, m_local), out_texel);
#endif
}
