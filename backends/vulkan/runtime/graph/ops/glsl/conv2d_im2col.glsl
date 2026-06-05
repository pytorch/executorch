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
 * The output is a 2D matrix of shape [M, K_total] where
 *   M       = H_out * W_out                 (number of output spatial positions)
 *   K_total = Kh * Kw * align_up_4(C_in)    (flattened receptive field)
 *
 * K layout (so a 4-tile in K — one vec4 — holds the same kernel position):
 *   K = (ki * Kw + kj) * Cin_padded + ci
 *
 * Three codegen'd storage variants of the output tensor:
 *   - texture2d, width-packed: texel at (k4, m) holds 4 K values for spatial
 *     position m.  Extents = (K_total/4, M).
 *   - texture3d, channels-packed: texel at (ow, oh, k4) holds 4 K values
 *     for output spatial position (oh, ow).  Extents = (W_out, H_out, K4).
 *     Used as a fallback when M would exceed max_texture2d_dim.
 *   - buffer: vec4 at offset (m * K4 + k4), same K packing.
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
// with the per-entry size limit.
layout(push_constant) uniform restrict Block {
  ivec4 kernel_stride;  // (Kh, Kw, Sh, Sw)
  ivec4 padding_dil;    // (Ph, Pw, Dh, Dw)
  ivec4 dims;           // (Cin_padded, W_out, H_out, K4_total)
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
#define W_OUT      dims.y
#define H_OUT      dims.z
#define K4_TOTAL   dims.w

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

void main() {
  const int k4 = int(gl_GlobalInvocationID.x);
  const int m  = int(gl_GlobalInvocationID.y);
  const int M  = H_OUT * W_OUT;

  if (k4 >= K4_TOTAL || m >= M) {
    return;
  }

  const int k_start = k4 * 4;

  // K = (ki * Kw + kj) * Cin_padded + ci ; since Cin_padded % 4 == 0, all 4
  // K values in this texel share the same (ki, kj) and span 4 consecutive
  // ci values starting at ci_start.
  const int krow_idx = k_start / CIN_PADDED; // ki * Kw + kj
  const int ci_start = k_start % CIN_PADDED;
  const int kj       = krow_idx % KERNEL_W;
  const int ki       = krow_idx / KERNEL_W;
  const int ci_blk   = ci_start >> 2;        // ci_start / 4

  // Decompose flat output position m back into (oh, ow).
  const int ow = m % W_OUT;
  const int oh = m / W_OUT;

  // Compute the input spatial position for this (oh, ow, ki, kj).
  const int ih = oh * STRIDE_H - PADDING_H + ki * DILATION_H;
  const int iw = ow * STRIDE_W - PADDING_W + kj * DILATION_W;

  VEC4_T out_texel = VEC4_T(0);
  if (ih >= 0 && ih < in_sizes.y && iw >= 0 && iw < in_sizes.x) {
    out_texel = texelFetch(t_in, ivec3(iw, ih, ci_blk), 0);
  }

#if defined(OUTPUT_BUFFER)
  t_out[m * K4_TOTAL + k4] = VEC4_BUF_T(out_texel);
#elif defined(OUTPUT_TEXTURE3D)
  imageStore(t_out, ivec3(ow, oh, k4), out_texel);
#else
  imageStore(t_out, ivec2(k4, m), out_texel);
#endif
}
