/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

${define_required_extensions(STORAGE, DTYPE)}

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}

${define_active_storage_type(STORAGE)}

layout(std430) buffer;

#include "indexing.glslh"

${layout_declare_tensor(B, "w", "t_xqout", DTYPE, STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "w", "t_xkout", DTYPE, STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_xq", DTYPE, STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_xk", DTYPE, STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_freqs_cos", DTYPE, STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_freqs_sin", DTYPE, STORAGE, is_scalar_array=False)}

$if STORAGE == "buffer":
  ${layout_declare_ubo(B, "BufferMetadata", "xqout")}
  ${layout_declare_ubo(B, "BufferMetadata", "xkout")}
  ${layout_declare_ubo(B, "BufferMetadata", "freqs_cos")}
$else:
  ${layout_declare_ubo(B, "TextureMetadata", "xqout")}
  ${layout_declare_ubo(B, "TextureMetadata", "xkout")}
  ${layout_declare_ubo(B, "TextureMetadata", "freqs_cos")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

/*
 * This shader computes rotary positional embeddings which are used in the Llama
 * model architecture. There are 4 input tensors with the following shapes.
 * Note that head_dim = embedding_dim / num_heads
 *
 * 1. xq (batch_size, sequence_len, num_heads, head_dim)
 * 2. xk (batch_size, sequence_len, num_kv_heads, head_dim)
 * 3. freqs_cos (sequence_len, head_dim / 2)
 * 4. freqs_sin (sequence_len, head_dim / 2)
 *
 * Two output tensors are produced, with the same shapes as xq and xk
 * respectively.
 *
 * The computation of rotary positional embeddings can be summarized with the
 * following equations:
 *
 * xq_out[2i] = xq[2i] * freqs_cos[i] - xq[2i + 1] * freqs_sin[i]
 * xq_out[2i + 1] = xq[2i] * freqs_sin[i] + xq[2i + 1] * freqs_cos[i]
 *
 * Essentially, taking each row along head_dim of the xq and xk tensors, each
 * row is split into even and odd elements (xq[2i] and xq[2i + 1] respectively).
 * The even components of the output multiply the even components of the inputs
 * with the freqs_cos tensor, and the odd components of the inputs with the
 * freqs_sin tensor. The odd components of the output swap this. Throughout the
 * implementation the even components have the _r suffix and the odd components
 * have the _i suffix; this is a reference to complex numbers which can be used
 * to represent rotations.
 *
 * Note that this implementation assumes that all input tensors have the width
 * dim as the packed dim.
 */
void main() {
  // Each thread will write to two output locations to maximize data re-use.
  // One texel loaded from the freqs_cos/freqs_sin tensors can be used to
  // calculate two output texels.
  TensorIndex4D out_tidx_1 = zero_tensor4d_idx();
  out_tidx_1.data.x = int(gl_GlobalInvocationID.x) * 8;
  out_tidx_1.data.yz = ivec2(gl_GlobalInvocationID.yz);

  TensorIndex4D out_tidx_2 = out_tidx_1;
  out_tidx_2.data.x += 4;

  if (out_of_bounds(out_tidx_2, xqout)) {
    return;
  }

  TensorIndex4D freqs_tidx = zero_tensor4d_idx();
  freqs_tidx.data.x = int(gl_GlobalInvocationID.x) * 4;
  freqs_tidx.data.y = out_tidx_1.data.z;

#ifdef USING_BUFFER
  const uint freqs_texel_bufi = div_4(tensor4d_idx_to_linear_idx(freqs_cos, freqs_tidx));
  VEC4_T cos_tex = t_freqs_cos[freqs_texel_bufi];
  VEC4_T sin_tex = t_freqs_sin[freqs_texel_bufi];

  uint x_texel_bufi_1 = div_4(tensor4d_idx_to_linear_idx(xqout, out_tidx_1));
  uint x_texel_bufi_2 = div_4(tensor4d_idx_to_linear_idx(xqout, out_tidx_2));
  VEC4_T x_tex_1 = t_xq[x_texel_bufi_1];
  VEC4_T x_tex_2 = t_xq[x_texel_bufi_2];

#else // USING_TEXTURE
  const ivec3 freqs_pos = tensor4d_idx_to_texel_pos_simple(freqs_cos, freqs_tidx);
  VEC4_T cos_tex = texelFetch(t_freqs_cos, freqs_pos, 0);
  VEC4_T sin_tex = texelFetch(t_freqs_sin, freqs_pos, 0);

  const ivec3 x_pos_1 = tensor4d_idx_to_texel_pos_simple(xqout, out_tidx_1);
  const ivec3 x_pos_2 = tensor4d_idx_to_texel_pos_simple(xqout, out_tidx_2);
  VEC4_T x_tex_1 = texelFetch(t_xq, x_pos_1, 0);
  VEC4_T x_tex_2 = texelFetch(t_xq, x_pos_2, 0);
#endif

  // Compute xqout

  // Separate into even and odd elements
  VEC4_T x_r = VEC4_T(x_tex_1.xz, x_tex_2.xz);
  VEC4_T x_i = VEC4_T(x_tex_1.yw, x_tex_2.yw);

  VEC4_T xout_r = x_r * cos_tex - x_i * sin_tex;
  VEC4_T xout_i = x_r * sin_tex + x_i * cos_tex;

  VEC4_T xout_tex_1 = VEC4_T(xout_r.x, xout_i.x, xout_r.y, xout_i.y);
  VEC4_T xout_tex_2 = VEC4_T(xout_r.z, xout_i.z, xout_r.w, xout_i.w);

#ifdef USING_BUFFER
  t_xqout[x_texel_bufi_1] = xout_tex_1;
  t_xqout[x_texel_bufi_2] = xout_tex_2;
#else // USING_TEXTURE
  imageStore(t_xqout, x_pos_1, xout_tex_1);
  imageStore(t_xqout, x_pos_2, xout_tex_2);
#endif

  // n_heads will be greater than or equal to n_kv_heads, therefore xq and xqout
  // may have a larger height dim than xk and xkout. Only compute xkout if this
  // invocation is still within bounds.
  if (out_of_bounds(out_tidx_2, xkout)) {
    return;
  }

  // Compute xkout

#ifdef USING_BUFFER
  x_texel_bufi_1 = div_4(tensor4d_idx_to_linear_idx(xkout, out_tidx_1));
  x_texel_bufi_2 = div_4(tensor4d_idx_to_linear_idx(xkout, out_tidx_2));

  x_tex_1 = t_xk[x_texel_bufi_1];
  x_tex_2 = t_xk[x_texel_bufi_2];

#else // USING_TEXTURE
  x_tex_1 = texelFetch(t_xk, x_pos_1, 0);
  x_tex_2 = texelFetch(t_xk, x_pos_2, 0);
#endif

  x_r = VEC4_T(x_tex_1.xz, x_tex_2.xz);
  x_i = VEC4_T(x_tex_1.yw, x_tex_2.yw);

  xout_r = x_r * cos_tex - x_i * sin_tex;
  xout_i = x_r * sin_tex + x_i * cos_tex;

  xout_tex_1 = VEC4_T(xout_r.x, xout_i.x, xout_r.y, xout_i.y);
  xout_tex_2 = VEC4_T(xout_r.z, xout_i.z, xout_r.w, xout_i.w);

#ifdef USING_BUFFER
  t_xkout[x_texel_bufi_1] = xout_tex_1;
  t_xkout[x_texel_bufi_2] = xout_tex_2;
#else // USING_TEXTURE
  imageStore(t_xkout, x_pos_1, xout_tex_1);
  imageStore(t_xkout, x_pos_2, xout_tex_2);
#endif
}
