/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define VEC4_T ${texel_load_type(DTYPE, STORAGE)}

${define_required_extensions(DTYPE)}

layout(std430) buffer;

${layout_declare_tensor(B, "w", "xqout", DTYPE, STORAGE)}
${layout_declare_tensor(B, "w", "xkout", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "xq", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "xk", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "freqs_cos", DTYPE, STORAGE)}
${layout_declare_tensor(B, "r", "freqs_sin", DTYPE, STORAGE)}
${layout_declare_ubo(B, "ivec3", "xqout_limits")}
${layout_declare_ubo(B, "ivec3", "xkout_limits")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int packed_dim = 0;

#include "indexing_utils.h"

/*
 * This shader computes rotary positional embeddings which are used in the Llama
 * model architecture. There are 4 input tensors with the following shapes.
 * Note that head_dim = embedding_dim / num_heads
 *
 * 1. xq (batch_size, sequence_len, num_heads, head_dim)
 * 2. xk (batch_size, sequence_len, num_kv_heads, head_dim)
 * 3. freqs_cos (sequence_len, head_dim / 2)
 * 4. freqs_cos (sequence_len, head_dim / 2)
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
  const ivec3 x_pos_1 = ivec3(
      gl_GlobalInvocationID.x * 2, gl_GlobalInvocationID.yz);
  const ivec3 x_pos_2 = ivec3(x_pos_1.x + 1, x_pos_1.yz);

  if (any(greaterThanEqual(x_pos_2, xqout_limits))) {
    return;
  }

  const ivec3 freqs_pos = ivec3(gl_GlobalInvocationID.xz, 0);

  VEC4_T cos_tex = load_texel(freqs_cos, freqs_pos);
  VEC4_T sin_tex = load_texel(freqs_sin, freqs_pos);

  // Compute xqout

  VEC4_T x_tex_1 = load_texel(xq, x_pos_1);
  VEC4_T x_tex_2 = load_texel(xq, x_pos_2);

  // Separate into even and odd elements
  VEC4_T x_r = VEC4_T(x_tex_1.xz, x_tex_2.xz);
  VEC4_T x_i = VEC4_T(x_tex_1.yw, x_tex_2.yw);

  VEC4_T xout_r = x_r * cos_tex - x_i * sin_tex;
  VEC4_T xout_i = x_r * sin_tex + x_i * cos_tex;

  VEC4_T xout_tex_1 = VEC4_T(xout_r.x, xout_i.x, xout_r.y, xout_i.y);
  VEC4_T xout_tex_2 = VEC4_T(xout_r.z, xout_i.z, xout_r.w, xout_i.w);

  write_texel(xqout, x_pos_1, xout_tex_1);
  write_texel(xqout, x_pos_2, xout_tex_2);

  // n_heads will be greater than or equal to n_kv_heads, therefore xq and xqout
  // may have a larger height dim than xk and xkout. Only compute xkout if this
  // invocation is still within bounds.
  if (any(greaterThanEqual(x_pos_2, xkout_limits))) {
    return;
  }

  // Compute xkout

  x_tex_1 = load_texel(xk, x_pos_1);
  x_tex_2 = load_texel(xk, x_pos_2);

  x_r = VEC4_T(x_tex_1.xz, x_tex_2.xz);
  x_i = VEC4_T(x_tex_1.yw, x_tex_2.yw);

  xout_r = x_r * cos_tex - x_i * sin_tex;
  xout_i = x_r * sin_tex + x_i * cos_tex;

  xout_tex_1 = VEC4_T(xout_r.x, xout_i.x, xout_r.y, xout_i.y);
  xout_tex_2 = VEC4_T(xout_r.z, xout_i.z, xout_r.w, xout_i.w);

  write_texel(xkout, x_pos_1, xout_tex_1);
  write_texel(xkout, x_pos_2, xout_tex_2);
}
