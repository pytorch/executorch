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

${layout_declare_ubo(B, "int", "start_pos")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "xqout_layout", "CONTIG_LAYOUT_INT")}
${layout_declare_spec_const(C, "int", "freqs_layout", "CONTIG_LAYOUT_INT")}
// 0 = full rotation (rotary_dim == head_dim), 1 = partial rotation with
// passthrough region. Resolved at pipeline creation time, so the driver
// eliminates the dead branch entirely.
${layout_declare_spec_const(C, "int", "partial_rotary", "0")}

// Load/store helpers that abstract buffer vs texture access. The `layout`
// parameter is only used in the texture path; the buffer path ignores it.
#ifdef USING_BUFFER
#define LOAD(tensor, meta, tidx, layout) \
  tensor[div_4(tensor4d_idx_to_linear_idx(meta, tidx))]
#define STORE(tensor, meta, tidx, layout, val) \
  tensor[div_4(tensor4d_idx_to_linear_idx(meta, tidx))] = val
#else
#define LOAD(tensor, meta, tidx, layout) \
  texelFetch(tensor, tensor4d_idx_to_texel_pos_simple(meta, tidx, layout), 0)
#define STORE(tensor, meta, tidx, layout, val) \
  imageStore(tensor, tensor4d_idx_to_texel_pos_simple(meta, tidx, layout), val)
#endif

/*
 * HuggingFace-style rotary positional embeddings.
 *
 * Input tensors:
 *   xq          (batch, seq_len, n_heads, head_dim)
 *   xk          (batch, seq_len, n_kv_heads, head_dim)
 *   freqs_cos   (max_seq_len, rotary_dim)   rotary_dim <= head_dim
 *   freqs_sin   (max_seq_len, rotary_dim)
 *   start_pos   (int) offset into freqs table
 *
 * For i in [0, rotary_half):
 *   out[i]             = x[i]*cos[i]             - x[i+rotary_half]*sin[i]
 *   out[i+rotary_half] = x[i+rotary_half]*cos[i] + x[i]*sin[i]
 * When partial_rotary == 1, for i in [rotary_dim, head_dim):
 *   out[i] = x[i]   (passthrough)
 *
 * Each thread handles one texel (4 elements) along head_dim.
 * All input tensors must be width-packed.
 */
void main() {
#ifdef USING_BUFFER
  const int rotary_half = int(width(freqs_cos)) / 2;
#else
  const int rotary_half = freqs_cos.sizes.x / 2;
#endif

  const int x = int(gl_GlobalInvocationID.x) * 4;

  TensorIndex4D tidx = zero_tensor4d_idx();
  tidx.data.x = x;
  tidx.data.yz = ivec2(gl_GlobalInvocationID.yz);

  if (out_of_bounds(tidx, xqout)) {
    return;
  }

  const bool process_k = !out_of_bounds(tidx, xkout);

  // Passthrough region (only reachable when partial_rotary == 1).
  if (partial_rotary == 1 && x >= rotary_half * 2) {
    STORE(t_xqout, xqout, tidx, xqout_layout, LOAD(t_xq, xqout, tidx, xqout_layout));
    if (process_k) {
      STORE(t_xkout, xkout, tidx, xqout_layout, LOAD(t_xk, xkout, tidx, xqout_layout));
    }
    return;
  }

  // Rotation region: determine pair and freqs indices.
  const bool is_second_half = (x >= rotary_half);

  TensorIndex4D pair_tidx = tidx;
  pair_tidx.data.x = is_second_half ? (x - rotary_half) : (x + rotary_half);

  TensorIndex4D freqs_tidx = zero_tensor4d_idx();
  freqs_tidx.data.x = is_second_half ? (x - rotary_half) : x;
  freqs_tidx.data.y = tidx.data.z + start_pos;

  const VEC4_T cos_val = LOAD(t_freqs_cos, freqs_cos, freqs_tidx, freqs_layout);
  const VEC4_T sin_val = LOAD(t_freqs_sin, freqs_cos, freqs_tidx, freqs_layout);

  // First half:  out = x*cos - pair*sin
  // Second half: out = x*cos + pair*sin
  const VEC4_T xq_val = LOAD(t_xq, xqout, tidx, xqout_layout);
  const VEC4_T xq_pair = LOAD(t_xq, xqout, pair_tidx, xqout_layout);
  const VEC4_T sign = VEC4_T(is_second_half ? 1.0 : -1.0);

  STORE(t_xqout, xqout, tidx, xqout_layout, xq_val * cos_val + sign * xq_pair * sin_val);

  if (process_k) {
    const VEC4_T xk_val = LOAD(t_xk, xkout, tidx, xqout_layout);
    const VEC4_T xk_pair = LOAD(t_xk, xkout, pair_tidx, xqout_layout);
    STORE(t_xkout, xkout, tidx, xqout_layout, xk_val * cos_val + sign * xk_pair * sin_val);
  }
}
