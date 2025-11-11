/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}
#define VEC4_T ${texel_load_type(DTYPE, IO_STORAGE)}
#define T ${texel_load_component_type(DTYPE, IO_STORAGE)}

$if IO_STORAGE == "buffer":
  #define OUTPUT_BUFFER
  #define INPUT_BUFFER
$if V_CACHE_STORAGE == "buffer":
  #define V_CACHE_BUFFER

#define TILE_M4 ${TILE_M4}
// Equvalent to K4 in matrix multiplication
#define TILE_K4 ${TILE_K4}
// Equvalent to N4 in matrix multiplication
#define TILE_N4 ${TILE_N4}

#define TILE_M ${TILE_M4 * 4}
#define TILE_K ${TILE_K4 * 4}
#define TILE_N ${TILE_N4 * 4}

${define_required_extensions(DTYPE)}

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_output", DTYPE, IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_attn_weights", DTYPE, IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_v_cache", DTYPE, V_CACHE_STORAGE, is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "q_projected_sizes")}
${layout_declare_ubo(B, "ivec4", "v_cache_sizes")}
${layout_declare_ubo(B, "int", "input_pos")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#include "sdpa_fp_attn_weight_tile_load.glslh"
#include "sdpa_fp_v_cache_tile_load.glslh"
#include "linear_fp_output_tile_fp_compute.glslh"
#include "sdpa_fp_out_tile_store.glslh"

/*
 * Compute SDPA output given the attention weights and v_cache tensors.
 * attention weights has shape (batches, num_q_heads, seq_len, context_len)
 * v_cache has shape (batches, max_context_len, num_kv_heads, head_dim)
 * output has shape (batches, seq_len, num_q_heads, head_dim)
 */

void main() {
  const int tile_idx_x = int(gl_GlobalInvocationID.x);
  const int tile_idx_y = int(gl_GlobalInvocationID.y);
  // idx along output num_q_heads dim
  const int q_h = int(gl_GlobalInvocationID.z);

  // idx along the output head_dim dim
  const int d = tile_idx_x * TILE_N;
  const int d4 = div_4(d);

  // idx along the output seq_len dim
  const int s = tile_idx_y * TILE_M;

  // texel size of head_dim
  const int D4 = div_up_4(q_projected_sizes.x);
  // number of Q heads
  const int Q_H = q_projected_sizes.y;
  // sequence length
  const int S = q_projected_sizes.z;
  const int S_aligned = align_up_4(S);

  // number of K/V heads
  const int KV_H = v_cache_sizes.y;
  // Max context length
  const int C = v_cache_sizes.z;
  const int C4 = div_up_4(C);

  int kv_h = q_h;
  if (KV_H < Q_H) {
    kv_h = q_h / (Q_H / KV_H);
  }

  // current context length
  const int context_len = input_pos + S;
  const int context_texel_len = div_up_4(context_len);

  // bounds check
  if (d4 >= D4 || s >= S || q_h >= Q_H) {
    return;
  }

  FPOutTile out_tile;
  initialize(out_tile);

  FPInputTile attn_weight_tile;
  FPWeightTile w_tile;

  const int context_len_aligned_down = context_len - mod_4(context_len);
  const int C4_limit = div_4(context_len_aligned_down);

  for (int c4 = 0; c4 < C4_limit; c4++) {
    const int c = mul_4(c4);
    load_attn_weight_tile_no_checks(
      attn_weight_tile,
      c4,
      s,
      q_h,
      context_texel_len,
      S_aligned,
      Q_H);

    load_v_cache_tile_no_checks(
      w_tile,
      d4,
      c,
      kv_h,
      D4,
      context_len,
      C,
      KV_H);

    fp_accumulate_with_fp_weight(out_tile, attn_weight_tile, w_tile);
  }
  for (int c4 = C4_limit; c4 < context_texel_len; c4++) {
    const int c = mul_4(c4);
    load_attn_weight_tile_with_checks(
      attn_weight_tile,
      c4,
      s,
      q_h,
      context_texel_len,
      S_aligned,
      Q_H);

    load_v_cache_tile_with_checks(
      w_tile,
      d4,
      c,
      kv_h,
      D4,
      context_len,
      C,
      KV_H);

    fp_accumulate_with_fp_weight(out_tile, attn_weight_tile, w_tile);
  }

  store_sdpa_out_tile_with_checks(
    out_tile,
    d4,
    s,
    q_h,
    D4,
    S,
    Q_H);
}
