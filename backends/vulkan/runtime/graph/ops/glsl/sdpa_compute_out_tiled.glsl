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

$if MODE == "llm":
  #define HAS_INPUT_POS
  #define HAS_GQA
  #define V_LAYOUT DHSB
  #define OUT_LAYOUT DHSB
$else:
  #define V_LAYOUT DSHB
  #define OUT_LAYOUT DSHB

#define TILE_M4 ${TILE_M4}
// Equvalent to K4 in matrix multiplication
#define TILE_K4 ${TILE_K4}
// Equvalent to N4 in matrix multiplication
#define TILE_N4 ${TILE_N4}

#define TILE_M ${TILE_M4 * 4}
#define TILE_K ${TILE_K4 * 4}
#define TILE_N ${TILE_N4 * 4}

${define_required_extensions(IO_STORAGE, DTYPE)}

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_output", DTYPE, IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_attn_weights", DTYPE, IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_v", DTYPE, V_CACHE_STORAGE, is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "q_sizes")}
${layout_declare_ubo(B, "ivec4", "v_sizes")}
$if MODE == "llm":
  ${layout_declare_ubo(B, "int", "input_pos")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

#include "sdpa_fp_attn_weight_tile_load.glslh"
#include "sdpa_fp_v_cache_tile_load.glslh"
#include "linear_fp_output_tile_fp_compute.glslh"
#include "sdpa_fp_out_tile_store.glslh"

/*
 * Compute SDPA output given the attention weights and V tensors.
 *
 * LLM SDPA (HAS_INPUT_POS, HAS_GQA):
 *   attn_weights: [B, H_q, S, context_len]
 *   v (v_cache):  [B, C_max, H_kv, D]    (DHSB layout)
 *   output:       [B, S, H_q, D]         (DHSB layout)
 *   current context_len = input_pos + S
 *   GQA: Q heads may be > KV heads; kv_h = q_h / (H_q / H_kv)
 *
 * Fused SDPA:
 *   attn_weights: [B, H, S, context_len]
 *   v:            [B, H, context_len, D] (DSHB layout)
 *   output:       [B, H, S, D]           (DSHB layout)
 *
 * Dispatch: (D_tiles, S_tiles, H * B) — for LLM (batch=1), H * B == H_q.
 */

void main() {
  const int tile_idx_x = int(gl_GlobalInvocationID.x);
  const int tile_idx_y = int(gl_GlobalInvocationID.y);
  // For LLM: q_head index. For fused: combined batch*H + head index.
  const int q_h = int(gl_GlobalInvocationID.z);

  // idx along the output head_dim dim
  const int d = tile_idx_x * TILE_N;
  const int d4 = div_4(d);

  // idx along the output seq_len dim
  const int s = tile_idx_y * TILE_M;

#ifdef HAS_INPUT_POS
  // LLM: q_sizes is WHCN {D, H_q, S, B}
  const int D = q_sizes.x;
  const int Q_H = q_sizes.y;
  const int S = q_sizes.z;
  // v_sizes is WHCN {D, H_kv, C_max, B}
  const int KV_H = v_sizes.y;
  const int C = v_sizes.z;
#else
  // Fused: q_sizes is WHCN {D, S, H, B}
  const int D = q_sizes.x;
  const int S = q_sizes.y;
  const int Q_H = q_sizes.z;
  // v_sizes is WHCN {D, context_len, H, B}
  const int KV_H = v_sizes.z;
  const int C = v_sizes.y;
#endif
  const int D4 = div_up_4(D);
  const int S_aligned = align_up_4(S);

#ifdef HAS_INPUT_POS
  // current context length for LLM decode/prefill
  const int context_len = input_pos + S;
#else
  // fused: full key sequence length from v_sizes (DSHB: {D, L, H, B})
  const int context_len = v_sizes.y;
#endif
  const int context_texel_len = div_up_4(context_len);

#ifdef HAS_GQA
  int kv_h = q_h;
  if (KV_H < Q_H) {
    kv_h = q_h / (Q_H / KV_H);
  }
#else
  const int kv_h = q_h;
#endif

  // bounds check — q_h bound is Q_H * batch_size; for LLM (batch=1) this
  // equals Q_H, for fused this equals H * B.
  if (d4 >= D4 || s >= S || q_h >= Q_H * q_sizes.w) {
    return;
  }

  FPOutTile out_tile;
  initialize(out_tile);

  FPInputTile attn_weight_tile;
  FPWeightTile w_tile;

  // For LLM, the attn_weights tensor has seq_len padded up to a multiple of 4
  // (S_aligned). The loader accesses (head * attn_S * C4 + s * C4 + c4), so
  // pass S_aligned in LLM mode and S in fused mode.
#ifdef HAS_INPUT_POS
  const int attn_S = S_aligned;
#else
  const int attn_S = S;
#endif

  // Split loop into aligned + tail for efficiency
  const int context_len_aligned_down = context_len - mod_4(context_len);
  const int C4_limit = div_4(context_len_aligned_down);

  for (int c4 = 0; c4 < C4_limit; c4++) {
    const int c = mul_4(c4);
    load_attn_weight_tile_no_checks(
        attn_weight_tile, c4, s, q_h, context_texel_len, attn_S, Q_H);
    load_v_cache_tile_no_checks(w_tile, d4, c, kv_h, D4, context_len, C, KV_H);
    fp_accumulate_with_fp_weight(out_tile, attn_weight_tile, w_tile);
  }
  for (int c4 = C4_limit; c4 < context_texel_len; c4++) {
    const int c = mul_4(c4);
    load_attn_weight_tile_with_checks(
        attn_weight_tile, c4, s, q_h, context_texel_len, attn_S, Q_H);
    load_v_cache_tile_with_checks(w_tile, d4, c, kv_h, D4, context_len, C, KV_H);
    fp_accumulate_with_fp_weight(out_tile, attn_weight_tile, w_tile);
  }

  store_sdpa_out_tile_with_checks(out_tile, d4, s, q_h, D4, S, Q_H);
}
