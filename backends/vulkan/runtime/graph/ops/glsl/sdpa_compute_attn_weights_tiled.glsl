/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define IN_DTYPE ${IN_DTYPE}
#define OUT_DTYPE ${OUT_DTYPE}

#define VEC4_T ${texel_load_type(IN_DTYPE, IO_STORAGE)}
#define T ${texel_load_component_type(IN_DTYPE, IO_STORAGE)}

#define LINEAR_FP_OUTPUT_TILE_VEC4_T ${texel_load_type(OUT_DTYPE, IO_STORAGE)}

$if IO_STORAGE == "buffer":
  #define OUTPUT_BUFFER
  #define INPUT_BUFFER
$if K_CACHE_STORAGE == "buffer":
  #define K_CACHE_BUFFER

$if MODE == "llm":
  #define HAS_INPUT_POS
  #define HAS_GQA
  #define Q_LAYOUT DHSB
  #define K_LAYOUT DHSB
$else:
  #define SDPA_PAD_D
  #define Q_LAYOUT DSHB
  #define K_LAYOUT DSHB

$if HAS_BIAS:
  #define HAS_BIAS

#define TILE_M4 ${TILE_M4}
#define TILE_K4 ${TILE_K4}
#define TILE_N4 ${TILE_N4}

#define TILE_M ${TILE_M4 * 4}
#define TILE_K ${TILE_K4 * 4}
#define TILE_N ${TILE_N4 * 4}

${define_required_extensions(IO_STORAGE, [IN_DTYPE, OUT_DTYPE])}

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_attn_weights", OUT_DTYPE, IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_q", IN_DTYPE, IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_k", IN_DTYPE, K_CACHE_STORAGE, is_scalar_array=False)}
$if HAS_BIAS:
  ${layout_declare_tensor(B, "r", "t_bias", IN_DTYPE, IO_STORAGE, is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "q_sizes")}
${layout_declare_ubo(B, "ivec4", "k_sizes")}
$if MODE == "llm":
  ${layout_declare_ubo(B, "int", "input_pos")}
$if HAS_BIAS:
  ${layout_declare_ubo(B, "ivec4", "bias_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "float", "inv_scale", "1.0")}

#include "sdpa_fp_q_projected_tile_load.glslh"
#include "sdpa_fp_k_cache_tile_load.glslh"
#include "linear_fp_output_tile_fp_compute.glslh"
#include "sdpa_fp_attn_weight_tile_store.glslh"

/*
 * Compute attention weights (Q @ K^T) given the Q and K tensors.
 *
 * LLM SDPA (HAS_INPUT_POS, HAS_GQA):
 *   q:            [B, S, H_q, D]        (DHSB layout)
 *   k (k_cache):  [B, C_max, H_kv, D]   (DHSB layout)
 *   attn_weights: [B, H_q, S, context_len] in input dtype
 *   current context_len = input_pos + S
 *   Applies combined scale + causal mask.
 *
 * Fused SDPA:
 *   q:            [B, H, S, D]          (DSHB layout)
 *   k:            [B, H, L, D]          (DSHB layout)
 *   attn_weights: [B, H, S, L] in fp32 to prevent fp16 overflow in Q@K^T
 *   Applies scalar scale, optionally adds bias.
 *
 * Dispatch: (context_tiles, S_tiles, H * B) — for LLM (batch=1), H * B == H_q.
 */

void main() {
  const int tile_idx_x = int(gl_GlobalInvocationID.x);
  const int tile_idx_y = int(gl_GlobalInvocationID.y);
  // For LLM: q_head index. For fused: combined batch*H + head index.
  const int q_h = int(gl_GlobalInvocationID.z);

  // idx along the output context_len dim
  const int c = tile_idx_x * TILE_N;
  const int c4 = div_4(c);

  // idx along the output seq_len dim
  const int s = tile_idx_y * TILE_M;

#ifdef HAS_INPUT_POS
  // LLM: q_sizes is WHCN {D, H_q, S, B}
  const int D = q_sizes.x;
  const int Q_H = q_sizes.y;
  const int S = q_sizes.z;
  // k_sizes is WHCN {D, H_kv, C_max, B}
  const int KV_H = k_sizes.y;
  const int C = k_sizes.z;
#else
  // Fused: q_sizes is WHCN {D, S, H, B}
  const int D = q_sizes.x;
  const int S = q_sizes.y;
  const int Q_H = q_sizes.z;
  // k_sizes is WHCN {D, L, H, B}
  const int KV_H = k_sizes.z;
  const int C = k_sizes.y;
#endif
  const int D4 = div_up_4(D);
  const int S_aligned = align_up_4(S);

#ifdef HAS_INPUT_POS
  // current context length for LLM decode/prefill
  const int context_len = input_pos + S;
#else
  // fused: full key sequence length from k_sizes
  const int context_len = k_sizes.y;
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
  if (c >= context_len || s >= S || q_h >= Q_H * q_sizes.w) {
    return;
  }

  FPOutTile out_tile;
  initialize(out_tile);

  FPInputTile q_tile;
  FPWeightTile w_tile;

  // The LLM attn_weights tensor is padded to S_aligned in its S dim, while
  // fused attn_weights is not padded. The store/bias helpers bound-check
  // against this.
#ifdef HAS_INPUT_POS
  const int attn_S = S_aligned;
#else
  const int attn_S = S;
#endif

#ifdef HAS_INPUT_POS
  // If the tile is completely inside the mask region, then there is no need to
  // compute the output tile. All the elements in the output tile can be set to
  // negative infinity.
  bool tile_in_mask_region = c > (input_pos + s + (TILE_M - 1));
  if (tile_in_mask_region) {
    const VEC4_T negative_infinity_vec = VEC4_T(negative_infinity_val);
    set_out_tile_to_vec(out_tile, negative_infinity_vec);
  } else
#endif
  {
    for (int d4 = 0; d4 < D4; d4++) {
      load_q_projected_tile_with_checks(
        q_tile,
        d4,
        s,
        q_h,
        D4,
        D,
        S,
        Q_H);

      load_k_cache_tile_with_checks(
        w_tile,
        d4,
        c,
        kv_h,
        D4,
        D,
        context_len,
        C,
        KV_H);

      fp_accumulate_with_fp_weight(out_tile, q_tile, w_tile);
    }

#ifdef HAS_INPUT_POS
    // LLM: combined scale + causal mask
    VEC4_T inv_scale_vec = VEC4_T(inv_scale);
    apply_scale_and_mask(
      out_tile,
      inv_scale_vec,
      input_pos,
      c,
      s);
#else
    // Fused: scalar scale, optional bias
    apply_scale(out_tile, inv_scale);
  #ifdef HAS_BIAS
    apply_bias(out_tile, c4, s, q_h, context_texel_len, attn_S);
  #endif
#endif
  }

  store_attn_weight_tile_with_checks(
    out_tile,
    c4,
    s,
    q_h,
    context_texel_len,
    attn_S,
    Q_H);
}
