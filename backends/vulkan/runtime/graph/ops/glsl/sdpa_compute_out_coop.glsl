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
  #define ATTN_WEIGHTS_BUFFER
$if V_CACHE_STORAGE == "buffer":
  #define V_CACHE_BUFFER

#define V_LAYOUT DHSB
#define OUT_LAYOUT DHSB
#define SDPA_V_BUF t_v_cache

#define TILE_K4 ${TILE_K4}
#define TILE_N4 ${TILE_N4}

#define TILE_M 1
#define TILE_K ${TILE_K4 * 4}
#define TILE_N ${TILE_N4 * 4}

#define NUM_WORKERS_PER_OUT 64

$if GQA:
  #define GQA
  // Maximum grouped-query-attention group size (Hq / Hkv) supported by the GQA
  // variant. The actual group size G is passed as a specialization constant and
  // must satisfy G <= MAX_GROUP_SIZE. The accumulator array is sized to
  // MAX_GROUP_SIZE (a compile-time constant) and the group loop is bounded by
  // the spec-const G so the driver can fully unroll it at pipeline creation
  // time. Group sizes seen in practice: Llama G=4, Phi G=3, Qwen G=2.
  #define MAX_GROUP_SIZE 8

${define_required_extensions(IO_STORAGE, DTYPE)}

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_output", DTYPE, IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_attn_weights", DTYPE, IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_v_cache", DTYPE, V_CACHE_STORAGE, is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "q_sizes")}
${layout_declare_ubo(B, "ivec4", "v_sizes")}
${layout_declare_ubo(B, "int", "input_pos")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "float", "inv_scale", "1.0")}
$if GQA:
  ${layout_declare_spec_const(C, "int", "group_size", "1")}

#include "sdpa_fp_attn_weight_tile_load.glslh"
#include "sdpa_fp_v_cache_tile_load.glslh"
#include "linear_fp_output_tile_fp_compute.glslh"
#include "sdpa_fp_out_tile_store.glslh"

shared FPOutTile partial_sums[NUM_WORKERS_PER_OUT];

#ifdef GQA

/*
 * GQA-reuse variant of the AV coop GEMV.
 *
 * Grouped-query attention shares one KV head across G = Hq / Hkv query heads.
 * The AV computation out[q_h, d] = sum_c attn[c, q_h] * V[c, kv_h(q_h), d]
 * reads the SAME V cache for every query head in a group. In the per-query-head
 * coop variant each of the Hq heads gets its own workgroup and independently
 * re-loads the shared V cache, so V (the dominant traffic — head_dim wide per
 * context texel) is read G times. This variant assigns ONE workgroup per
 * (d4, kv_h): it loads each V texel once and reuses it across all G query heads
 * in the group, producing G output texels. This cuts V-cache traffic ~Gx for a
 * bandwidth-bound kernel. Dispatch sets the global wg z-dim to Hkv (not Hq).
 */

void main() {
  const int worker_id = int(gl_LocalInvocationID.y);

  const int tile_idx_x = int(gl_GlobalInvocationID.x);
  // idx along the K/V head dim
  const int kv_h = int(gl_GlobalInvocationID.z);

  // idx along the output head_dim dim
  const int d = tile_idx_x * TILE_N;
  const int d4 = div_4(d);

  // idx along the output seq_len dim. Note that for this shader seq_len will be
  // 1.
  const int s = 0;

  // texel size of head_dim
  const int D4 = div_up_4(q_sizes.x);
  // number of Q heads
  const int Q_H = q_sizes.y;
  // sequence length
  const int S = q_sizes.z;
  const int S_aligned = align_up_4(S);

  // number of K/V heads
  const int KV_H = v_sizes.y;
  // Max context length
  const int C = v_sizes.z;
  const int C4 = div_up_4(C);

  const int G = group_size;
  // First query head in this group.
  const int q_h_base = kv_h * G;

  // current context length
  const int context_len = input_pos + S;
  const int context_texel_len = div_up_4(context_len);

  // bounds check
  if (d4 >= D4 || s >= S || kv_h >= KV_H) {
    return;
  }

  // With head_dim output-tiling (TILE_N4 > 1) each workgroup owns TILE_N4
  // head_dim texels. Whether this workgroup owns a partial tile (only possible
  // on the final tile when D4 % TILE_N4 != 0). Uniform across the workgroup, so
  // the branch selecting the checked vs unchecked V load below is a uniform
  // branch. For the common even-D4 case (D=64 -> D4=16, D=128 -> D4=32, and
  // always for TILE_N4 == 1) this is false and the fast unchecked path is taken.
  const bool partial_n_tile = (d4 + TILE_N4) > D4;

  FPOutTile out_tile[MAX_GROUP_SIZE];
  [[unroll]] for (int g = 0; g < MAX_GROUP_SIZE; ++g) {
    if (g >= G) {
      break;
    }
    initialize(out_tile[g]);
  }

  FPInputTile attn_weight_tile;
  FPWeightTile w_tile;

  const int context_len_aligned_down = context_len - mod_4(context_len);
  const int C4_limit = div_up_4(context_len_aligned_down);

  // Main loop: each thread strides over context texels. The single V load per
  // texel is reused across all G query heads in the group.
  for (int c4 = worker_id; c4 < C4_limit; c4 += NUM_WORKERS_PER_OUT) {
    const int c = mul_4(c4);

    // The TILE_N4 V texels are loaded once per context texel and reused across
    // all G query heads in the group. Bounds-check the head_dim index only on a
    // partial final tile (uniform branch, see partial_n_tile).
    if (partial_n_tile) {
      load_v_cache_tile_with_checks(
          w_tile, d4, c, kv_h, D4, context_len, C, KV_H);
    } else {
      load_v_cache_tile_no_checks(
          w_tile, d4, c, kv_h, D4, context_len, C, KV_H);
    }

    [[unroll]] for (int g = 0; g < MAX_GROUP_SIZE; ++g) {
      if (g >= G) {
        break;
      }
      load_attn_weight_tile_no_checks(
          attn_weight_tile,
          c4,
          s,
          q_h_base + g,
          context_texel_len,
          S_aligned,
          Q_H);
      fp_accumulate_with_fp_weight(out_tile[g], attn_weight_tile, w_tile);
    }
  }

  // first worker in the work group will handle final texel, which may contain
  // padding elements.
  if (worker_id == 0) {
    for (int c4 = C4_limit; c4 < context_texel_len; c4++) {
      const int c = mul_4(c4);

      load_v_cache_tile_with_checks(
          w_tile, d4, c, kv_h, D4, context_len, C, KV_H);

      [[unroll]] for (int g = 0; g < MAX_GROUP_SIZE; ++g) {
        if (g >= G) {
          break;
        }
        load_attn_weight_tile_with_checks(
            attn_weight_tile,
            c4,
            s,
            q_h_base + g,
            context_texel_len,
            S_aligned,
            Q_H);
        fp_accumulate_with_fp_weight(out_tile[g], attn_weight_tile, w_tile);
      }
    }
  }

  // Combine the per-worker partial sums for each group with a shared-memory tree
  // reduction. partial_sums is reused across groups; no trailing barrier is
  // needed because each worker only writes its own slot and the next group's
  // leading barrier orders those writes after this group's reduction reads
  // (worker 0's store reads only slot 0, which only worker 0 writes).
  [[unroll]] for (int g = 0; g < MAX_GROUP_SIZE; ++g) {
    if (g >= G) {
      break;
    }
    partial_sums[worker_id] = out_tile[g];

    memoryBarrierShared();
    barrier();

    for (int i = NUM_WORKERS_PER_OUT / 2; i > 0; i /= 2) {
      if (worker_id < i) {
        accumulate_out_tile_with_out_tile(
            partial_sums[worker_id], partial_sums[worker_id + i]);
      }
      memoryBarrierShared();
      barrier();
    }

    if (worker_id == 0) {
      out_tile[g] = partial_sums[0];
      store_sdpa_out_tile_with_checks(
          out_tile[g], d4, s, q_h_base + g, D4, S, Q_H);
    }
  }
}

#else

/*
 * See the tiled variant of this shader for the implemented behavior. This
 * shader is implements an optimization for cases where sequence length is 1; in
 * these cases, the matrix multiplication being performed is akin to gemv, which
 * benefits from using a co-operative algorithm for reduction. For this shader
 * the entire work group co-operates to compute one reduction output.
 */

void main() {
  const int worker_id = int(gl_LocalInvocationID.y);

  const int tile_idx_x = int(gl_GlobalInvocationID.x);
  // idx along output num_q_heads dim
  const int q_h = int(gl_GlobalInvocationID.z);

  // idx along the output head_dim dim
  const int d = tile_idx_x * TILE_N;
  const int d4 = div_4(d);

  // idx along the output seq_len dim. Note that for this shader seq_len will be
  // 1.
  const int s = 0;

  // texel size of head_dim
  const int D4 = div_up_4(q_sizes.x);
  // number of Q heads
  const int Q_H = q_sizes.y;
  // sequence length
  const int S = q_sizes.z;
  const int S_aligned = align_up_4(S);

  // number of K/V heads
  const int KV_H = v_sizes.y;
  // Max context length
  const int C = v_sizes.z;
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
  const int C4_limit = div_up_4(context_len_aligned_down);

  for (int c4 = worker_id; c4 < C4_limit; c4 += NUM_WORKERS_PER_OUT) {
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
  // first worker in the work group will handle final texel, which may contain
  // padding elements.
  if (worker_id == 0) {
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
  }

  partial_sums[worker_id] = out_tile;

  memoryBarrierShared();
  barrier();

  // Tree reduction to compute the overall result.
  for (int i = NUM_WORKERS_PER_OUT / 2; i > 0; i /= 2) {
    if (worker_id < i) {
      accumulate_out_tile_with_out_tile(
          partial_sums[worker_id], partial_sums[worker_id + i]);
    }
    memoryBarrierShared();
    barrier();
  }

  // Only the first thread will write out the result
  if (worker_id == 0) {
    out_tile = partial_sums[0];
    store_sdpa_out_tile_with_checks(
      out_tile,
      d4,
      s,
      q_h,
      D4,
      S,
      Q_H);
  }
}

#endif // GQA
