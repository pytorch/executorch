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
$if K_CACHE_STORAGE == "buffer":
  #define K_CACHE_BUFFER

#define Q_LAYOUT DHSB
#define K_LAYOUT DHSB

#define TILE_K4 ${TILE_K4}
#define TILE_N4 ${TILE_N4}

#define TILE_M 1
#define TILE_K ${TILE_K4 * 4}
#define TILE_N ${TILE_N4 * 4}

#define NUM_WORKERS_PER_OUT 64

$if FUSE_SOFTMAX:
  #define FUSE_SOFTMAX
  // Cap on the attn_weights row held in shared memory, in texels. The host only
  // selects this variant when the context fits within this bound (see
  // use_fused_qk_softmax() in SDPA.cpp), so the indexing below is in range.
  #define MAX_CONTEXT_TEXEL_LEN 1024

${define_required_extensions(IO_STORAGE, DTYPE)}

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_attn_weights", DTYPE, IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_q", DTYPE, IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_k", DTYPE, K_CACHE_STORAGE, is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "q_sizes")}
${layout_declare_ubo(B, "ivec4", "k_sizes")}
${layout_declare_ubo(B, "int", "input_pos")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "float", "inv_scale", "1.0")}

#include "sdpa_fp_q_projected_tile_load.glslh"
#include "sdpa_fp_k_cache_tile_load.glslh"
#include "linear_fp_output_tile_fp_compute.glslh"
#include "sdpa_fp_attn_weight_tile_store.glslh"

#ifdef FUSE_SOFTMAX

// The scaled and masked attn_weights row, one array per texel component.
shared T row_attn_w_x[MAX_CONTEXT_TEXEL_LEN];
shared T row_attn_w_y[MAX_CONTEXT_TEXEL_LEN];
shared T row_attn_w_z[MAX_CONTEXT_TEXEL_LEN];
shared T row_attn_w_w[MAX_CONTEXT_TEXEL_LEN];

#define NUM_WORKERS_PER_WG NUM_WORKERS_PER_OUT
#define SOFTMAX_IN_VEC4_T VEC4_T
#define SOFTMAX_ACC_T T

// The softmax reads the row back from shared memory...
VEC4_T load_attn_weights_c4(
    const int c4,
    const int s,
    const int q_h,
    const int C4,
    const int S,
    const int Q_H) {
  return VEC4_T(
      row_attn_w_x[c4], row_attn_w_y[c4], row_attn_w_z[c4], row_attn_w_w[c4]);
}

// ...and writes the normalized weights to t_attn_weights.
void store_attn_weights_softmax_c4(
    const VEC4_T out_texel,
    const int c4,
    const int s,
    const int q_h,
    const int C4,
    const int S,
    const int Q_H) {
  store_attn_weight_c4(out_texel, c4, s, q_h, C4, S, Q_H);
}

#include "sdpa_attn_weights_softmax_row.glslh"

#else

shared FPOutTile partial_sums[NUM_WORKERS_PER_OUT];

#endif

/*
 * Accumulates Q @ K^T for the attn_weights tile at (c, s, q_h) over the head
 * dim texels d4_start, d4_start + d4_step, ... A tile entirely inside the
 * causal mask is set to negative infinity instead, and true is returned, in
 * which case no scale or mask needs to be applied to it.
 */
bool compute_attn_weight_tile(
    out FPOutTile out_tile,
    const int d4_start,
    const int d4_step,
    const int c,
    const int s,
    const int q_h,
    const int kv_h,
    const int D4,
    const int D,
    const int S,
    const int Q_H,
    const int context_len,
    const int C,
    const int KV_H) {
  initialize(out_tile);

  FPInputTile q_tile;
  FPWeightTile w_tile;

  // If the tile is completely inside the mask region, then there is no need to
  // compute the output tile. All the elements in the output tile can be set to
  // negative infinity.
  bool tile_in_mask_region = c > (input_pos + s + (TILE_M - 1));
  if (tile_in_mask_region) {
    const VEC4_T negative_infinity_vec = VEC4_T(negative_infinity_val);
    set_out_tile_to_vec(out_tile, negative_infinity_vec);
  }
  // Otherwise, need to actually compute output tile
  else {
    for (int d4 = d4_start; d4 < D4; d4 += d4_step) {
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
  }
  return tile_in_mask_region;
}

/*
 * See the tiled variant of this shader for the implemented behavior. This
 * shader is implements an optimization for cases where sequence length is 1; in
 * these cases, the matrix multiplication being performed is akin to gemv, which
 * benefits from using a co-operative algorithm for reduction. For this shader
 * the entire work group co-operates to compute one reduction output.
 *
 * The FUSE_SOFTMAX variant instead dispatches one work group per (s, q_h) row.
 * Each worker computes whole texels of the row, which is kept in shared memory
 * so that the work group can apply the softmax and write normalized weights
 * directly. That saves the separate softmax dispatch and the round trip of
 * attn_weights through global memory, at the cost of parallelism along the
 * context dim, so the host only selects it for short contexts.
 */

void main() {
#ifdef FUSE_SOFTMAX
  const int worker_id = int(gl_LocalInvocationID.x);

  // idx along the output seq_len dim
  const int s = int(gl_GlobalInvocationID.y);
#else
  const int worker_id = int(gl_LocalInvocationID.y);

  const int tile_idx_x = int(gl_GlobalInvocationID.x);

  // idx along the output context_len dim
  const int c = tile_idx_x * TILE_N;
  const int c4 = div_4(c);

  // idx along the output seq_len dim. Note that for this shader seq_len will be
  // 1.
  const int s = 0;
#endif
  // idx along output num_q_heads dim
  const int q_h = int(gl_GlobalInvocationID.z);

  // head dimension
  const int D = q_sizes.x;
  // texel size of head_dim, over which the dot product is accumulated
  const int D4 = div_up_4(D);
  // number of Q heads
  const int Q_H = q_sizes.y;
  // sequence length
  const int S = q_sizes.z;
  const int S_aligned = align_up_4(S);

  // number of K/V heads
  const int KV_H = k_sizes.y;
  // Max context length
  const int C = k_sizes.z;
  const int C4 = div_up_4(C);

  int kv_h = q_h;
  if (KV_H < Q_H) {
    kv_h = q_h / (Q_H / KV_H);
  }

  // current context length
  const int context_len = input_pos + S;
  const int context_texel_len = div_up_4(context_len);

#ifdef FUSE_SOFTMAX
  // bounds check
  if (s >= S || q_h >= Q_H) {
    return;
  }

  // Compute the row into shared memory; worker w owns c4 = w, w + 64, ...
  for (int c4 = worker_id; c4 < context_texel_len;
       c4 += NUM_WORKERS_PER_OUT) {
    const int c = mul_4(c4);

    FPOutTile out_tile;
    const bool tile_in_mask_region = compute_attn_weight_tile(
        out_tile, 0, 1, c, s, q_h, kv_h, D4, D, S, Q_H, context_len, C, KV_H);
    if (!tile_in_mask_region) {
      VEC4_T inv_scale_vec = VEC4_T(inv_scale);
      apply_scale_and_mask(out_tile, inv_scale_vec, input_pos, c, s);
    }

    // TILE_M and TILE_N4 are 1, so the tile holds a single texel.
    row_attn_w_x[c4] = out_tile.data[0][0].x;
    row_attn_w_y[c4] = out_tile.data[0][0].y;
    row_attn_w_z[c4] = out_tile.data[0][0].z;
    row_attn_w_w[c4] = out_tile.data[0][0].w;
  }

  memoryBarrierShared();
  barrier();

  softmax_attn_weights_row(worker_id, s, q_h, context_len, S_aligned, Q_H);
#else
  // bounds check
  if (c >= context_len || s >= S || q_h >= Q_H) {
    return;
  }

  FPOutTile out_tile;
  const bool tile_in_mask_region = compute_attn_weight_tile(
      out_tile,
      worker_id,
      NUM_WORKERS_PER_OUT,
      c,
      s,
      q_h,
      kv_h,
      D4,
      D,
      S,
      Q_H,
      context_len,
      C,
      KV_H);

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
    // Apply scale and mask if the tile was not entirely in the mask region
    if (!tile_in_mask_region) {
      VEC4_T inv_scale_vec = VEC4_T(inv_scale);
      apply_scale_and_mask(
        out_tile,
        inv_scale_vec,
        input_pos,
        c,
        s);
    }

    store_attn_weight_tile_with_checks(
      out_tile,
      c4,
      s,
      q_h,
      context_texel_len,
      S_aligned,
      Q_H);
  }
#endif
}
