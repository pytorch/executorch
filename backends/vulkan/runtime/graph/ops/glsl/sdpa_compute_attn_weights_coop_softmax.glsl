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

// Cap on the attn_weights row held in shared memory, in texels. The host only
// selects this shader when div_up_4(max_context_len) fits within this bound
// (see use_fused_qk_softmax() in SDPA.cpp), so the indexing below is in range.
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

// The attn_weights row after scale and causal masking, plus per-worker scratch
// for the row-max and exp-sum tree reductions.
shared T row_attn_w_x[MAX_CONTEXT_TEXEL_LEN];
shared T row_attn_w_y[MAX_CONTEXT_TEXEL_LEN];
shared T row_attn_w_z[MAX_CONTEXT_TEXEL_LEN];
shared T row_attn_w_w[MAX_CONTEXT_TEXEL_LEN];
shared T shared_max[NUM_WORKERS_PER_OUT];
shared T shared_sum[NUM_WORKERS_PER_OUT];

/*
 * Fused Q @ K^T and softmax for the decode path (sequence length 1).
 *
 * The coop sibling of this shader dispatches one work group per (c4, q_h) and
 * tree-reduces across head_dim, each group emitting one texel of attn_weights;
 * a separate shader then performs the softmax across the context dim in a
 * three-pass max/subtract/normalize algorithm.
 *
 * This variant instead dispatches one work group per (s, q_h), whose 64 workers
 * cooperatively walk every c4 of that row. The row stays in shared memory, so
 * the group can reduce it to row_max and exp_sum and write normalized softmax
 * values straight to t_attn_weights. That removes one dispatch per layer along
 * with the round trip through attn_weights between the matmul and the softmax.
 *
 * The trade-off is the loss of parallelism along the context axis (from one
 * work group per context tile down to one), which is why the host restricts
 * this shader to sequence length 1, where per-dispatch overhead dominates.
 */
void main() {
  const int worker_id = int(gl_LocalInvocationID.x);

  // idx along the output seq_len dim; this shader is only dispatched for S == 1
  const int s = int(gl_GlobalInvocationID.y);
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

  // bounds check
  if (s >= S || q_h >= Q_H) {
    return;
  }

  const VEC4_T inv_scale_vec = VEC4_T(inv_scale);
  T local_max = T(negative_infinity_val);

  FPInputTile q_tile;
  FPWeightTile w_tile;
  FPOutTile out_tile;

  // Pass 1: compute the attn_weight texels this worker owns, apply scale and
  // the causal mask, stash the row in shared memory and track a local max.
  // Worker w owns c4 = w, w + 64, w + 128, ...
  for (int c4 = worker_id; c4 < context_texel_len; c4 += NUM_WORKERS_PER_OUT) {
    const int c = mul_4(c4);

    initialize(out_tile);

    // If the tile is completely inside the mask region there is nothing to
    // compute; every element is negative infinity.
    const bool tile_in_mask_region = c > (input_pos + s + (TILE_M - 1));
    if (tile_in_mask_region) {
      const VEC4_T negative_infinity_vec = VEC4_T(negative_infinity_val);
      set_out_tile_to_vec(out_tile, negative_infinity_vec);
    } else {
      for (int d4 = 0; d4 < D4; ++d4) {
        load_q_projected_tile_with_checks(
            q_tile, d4, s, q_h, D4, D, S, Q_H);

        load_k_cache_tile_with_checks(
            w_tile, d4, c, kv_h, D4, D, context_len, C, KV_H);

        fp_accumulate_with_fp_weight(out_tile, q_tile, w_tile);
      }

      apply_scale_and_mask(out_tile, inv_scale_vec, input_pos, c, s);
    }

    // TILE_M is 1 and TILE_N4 is 1, so the tile holds a single texel.
    VEC4_T raw = out_tile.data[0][0];

    // Positions past context_len inside the final texel must not contribute to
    // the row max; setting them to negative infinity makes their exp() zero.
    if (c + 0 >= context_len) {
      raw.x = T(negative_infinity_val);
    }
    if (c + 1 >= context_len) {
      raw.y = T(negative_infinity_val);
    }
    if (c + 2 >= context_len) {
      raw.z = T(negative_infinity_val);
    }
    if (c + 3 >= context_len) {
      raw.w = T(negative_infinity_val);
    }

    row_attn_w_x[c4] = raw.x;
    row_attn_w_y[c4] = raw.y;
    row_attn_w_z[c4] = raw.z;
    row_attn_w_w[c4] = raw.w;

    local_max = max(local_max, max(max(raw.x, raw.y), max(raw.z, raw.w)));
  }

  // Reduce the row max across workers.
  shared_max[worker_id] = local_max;
  memoryBarrierShared();
  barrier();

  for (int i = NUM_WORKERS_PER_OUT / 2; i > 0; i /= 2) {
    if (worker_id < i) {
      shared_max[worker_id] =
          max(shared_max[worker_id], shared_max[worker_id + i]);
    }
    memoryBarrierShared();
    barrier();
  }
  const T row_max = shared_max[0];

  // Pass 2: each worker sums exp(x - row_max) over the texels it owns.
  T local_sum = T(0);
  for (int c4 = worker_id; c4 < context_texel_len; c4 += NUM_WORKERS_PER_OUT) {
    local_sum += exp(row_attn_w_x[c4] - row_max);
    local_sum += exp(row_attn_w_y[c4] - row_max);
    local_sum += exp(row_attn_w_z[c4] - row_max);
    local_sum += exp(row_attn_w_w[c4] - row_max);
  }

  shared_sum[worker_id] = local_sum;
  memoryBarrierShared();
  barrier();

  for (int i = NUM_WORKERS_PER_OUT / 2; i > 0; i /= 2) {
    if (worker_id < i) {
      shared_sum[worker_id] = shared_sum[worker_id] + shared_sum[worker_id + i];
    }
    memoryBarrierShared();
    barrier();
  }
  const T row_sum = shared_sum[0];

  // Pass 3: write exp(x - row_max) / row_sum. Out of bounds positions in the
  // final texel were set to negative infinity above, so they store zero, which
  // is what the consumer's load-with-checks would have supplied anyway.
  for (int c4 = worker_id; c4 < context_texel_len; c4 += NUM_WORKERS_PER_OUT) {
    VEC4_T raw;
    raw.x = row_attn_w_x[c4];
    raw.y = row_attn_w_y[c4];
    raw.z = row_attn_w_z[c4];
    raw.w = row_attn_w_w[c4];

    FPOutTile out_tile_normalized;
    out_tile_normalized.data[0][0] = exp(raw - VEC4_T(row_max)) / row_sum;

    store_attn_weight_tile_with_checks(
        out_tile_normalized, c4, s, q_h, context_texel_len, S_aligned, Q_H);
  }
}
