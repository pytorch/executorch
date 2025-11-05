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
$if K_CACHE_STORAGE == "buffer":
  #define K_CACHE_BUFFER

#define TILE_K4 ${TILE_K4}
#define TILE_N4 ${TILE_N4}

#define TILE_M 1
#define TILE_K ${TILE_K4 * 4}
#define TILE_N ${TILE_N4 * 4}

#define NUM_WORKERS_PER_OUT 64

${define_required_extensions(DTYPE)}

layout(std430) buffer;

#include "common.glslh"

${layout_declare_tensor(B, "w", "t_attn_weights", DTYPE, IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_q_projected", DTYPE, IO_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_k_cache", DTYPE, K_CACHE_STORAGE, is_scalar_array=False)}

${layout_declare_ubo(B, "ivec4", "q_projected_sizes")}
${layout_declare_ubo(B, "ivec4", "k_cache_sizes")}
${layout_declare_ubo(B, "int", "input_pos")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "float", "inv_scale", "1.0")}

#include "sdpa_fp_q_projected_tile_load.glslh"
#include "sdpa_fp_k_cache_tile_load.glslh"
#include "linear_fp_output_tile_fp_compute.glslh"
#include "sdpa_fp_attn_weight_tile_store.glslh"

shared FPOutTile partial_sums[NUM_WORKERS_PER_OUT];

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

  // idx along the output context_len dim
  const int c = tile_idx_x * TILE_N;
  const int c4 = div_4(c);

  // idx along the output seq_len dim. Note that for this shader seq_len will be
  // 1.
  const int s = 0;

  // texel size of head_dim, over which the dot product is accumulated
  const int D4 = div_up_4(q_projected_sizes.x);
  // number of Q heads
  const int Q_H = q_projected_sizes.y;
  // sequence length
  const int S = q_projected_sizes.z;
  const int S_aligned = align_up_4(S);

  // number of K/V heads
  const int KV_H = k_cache_sizes.y;
  // Max context length
  const int C = k_cache_sizes.z;
  const int C4 = div_up_4(C);

  int kv_h = q_h;
  if (KV_H < Q_H) {
    kv_h = q_h / (Q_H / KV_H);
  }

  // current context length
  const int context_len = input_pos + S;
  const int context_texel_len = div_up_4(context_len);

  // bounds check
  if (c >= context_len || s >= S || q_h >= Q_H) {
    return;
  }

  FPOutTile out_tile;
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
    const bool dont_check_bounds = (S - s) >= TILE_M &&
        (context_len - c) >= TILE_N;

    if (dont_check_bounds) {
      for (int d4 = worker_id; d4 < D4; d4 += NUM_WORKERS_PER_OUT) {
        load_q_projected_tile_no_checks(
          q_tile,
          d4,
          s,
          q_h,
          D4,
          Q_H,
          S);

        load_k_cache_tile_no_checks(
          w_tile,
          d4,
          c,
          kv_h,
          D4,
          context_len,
          C,
          KV_H);

        fp_accumulate_with_fp_weight(out_tile, q_tile, w_tile);
      }
    } else {
      for (int d4 = worker_id; d4 < D4; d4 += NUM_WORKERS_PER_OUT) {
        load_q_projected_tile_with_checks(
          q_tile,
          d4,
          s,
          q_h,
          D4,
          Q_H,
          S);

        load_k_cache_tile_with_checks(
          w_tile,
          d4,
          c,
          kv_h,
          D4,
          context_len,
          C,
          KV_H);

        fp_accumulate_with_fp_weight(out_tile, q_tile, w_tile);
      }
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
}
