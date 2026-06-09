/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * KHR Cooperative Matrix variant of linear_dq8ca_q8csw_tiled.
 *
 * Performs: out[M,N] = dequant(int8_act) * dequant(int8_w_perchannel) (+ bias)
 *
 * Uses coopmat<int8> × coopmat<int8> → coopmat<int32> on the matrix unit.
 *
 * Math (per output tile element):
 *   accum_int32 = sum_k(int8_in_k * int8_weight_k)        // coopMatMulAdd
 *   adjusted    = accum_int32 - input_zp[m] * weight_sum[n]
 *   result_fp   = float(adjusted) * input_scale[m] * weight_scale[n]
 *
 * Differences from linear_dq8ca_q4gsw_coopmat (the int4 sibling):
 *   1. B-stage loads int8 weight directly (no nibble unpack, no -8 bias).
 *   2. No per-group loop — per-channel weight quant has no groups, so a
 *      single K loop runs the full accumulation, then one epilog dequant.
 *   3. wsum / wsc / izp / ifs are all loaded ONCE per WG tile (no group
 *      ping-pong).
 *
 * Loop structure follows the NVIDIA double-buffered GEMM reference
 * (shmem_double_buf4.comp "store-first" variant; see gemm_double_buf.glsl in
 * test/custom_ops): prologue register prefetch, one barrier per K-chunk,
 * ping-pong LDS slices; the prefetch is pure loads, in flight during the MMA.
 *
 * LDS layout for the MMA operands: K-slab split + ColumnMajor B + per-col
 * skew padding (see the comments in the staging blocks; the source int8
 * weight block already packs 4 K-contiguous bytes per N-col, so the
 * ColumnMajor LDS write is a straight uint copy).
 *
 * Tile hierarchy (yaml): MMA 16x16x16 int8, WG_TILE 128x64, WG_TILE_K = 32,
 * 4 subgroups × 64 threads. The double-buffered reference's subgroup-32
 * layout is NOT used: the Xclipse PAL compiler crashes in
 * vkCreateComputePipelines when int8 WMMA is compiled at forced subgroup
 * size 32 (fp16 WMMA at 32 is fine; see linear_q4gsw_coopmat).
 *
 * Hard preconditions:
 *   M % WG_TILE_M == 0, N % WG_TILE_N == 0, K % WG_TILE_K == 0,
 *   device exposes coopmat<int8>×<int8>→<int32> at 16x16x16.
 */

#version 450 core

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : enable

#define PRECISION ${PRECISION}

$if HAS_BIAS:
  #define HAS_BIAS

$if WEIGHT_STORAGE == "buffer":
  #define WEIGHT_BUFFER

layout(std430) buffer;

#include "common.glslh"

// Bindings — match add_linear_dqa_qw_node arg order:
//   output(0), fp_input(1), packed_int8_input(2), int_input_sums(3 - unused),
//   input_scales(4), input_zps(5), packed_int8_weight(6), weight_sums(7),
//   weight_scales(8), bias(9).
${layout_declare_tensor(B, "w", "t_output",              "half", "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_input",               "half", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_packed_int8_input",   "int",  "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_int8_input_sums",     "int",  "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_int8_input_scales",   "half", "texture3d")}
${layout_declare_tensor(B, "r", "t_int8_input_zps",      "int8", "texture3d")}
${layout_declare_tensor(B, "r", "t_packed_int8_weight",  "int",  WEIGHT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_sums",         "int",  "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_weight_scales",       "half", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias",                "half", "buffer", is_scalar_array=True)}

${layout_declare_ubo(B, "ivec4", "output_sizes")}
${layout_declare_ubo(B, "ivec4", "input_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "apply_bias",   "0")}
// K4_per_group kept as an inert spec const so the dispatcher binding (which
// passes {apply_bias, K4_per_group} unconditionally) lines up.  Per-channel
// weight has no groups; the shader ignores this value.
${layout_declare_spec_const(C, "int", "K4_per_group", "0")}
${layout_declare_spec_const(C, "int", "k_chunks_arg", "0")}
// Output width N for coopMatStore: the Xclipse compiler MISCOMPILES
// coopMatStore whose offset/stride derive from a UBO value (only the first
// store per subgroup lands correctly; standalone repro cm_acc2).
${layout_declare_spec_const(C, "int", "out_N_arg", "0")}

// Tile geometry
const uint MMA_M = ${MMA_M};
const uint MMA_N = ${MMA_N};
const uint MMA_K = ${MMA_K};

const uint WG_TILE_M = ${WG_TILE_M};
const uint WG_TILE_N = ${WG_TILE_N};
const uint WG_TILE_K = ${WG_TILE_K};

const uint SG_GRID_X = ${SG_GRID_X};
const uint SG_GRID_Y = ${SG_GRID_Y};
const uint SUBGROUP_SIZE = ${SUBGROUP_SIZE};
const uint NUM_SUBGROUPS = SG_GRID_X * SG_GRID_Y;
const uint WG_SIZE = NUM_SUBGROUPS * SUBGROUP_SIZE;

const uint SG_TILE_M = WG_TILE_M / SG_GRID_Y;
const uint SG_TILE_N = WG_TILE_N / SG_GRID_X;
const uint MMAS_PER_SG_M = SG_TILE_M / MMA_M;
const uint MMAS_PER_SG_N = SG_TILE_N / MMA_N;

// LDS layout: K-slab split + ColumnMajor B + per-col skew padding on B (see
// the pre-dbuf revision of this file for the full rationale: matB lane
// layout wants 4 K-contiguous bytes per lane; ColumnMajor + a +1-uint skew
// per col gives one ds_load_b32 per lane, bank-conflict-free).
const uint A_SLAB_INT8     = WG_TILE_M * MMA_K;
const uint B_USEFUL_U32    = MMA_K / 4u;
const uint B_STRIDE_U32    = B_USEFUL_U32 + 1u;
const uint B_SLAB_U32      = WG_TILE_N * B_STRIDE_U32;
const uint NUM_K_SLABS     = WG_TILE_K / MMA_K;

const uint A_SLAB_U32      = A_SLAB_INT8 / 4u;
const uint A_STRIDE_U32    = MMA_K / 4u;

// One ping-pong slice covers all K-slabs of one chunk.
const uint ASH_SLICE_U32 = NUM_K_SLABS * A_SLAB_U32;
const uint BSH_SLICE_U32 = NUM_K_SLABS * B_SLAB_U32;

// Double-buffered MMA operand staging.
shared uint Ash_int8[2u * ASH_SLICE_U32];
shared uint Bsh_int8[2u * BSH_SLICE_U32];

// Per-WG-tile-row activation params (loaded ONCE at WG start).
shared int   izp_sh[WG_TILE_M];   // int32 (cast from int8 source)
shared float ifs_sh[WG_TILE_M];   // float32 (cast from fp16 source)

// Per-output-channel weight params (loaded ONCE at WG start — per-channel,
// not per-group, unlike the q4gsw_coopmat variant).
shared int   wsum_sh[WG_TILE_N];
shared float wsc_sh[WG_TILE_N];

#ifdef HAS_BIAS
shared float bias_sh[WG_TILE_N];
#endif

void main() {
  const uvec2 tileID = uvec2(gl_WorkGroupID.xy);
  const uvec2 warpInTile = uvec2(
      gl_SubgroupID % SG_GRID_X,
      gl_SubgroupID / SG_GRID_X);

  const uint K = uint(input_sizes.x);
  const uint N = uint(output_sizes.x);
  const uint N4 = (N + 3u) / 4u;
  const uint nblocks_x_A = (K + 3u) >> 2u;
  const uint num_chunks = uint(k_chunks_arg);

  const uint tile_m_start = WG_TILE_M * tileID.y;
  const uint tile_n_start = WG_TILE_N * tileID.x;

  // --- One-time stage: per-row input zp + scale ---
  if (gl_LocalInvocationID.x < (WG_TILE_M >> 2u)) {
    const uint m4 = (tile_m_start >> 2u) + gl_LocalInvocationID.x;
    const vec4  sc = vec4(texelFetch(t_int8_input_scales, ivec3(m4, 0, 0), 0));
    const ivec4 zp = texelFetch(t_int8_input_zps,         ivec3(m4, 0, 0), 0);
    const uint base = gl_LocalInvocationID.x * 4u;
    ifs_sh[base + 0u] = sc.x;  ifs_sh[base + 1u] = sc.y;
    ifs_sh[base + 2u] = sc.z;  ifs_sh[base + 3u] = sc.w;
    izp_sh[base + 0u] = zp.x;  izp_sh[base + 1u] = zp.y;
    izp_sh[base + 2u] = zp.z;  izp_sh[base + 3u] = zp.w;
  }

  // --- One-time stage: per-output-channel weight scale + sum ---
  if (gl_LocalInvocationID.x < WG_TILE_N) {
    const uint n_idx = tile_n_start + gl_LocalInvocationID.x;
    const uint n4_idx = n_idx >> 2u;
    const uint n4_off = n_idx & 3u;
    f16vec4 sv = t_weight_scales[n4_idx];
    wsc_sh[gl_LocalInvocationID.x] = float(sv[n4_off]);
    wsum_sh[gl_LocalInvocationID.x] = t_weight_sums[n_idx];
  }
  memoryBarrierShared();
  barrier();

  // --- Single INT32 cooperative-matrix accumulator (full K accumulation) ---
  coopmat<int32_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>
      accum_int32[MMAS_PER_SG_M][MMAS_PER_SG_N];
  [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
    [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
      accum_int32[i][j] =
          coopmat<int32_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(0);
    }
  }

  // --- A staging thread map: one (m4, k4) ivec4 block per active thread ---
  // (4 M-rows x 4 K-positions; each block expands to 4 slab-major LDS uints.)
  const uint K_BLOCKS_PER_CHUNK = WG_TILE_K >> 2u;
  const uint A_ACTIVE_THREADS = (WG_TILE_M >> 2u) * K_BLOCKS_PER_CHUNK;
  const uint a_m_block = gl_LocalInvocationID.x / K_BLOCKS_PER_CHUNK;
  const uint a_k_block = gl_LocalInvocationID.x % K_BLOCKS_PER_CHUNK;
  const bool a_active = gl_LocalInvocationID.x < A_ACTIVE_THREADS;

  // --- B staging thread map: one (k4, n4) ivec4 block per active thread ---
  // wblk[n_in_blk] packs 4 K-contiguous bytes for N-col (n4*4 + n_in_blk) —
  // exactly one ColumnMajor LDS uint, written as-is (no byte repack).
  const uint B_FETCH_SLOTS = K_BLOCKS_PER_CHUNK * (WG_TILE_N >> 2u);
  const uint N4_PER_TILE = WG_TILE_N >> 2u;
  const uint b_k4_in_chunk = gl_LocalInvocationID.x / N4_PER_TILE;
  const uint b_n_uint_col = gl_LocalInvocationID.x % N4_PER_TILE;
  const bool b_active = gl_LocalInvocationID.x < B_FETCH_SLOTS;

  // Prefetch temp registers.
  ivec4 temp_A;
  ivec4 temp_B;

  // =========================================================
  // PROLOGUE: prefetch chunk 0 into temp registers, then store to slice 0
  // (no barrier; the first loop iteration's barrier publishes it).
  // =========================================================
  if (a_active) {
    const uint m4_global = (tile_m_start >> 2u) + a_m_block;
    temp_A = t_packed_int8_input[m4_global * nblocks_x_A + a_k_block];
  }
  if (b_active) {
    const uint block_x_w = (tile_n_start >> 2u) + b_n_uint_col;
#ifdef WEIGHT_BUFFER
    temp_B = t_packed_int8_weight[(b_k4_in_chunk * N4) + block_x_w];
#else
    temp_B = texelFetch(t_packed_int8_weight, ivec2(block_x_w, b_k4_in_chunk), 0);
#endif
  }
  {
    if (a_active) {
      const uint slab_idx       = a_k_block / (MMA_K >> 2u);
      const uint k_uint_in_slab = a_k_block % (MMA_K >> 2u);
      const uint base_row = a_m_block * 4u;
      [[unroll]] for (uint m4i = 0; m4i < 4u; ++m4i) {
        Ash_int8[slab_idx * A_SLAB_U32 + (base_row + m4i) * A_STRIDE_U32 + k_uint_in_slab] =
            uint(temp_A[m4i]);
      }
    }
    if (b_active) {
      const uint slab_idx   = b_k4_in_chunk / (MMA_K >> 2u);
      const uint k4_in_slab = b_k4_in_chunk % (MMA_K >> 2u);
      const uint n_col_base = b_n_uint_col * 4u;
      [[unroll]] for (uint n_in_blk = 0u; n_in_blk < 4u; ++n_in_blk) {
        Bsh_int8[slab_idx * B_SLAB_U32 + (n_col_base + n_in_blk) * B_STRIDE_U32 + k4_in_slab] =
            uint(temp_B[n_in_blk]);
      }
    }
  }

  // =========================================================
  // MAIN LOOP — one barrier per chunk. Iteration `chunk`:
  //   1. barrier   — slice (chunk%2) fully written
  //   2. prefetch  — chunk+1 into temp (skipped on the final chunk)
  //   3. int8 MMA  — on slice (chunk%2) into accum_int32
  //   4. store     — temp -> slice ((chunk+1)%2)
  // =========================================================
  for (uint chunk = 0; chunk < num_chunks; ++chunk) {
    const bool has_next = chunk + 1u < num_chunks;
    const uint cur_a = (chunk % 2u) * ASH_SLICE_U32;
    const uint cur_b = (chunk % 2u) * BSH_SLICE_U32;
    const uint nxt_a = ((chunk + 1u) % 2u) * ASH_SLICE_U32;
    const uint nxt_b = ((chunk + 1u) % 2u) * BSH_SLICE_U32;

    barrier();

    // --- 2. prefetch chunk+1 -> temp ---
    if (has_next) {
      const uint chunkK_nxt = (chunk + 1u) * WG_TILE_K;
      if (a_active) {
        const uint m4_global = (tile_m_start >> 2u) + a_m_block;
        const uint k4_global = (chunkK_nxt >> 2u) + a_k_block;
        temp_A = t_packed_int8_input[m4_global * nblocks_x_A + k4_global];
      }
      if (b_active) {
        const uint block_y_w = (chunkK_nxt >> 2u) + b_k4_in_chunk;
        const uint block_x_w = (tile_n_start >> 2u) + b_n_uint_col;
#ifdef WEIGHT_BUFFER
        temp_B = t_packed_int8_weight[(block_y_w * N4) + block_x_w];
#else
        temp_B = texelFetch(t_packed_int8_weight, ivec2(block_x_w, block_y_w), 0);
#endif
      }
    }

    // --- 3. int8 MMA on the cur slice ---
    [[unroll]] for (uint k = 0; k < NUM_K_SLABS; ++k) {
      const uint slab_a_base_u32 = cur_a + k * A_SLAB_U32;
      const uint slab_b_base_u32 = cur_b + k * B_SLAB_U32;

      coopmat<int8_t, gl_ScopeSubgroup, MMA_M, MMA_K, gl_MatrixUseA> matA[MMAS_PER_SG_M];
      [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
        const uint row_a = MMA_M * (MMAS_PER_SG_M * warpInTile.y + i);
        coopMatLoad(
            matA[i], Ash_int8,
            slab_a_base_u32 + row_a * A_STRIDE_U32,
            A_STRIDE_U32,
            gl_CooperativeMatrixLayoutRowMajor);
      }

      coopmat<int8_t, gl_ScopeSubgroup, MMA_K, MMA_N, gl_MatrixUseB> matB;
      [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
        const uint col_b = MMA_N * (MMAS_PER_SG_N * warpInTile.x + j);
        coopMatLoad(
            matB, Bsh_int8,
            slab_b_base_u32 + col_b * B_STRIDE_U32,
            B_STRIDE_U32,
            gl_CooperativeMatrixLayoutColumnMajor);
        [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
          accum_int32[i][j] = coopMatMulAdd(matA[i], matB, accum_int32[i][j]);
        }
      }
    }

    // --- 4. store temp (chunk+1) -> nxt slice ---
    if (has_next) {
      if (a_active) {
        const uint slab_idx       = a_k_block / (MMA_K >> 2u);
        const uint k_uint_in_slab = a_k_block % (MMA_K >> 2u);
        const uint base_row = a_m_block * 4u;
        [[unroll]] for (uint m4i = 0; m4i < 4u; ++m4i) {
          Ash_int8[nxt_a + slab_idx * A_SLAB_U32 + (base_row + m4i) * A_STRIDE_U32 + k_uint_in_slab] =
              uint(temp_A[m4i]);
        }
      }
      if (b_active) {
        const uint slab_idx   = b_k4_in_chunk / (MMA_K >> 2u);
        const uint k4_in_slab = b_k4_in_chunk % (MMA_K >> 2u);
        const uint n_col_base = b_n_uint_col * 4u;
        [[unroll]] for (uint n_in_blk = 0u; n_in_blk < 4u; ++n_in_blk) {
          Bsh_int8[nxt_b + slab_idx * B_SLAB_U32 + (n_col_base + n_in_blk) * B_STRIDE_U32 + k4_in_slab] =
              uint(temp_B[n_in_blk]);
        }
      }
    }
  }  // K chunks

  // --- Single epilog: coopmat-only dequant of accum_int32 -> fp result ---
  //   adjusted = accum_int32 - izp_bcast * wsum_bcast    (int32 element-wise)
  //   result   = float(adjusted) * (ifs_bcast * wsc_bcast)  (fp element-wise)
  coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>
      result[MMAS_PER_SG_M][MMAS_PER_SG_N];

  [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
    [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
      const uint local_m_base = MMA_M * (MMAS_PER_SG_M * warpInTile.y + i);
      const uint local_n_base = MMA_N * (MMAS_PER_SG_N * warpInTile.x + j);

      coopmat<int32_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> wsum_bcast;
      coopMatLoad(
          wsum_bcast, wsum_sh,
          local_n_base, /*stride=*/0u,
          gl_CooperativeMatrixLayoutRowMajor);

      coopmat<int32_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> izp_bcast;
      coopMatLoad(
          izp_bcast, izp_sh,
          local_m_base, /*stride=*/0u,
          gl_CooperativeMatrixLayoutColumnMajor);

      coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> wsc_bcast;
      coopMatLoad(
          wsc_bcast, wsc_sh,
          local_n_base, /*stride=*/0u,
          gl_CooperativeMatrixLayoutRowMajor);

      coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> ifs_bcast;
      coopMatLoad(
          ifs_bcast, ifs_sh,
          local_m_base, /*stride=*/0u,
          gl_CooperativeMatrixLayoutColumnMajor);

      coopmat<int32_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> adjusted =
          accum_int32[i][j] - izp_bcast * wsum_bcast;

      coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> adjusted_fp =
          coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(adjusted);

      coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> scales_outer =
          ifs_bcast * wsc_bcast;

      result[i][j] = adjusted_fp * scales_outer;
    }
  }

  // --- Bias (optional) ---
#ifdef HAS_BIAS
  if (apply_bias > 0) {
    for (uint t = gl_LocalInvocationID.x; t < WG_TILE_N; t += WG_SIZE) {
      bias_sh[t] = float(t_bias[tile_n_start + t]);
    }
    memoryBarrierShared();
    barrier();
  }
#endif

  // --- Store result tile ---
  // N for the store address math MUST come from the spec constant, not the
  // sizes UBO (see out_N_arg above).
  const uint N_out = uint(out_N_arg);
  [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
    [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
      const uint gi = tile_m_start + MMA_M * (MMAS_PER_SG_M * warpInTile.y + i);
      const uint gj = tile_n_start + MMA_N * (MMAS_PER_SG_N * warpInTile.x + j);

#ifdef HAS_BIAS
      if (apply_bias > 0) {
        const uint local_n = MMA_N * (MMAS_PER_SG_N * warpInTile.x + j);
        coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> bias_tile;
        coopMatLoad(bias_tile, bias_sh, local_n, 0u, gl_CooperativeMatrixLayoutRowMajor);
        result[i][j] += bias_tile;
      }
#endif

      coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> out_tile =
          coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(result[i][j]);
      coopMatStore(
          out_tile, t_output,
          gi * N_out + gj, N_out,
          gl_CooperativeMatrixLayoutRowMajor);
    }
  }
}
