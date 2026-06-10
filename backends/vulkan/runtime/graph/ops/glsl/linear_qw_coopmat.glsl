/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * KHR Cooperative Matrix variants of the weight-only quantized linear tiled
 * shaders. One template, two weight formats (WEIGHT_NBITS in the yaml):
 *   4  ->  linear_q4gsw_coopmat   INT4 group-symmetric weight
 *          (group_size = 4 * K4_per_group)
 *   8  ->  linear_q8csw_coopmat   INT8 per-channel symmetric weight
 *
 * Performs: out[M,N] = activation[M,K] * weight^T[N,K] (+ bias)
 *
 * Inner-loop math is pure fp16 -> fp32 MMA via coopMatMulAdd for both
 * formats. The weight scale is applied during the B-tile store to shared
 * memory: each int weight is unpacked (nibble - 8 for INT4; bitfieldExtract
 * for INT8), cast to fp16, and multiplied by its scale before it lands in
 * Bsh, keeping the K-loop a clean fp16 MMA.
 *
 * Loop structure follows the NVIDIA double-buffered GEMM reference
 * (shmem_double_buf4.comp, "store-first" variant; see coopmat_mm_ref.glsl
 * in test/custom_ops — measured 1.5x faster than the previous
 * single-buffered skeleton at fp16 on Xclipse 970):
 *   - PROLOGUE: prefetch tile 0 from global memory into temp registers, then
 *     store it to shared-memory slice 0 (no barrier).
 *   - Each iteration: barrier -> global prefetch of the NEXT tile into temp
 *     -> MMA math on the CURRENT slice -> store temp into the OTHER slice.
 *     One barrier per iteration; the prefetch loads are in flight during the
 *     math and are only consumed at the store stage.
 *   - Ping-pong shared-memory slices make the overlap safe.
 *
 * Each thread keeps its 8 weight scales (2 f16vec4) in registers. For INT4
 * they are reloaded from global only when the prefetched chunk crosses a
 * group boundary (a workgroup-uniform branch); for INT8 (per-channel = a
 * single group spanning all of K) they are loaded once in the prologue.
 * There is no scales staging in shared memory and no extra barrier.
 *
 * Tile hierarchy (yaml; mirrors the double-buffered reference):
 *   MMA_*         per-MMA-instruction shape (16x16x16 fp16)
 *   WG_TILE_*     output tile per workgroup (128x128)
 *   SG_GRID_*     subgroup grid inside workgroup (4x2 = 8 subgroups)
 *   SUBGROUP_SIZE 32, forced at pipeline creation via the
 *                 REQUIRED_SUBGROUP_SIZE annotation below
 *
 * Storage: activation/output forced to buffer; INT weight = texture2d or
 * buffer (yaml variant). DTYPE = half only.
 *
 * Hard preconditions (no shape/alignment checks inside the shader):
 *   M % WG_TILE_M == 0
 *   N % WG_TILE_N == 0
 *   K % WG_TILE_K == 0
 *   INT4: group_size % WG_TILE_K == 0  (each group = whole number of chunks)
 * Misaligned shapes silently miscompute / overrun — gate at dispatch time.
 */

// REQUIRED_SUBGROUP_SIZE = 32

#version 450 core

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : enable

#define PRECISION ${PRECISION}

$if WEIGHT_NBITS == 4:
  #define WEIGHT_INT4

$if HAS_BIAS:
  #define HAS_BIAS

$if WEIGHT_STORAGE == "buffer":
  #define WEIGHT_BUFFER

layout(std430) buffer;

#include "common.glslh"

// Bindings — match the order used by add_linear_qw_node so the dispatch
// site can reuse the same arg layout.
${layout_declare_tensor(B, "w", "t_output",         "half", "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_input",          "half", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_packed_weight",  "int",  WEIGHT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_scales",  "half", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias",           "half", "buffer", is_scalar_array=True)}

${layout_declare_ubo(B, "ivec4", "output_sizes")}
${layout_declare_ubo(B, "ivec4", "input_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "apply_bias",   "0")}
// INT4 only; inert (0) for INT8 so the dispatcher's spec list lines up.
${layout_declare_spec_const(C, "int", "K4_per_group", "0")}
// Trip-count source for the coopmat K loop, passed as a spec constant (not
// derived from the runtime sizes UBO): the Xclipse/AMD-PAL shader compiler
// crashes (null deref in vkCreateComputePipelines) when a loop containing
// coopMatMulAdd has a UBO-derived trip count. INT4: number of quant groups;
// INT8: number of K-chunks.
${layout_declare_spec_const(C, "int", "num_groups_arg", "0")}
// Output width N for coopMatStore, as a spec constant: the same compiler
// MISCOMPILES coopMatStore whose offset/stride derive from a UBO value (only
// the first store per subgroup lands correctly; standalone repro cm_acc2).
${layout_declare_spec_const(C, "int", "out_N_arg", "0")}

// --- Tile geometry (from yaml; defaults match coopmat_mm_ref) ---
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

// fp16: 8 elements per uvec4 (128-bit)
const uint FP16_PER_VEC4 = 8;
const uint A_STRIDE_VEC4 = (WG_TILE_K + FP16_PER_VEC4) / FP16_PER_VEC4;
const uint B_STRIDE_VEC4 = (WG_TILE_N + FP16_PER_VEC4) / FP16_PER_VEC4;

// One ping-pong slice of each shared-memory buffer (in uvec4 units).
const uint ASH_SLICE = WG_TILE_M * A_STRIDE_VEC4;
const uint BSH_SLICE = WG_TILE_K * B_STRIDE_VEC4;

// Double-buffered shared memory.
shared uvec4 Ash[2 * ASH_SLICE];
shared uvec4 Bsh[2 * BSH_SLICE];
#ifdef HAS_BIAS
shared float bias_sh[WG_TILE_N];
#endif

// Staging thread maps: each thread covers one uvec4 (8 fp16) per pass.
const uint INVS_PER_ROW_A = WG_TILE_K / FP16_PER_VEC4;
const uint A_ROWS_PER_PASS = WG_SIZE / INVS_PER_ROW_A;
const uint A_PASSES = WG_TILE_M / A_ROWS_PER_PASS;
const uint INVS_PER_ROW_B = WG_TILE_N / FP16_PER_VEC4;
const uint B_ROWS_PER_PASS = WG_SIZE / INVS_PER_ROW_B;
const uint B_PASSES = WG_TILE_K / B_ROWS_PER_PASS;

// Fp32 accumulator coopmats (MMAS_PER_SG_M x MMAS_PER_SG_N per thread)
coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>
    result[MMAS_PER_SG_M][MMAS_PER_SG_N];

#ifdef WEIGHT_INT4

// Dequant one packed INT4 block column-pair into 8 scaled fp16 weights
// (one Bsh uvec4). col_lo/col_hi select the K row within the block.
uvec4 dequant_block(
    const ivec4 wb,
    const uint col_lo,
    const uint col_hi,
    const f16vec4 s0,
    const f16vec4 s1) {
  f16vec4 v0;
  v0.x = float16_t(int(((wb[0] >> (4 * col_lo)) & 0xF)) - 8) * s0.x;
  v0.y = float16_t(int(((wb[1] >> (4 * col_lo)) & 0xF)) - 8) * s0.y;
  v0.z = float16_t(int(((wb[2] >> (4 * col_lo)) & 0xF)) - 8) * s0.z;
  v0.w = float16_t(int(((wb[3] >> (4 * col_lo)) & 0xF)) - 8) * s0.w;
  f16vec4 v1;
  v1.x = float16_t(int(((wb[0] >> (4 * col_hi)) & 0xF)) - 8) * s1.x;
  v1.y = float16_t(int(((wb[1] >> (4 * col_hi)) & 0xF)) - 8) * s1.y;
  v1.z = float16_t(int(((wb[2] >> (4 * col_hi)) & 0xF)) - 8) * s1.z;
  v1.w = float16_t(int(((wb[3] >> (4 * col_hi)) & 0xF)) - 8) * s1.w;
  return uvec4(
      packFloat2x16(v0.xy), packFloat2x16(v0.zw),
      packFloat2x16(v1.xy), packFloat2x16(v1.zw));
}

#else // INT8

// Dequant 8 int8 weights (two ivec4 blocks, one K-row selected by shift)
// into 8 scaled fp16 weights (one Bsh uvec4).
uvec4 dequant_block(
    const ivec4 wa,
    const ivec4 wb,
    const int shift,
    const f16vec4 s0,
    const f16vec4 s1) {
  f16vec4 v0;
  v0.x = float16_t(bitfieldExtract(wa.x, shift, 8)) * s0.x;
  v0.y = float16_t(bitfieldExtract(wa.y, shift, 8)) * s0.y;
  v0.z = float16_t(bitfieldExtract(wa.z, shift, 8)) * s0.z;
  v0.w = float16_t(bitfieldExtract(wa.w, shift, 8)) * s0.w;
  f16vec4 v1;
  v1.x = float16_t(bitfieldExtract(wb.x, shift, 8)) * s1.x;
  v1.y = float16_t(bitfieldExtract(wb.y, shift, 8)) * s1.y;
  v1.z = float16_t(bitfieldExtract(wb.z, shift, 8)) * s1.z;
  v1.w = float16_t(bitfieldExtract(wb.w, shift, 8)) * s1.w;
  return uvec4(
      packFloat2x16(v0.xy), packFloat2x16(v0.zw),
      packFloat2x16(v1.xy), packFloat2x16(v1.zw));
}

#endif // WEIGHT_INT4

void main() {
  const uvec2 tileID = uvec2(gl_WorkGroupID.xy);
  const uvec2 warpInTile = uvec2(
      gl_SubgroupID % SG_GRID_X,
      gl_SubgroupID / SG_GRID_X);

  const uint K = uint(input_sizes.x);
  const uint K4 = (K + 3u) / 4u;
  const uint N4 = (uint(output_sizes.x) + 3u) / 4u;

#ifdef WEIGHT_INT4
  const uint CHUNKS_PER_GROUP = uint(K4_per_group) * 4u / WG_TILE_K;
  const uint num_chunks = uint(num_groups_arg) * CHUNKS_PER_GROUP;
#else
  const uint num_chunks = uint(num_groups_arg);
#endif

  const uint tile_m_start = WG_TILE_M * tileID.y;
  const uint tile_n_start = WG_TILE_N * tileID.x;

  // Initialize fp32 accumulators to zero.
  [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
    [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
      result[i][j] = coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(0.0);
    }
  }

  const uint a_col = gl_LocalInvocationID.x % INVS_PER_ROW_A;
  const uint a_row_offset = gl_LocalInvocationID.x / INVS_PER_ROW_A;
  const uint b_col = gl_LocalInvocationID.x % INVS_PER_ROW_B;
  const uint b_row_offset = gl_LocalInvocationID.x / INVS_PER_ROW_B;

#ifdef WEIGHT_INT4
  // INT4 weight block grid (see pack_q4_linear_weight.glsl): block (k4, n8)
  // covers K=[k4*4, k4*4+3] x N=[n8*8, n8*8+7]; buffer pitch = K4 blocks per
  // n8 row, texture coord = ivec2(x=k4, y=n8). This thread's 8 N-values at
  // any K-row live in column n8_blk of the block grid:
  const uint n8_blk = (tile_n_start + b_col * 8u) >> 3u;

  // The K row within a block depends only on (b_row_offset & 3): chunkK and
  // the pass offset are both multiples of 4.
  const uint col_lo = 2u * (b_row_offset & 3u);
  const uint col_hi = col_lo + 1u;

  // Per-thread per-group weight scales (8 consecutive N), kept in registers
  // and reloaded only when the prefetched chunk crosses a group boundary.
  const uint sc_n4 = (tile_n_start + b_col * 8u) >> 2u;
  uint cached_group = 0xFFFFFFFFu;
  f16vec4 sc0;
  f16vec4 sc1;

  // Temp registers holding the prefetched (next) tile.
  uvec4 temp_A[A_PASSES];
  ivec4 temp_B[B_PASSES]; // raw packed INT4 blocks; dequant at the store stage
#else
  // INT8 weight block layout: t_packed_weight[k4 * N4 + n4] = ivec4 whose
  // component n_in_blk packs 4 K-bytes (K of block k4) for N-col
  // (n4*4 + n_in_blk). This thread's 8 N-values span two adjacent n4 blocks:
  const uint n4_a = (tile_n_start + b_col * 8u) >> 2u; // n_start mult of 8 -> even

  // The byte within a packed uint depends only on (b_row_offset & 3): chunkK
  // and the pass offset are both multiples of 4.
  const int b_shift = int(8u * (b_row_offset & 3u));

  // Per-thread per-channel weight scales (8 consecutive N), cached ONCE.
  f16vec4 sc0 = t_weight_scales[n4_a];
  f16vec4 sc1 = t_weight_scales[n4_a + 1u];

  // Temp registers holding the prefetched (next) tile.
  uvec4 temp_A[A_PASSES];
  ivec4 temp_Ba[B_PASSES]; // raw packed INT8 blocks; dequant at the store stage
  ivec4 temp_Bb[B_PASSES];
#endif

  // =========================================================
  // PROLOGUE: prefetch chunk 0 into temp registers, then store to slice 0.
  // =========================================================
  {
    [[unroll]] for (uint p = 0; p < A_PASSES; ++p) {
      const uint row = tile_m_start + p * A_ROWS_PER_PASS + a_row_offset;
      const uint k_hv4 = (a_col * FP16_PER_VEC4) / 4u;
      f16vec4 v0 = t_input[row * K4 + k_hv4];
      f16vec4 v1 = t_input[row * K4 + k_hv4 + 1u];
      temp_A[p] = uvec4(
          packFloat2x16(v0.xy), packFloat2x16(v0.zw),
          packFloat2x16(v1.xy), packFloat2x16(v1.zw));
    }
#ifdef WEIGHT_INT4
    [[unroll]] for (uint p = 0; p < B_PASSES; ++p) {
      const uint k_row = p * B_ROWS_PER_PASS + b_row_offset;
#ifdef WEIGHT_BUFFER
      temp_B[p] = t_packed_weight[n8_blk * K4 + (k_row >> 2u)];
#else
      temp_B[p] = texelFetch(t_packed_weight, ivec2(k_row >> 2u, n8_blk), 0);
#endif
    }
    cached_group = 0u;
    sc0 = t_weight_scales[sc_n4];
    sc1 = t_weight_scales[sc_n4 + 1u];
#else
    [[unroll]] for (uint p = 0; p < B_PASSES; ++p) {
      const uint k4 = (p * B_ROWS_PER_PASS + b_row_offset) >> 2u;
#ifdef WEIGHT_BUFFER
      temp_Ba[p] = t_packed_weight[k4 * N4 + n4_a];
      temp_Bb[p] = t_packed_weight[k4 * N4 + n4_a + 1u];
#else
      temp_Ba[p] = texelFetch(t_packed_weight, ivec2(n4_a, k4), 0);
      temp_Bb[p] = texelFetch(t_packed_weight, ivec2(n4_a + 1u, k4), 0);
#endif
    }
#endif
  }
  {
    [[unroll]] for (uint p = 0; p < A_PASSES; ++p) {
      Ash[(p * A_ROWS_PER_PASS + a_row_offset) * A_STRIDE_VEC4 + a_col] = temp_A[p];
    }
    [[unroll]] for (uint p = 0; p < B_PASSES; ++p) {
#ifdef WEIGHT_INT4
      Bsh[(p * B_ROWS_PER_PASS + b_row_offset) * B_STRIDE_VEC4 + b_col] =
          dequant_block(temp_B[p], col_lo, col_hi, sc0, sc1);
#else
      Bsh[(p * B_ROWS_PER_PASS + b_row_offset) * B_STRIDE_VEC4 + b_col] =
          dequant_block(temp_Ba[p], temp_Bb[p], b_shift, sc0, sc1);
#endif
    }
  }

  // =========================================================
  // MAIN LOOP — one barrier per iteration. Iteration `chunk` does:
  //   1. barrier      — slice (chunk%2) fully written
  //   2. prefetch     — chunk+1 from global into temp (in flight during math)
  //   3. MMA math     — on slice (chunk%2)
  //   4. store        — temp (chunk+1, dequantized) into slice ((chunk+1)%2)
  // =========================================================
  uint chunk;
  for (chunk = 0; chunk + 1u < num_chunks; ++chunk) {
    const uint cur_base_A = (chunk % 2u) * ASH_SLICE;
    const uint cur_base_B = (chunk % 2u) * BSH_SLICE;
    const uint nxt_base_A = ((chunk + 1u) % 2u) * ASH_SLICE;
    const uint nxt_base_B = ((chunk + 1u) % 2u) * BSH_SLICE;

    barrier();

    // --- prefetch chunk+1 -> temp ---
    {
      const uint chunkK_nxt = (chunk + 1u) * WG_TILE_K;

      [[unroll]] for (uint p = 0; p < A_PASSES; ++p) {
        const uint row = tile_m_start + p * A_ROWS_PER_PASS + a_row_offset;
        const uint k_hv4 = (chunkK_nxt + a_col * FP16_PER_VEC4) / 4u;
        f16vec4 v0 = t_input[row * K4 + k_hv4];
        f16vec4 v1 = t_input[row * K4 + k_hv4 + 1u];
        temp_A[p] = uvec4(
            packFloat2x16(v0.xy), packFloat2x16(v0.zw),
            packFloat2x16(v1.xy), packFloat2x16(v1.zw));
      }
#ifdef WEIGHT_INT4
      [[unroll]] for (uint p = 0; p < B_PASSES; ++p) {
        const uint k_row = chunkK_nxt + p * B_ROWS_PER_PASS + b_row_offset;
#ifdef WEIGHT_BUFFER
        temp_B[p] = t_packed_weight[n8_blk * K4 + (k_row >> 2u)];
#else
        temp_B[p] = texelFetch(t_packed_weight, ivec2(k_row >> 2u, n8_blk), 0);
#endif
      }
      const uint group_nxt = (chunk + 1u) / CHUNKS_PER_GROUP;
      if (group_nxt != cached_group) {
        cached_group = group_nxt;
        sc0 = t_weight_scales[group_nxt * N4 + sc_n4];
        sc1 = t_weight_scales[group_nxt * N4 + sc_n4 + 1u];
      }
#else
      [[unroll]] for (uint p = 0; p < B_PASSES; ++p) {
        const uint k4 = (chunkK_nxt + p * B_ROWS_PER_PASS + b_row_offset) >> 2u;
#ifdef WEIGHT_BUFFER
        temp_Ba[p] = t_packed_weight[k4 * N4 + n4_a];
        temp_Bb[p] = t_packed_weight[k4 * N4 + n4_a + 1u];
#else
        temp_Ba[p] = texelFetch(t_packed_weight, ivec2(n4_a, k4), 0);
        temp_Bb[p] = texelFetch(t_packed_weight, ivec2(n4_a + 1u, k4), 0);
#endif
      }
#endif
    }

    // --- MMA math on the cur slice ---
    [[unroll]] for (uint k = 0; k < WG_TILE_K / MMA_K; ++k) {
      const uint k_start = MMA_K * k;

      coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_K, gl_MatrixUseA> matA[MMAS_PER_SG_M];
      [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
        const uint row_a = MMA_M * (MMAS_PER_SG_M * warpInTile.y + i);
        coopMatLoad(
            matA[i], Ash,
            cur_base_A + row_a * A_STRIDE_VEC4 + k_start / FP16_PER_VEC4,
            A_STRIDE_VEC4,
            gl_CooperativeMatrixLayoutRowMajor);
      }

      coopmat<float16_t, gl_ScopeSubgroup, MMA_K, MMA_N, gl_MatrixUseB> matB;
      [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
        const uint col_b = MMA_N * (MMAS_PER_SG_N * warpInTile.x + j) / FP16_PER_VEC4;
        coopMatLoad(
            matB, Bsh,
            cur_base_B + k_start * B_STRIDE_VEC4 + col_b,
            B_STRIDE_VEC4,
            gl_CooperativeMatrixLayoutRowMajor);

        [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
          result[i][j] = coopMatMulAdd(matA[i], matB, result[i][j]);
        }
      }
    }

    // --- store temp (chunk+1) -> nxt slice, dequantizing B ---
    {
      [[unroll]] for (uint p = 0; p < A_PASSES; ++p) {
        Ash[nxt_base_A + (p * A_ROWS_PER_PASS + a_row_offset) * A_STRIDE_VEC4 + a_col] =
            temp_A[p];
      }
      [[unroll]] for (uint p = 0; p < B_PASSES; ++p) {
#ifdef WEIGHT_INT4
        Bsh[nxt_base_B + (p * B_ROWS_PER_PASS + b_row_offset) * B_STRIDE_VEC4 + b_col] =
            dequant_block(temp_B[p], col_lo, col_hi, sc0, sc1);
#else
        Bsh[nxt_base_B + (p * B_ROWS_PER_PASS + b_row_offset) * B_STRIDE_VEC4 + b_col] =
            dequant_block(temp_Ba[p], temp_Bb[p], b_shift, sc0, sc1);
#endif
      }
    }
  }

  // --- exit from MAIN LOOP: math on the last chunk ---
  {
    const uint cur_base_A = (chunk % 2u) * ASH_SLICE;
    const uint cur_base_B = (chunk % 2u) * BSH_SLICE;

    barrier();

    [[unroll]] for (uint k = 0; k < WG_TILE_K / MMA_K; ++k) {
      const uint k_start = MMA_K * k;

      coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_K, gl_MatrixUseA> matA[MMAS_PER_SG_M];
      [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
        const uint row_a = MMA_M * (MMAS_PER_SG_M * warpInTile.y + i);
        coopMatLoad(
            matA[i], Ash,
            cur_base_A + row_a * A_STRIDE_VEC4 + k_start / FP16_PER_VEC4,
            A_STRIDE_VEC4,
            gl_CooperativeMatrixLayoutRowMajor);
      }

      coopmat<float16_t, gl_ScopeSubgroup, MMA_K, MMA_N, gl_MatrixUseB> matB;
      [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
        const uint col_b = MMA_N * (MMAS_PER_SG_N * warpInTile.x + j) / FP16_PER_VEC4;
        coopMatLoad(
            matB, Bsh,
            cur_base_B + k_start * B_STRIDE_VEC4 + col_b,
            B_STRIDE_VEC4,
            gl_CooperativeMatrixLayoutRowMajor);

        [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
          result[i][j] = coopMatMulAdd(matA[i], matB, result[i][j]);
        }
      }
    }
  }

  // --- Bias staging (if any) ---
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
        coopMatLoad(
            bias_tile, bias_sh,
            local_n, /*stride=*/0u,
            gl_CooperativeMatrixLayoutRowMajor);
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
