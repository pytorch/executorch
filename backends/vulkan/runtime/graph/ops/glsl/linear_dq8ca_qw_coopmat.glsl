/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * KHR Cooperative Matrix variants of the dynamically-quantized-activation
 * linear tiled shaders. One template, two weight formats (WEIGHT_NBITS):
 *   4  ->  linear_dq8ca_q4gsw_coopmat   INT4 group-symmetric weight
 *   8  ->  linear_dq8ca_q8csw_coopmat   INT8 per-channel symmetric weight
 *
 * Performs: out[M,N] = dequant(int8_act) * dequant(int_w) (+ bias)
 * via coopmat<int8> x coopmat<int8> -> coopmat<int32> on the matrix unit.
 *
 * Math (per group; per-channel INT8 is the num_groups == 1 special case
 * where the single "group" spans all of K):
 *   accum_int32 = sum_k(int8_in_k * int_w_signed_k)       // coopMatMulAdd
 *   adjusted    = accum_int32 - input_zp[m] * wsum_signed[group, n]
 *   delta_fp    = float(adjusted) * (input_scale[m] * weight_scale[group, n])
 *   result_fp  += delta_fp                                // across groups
 *
 * Because INT4 weights are sign-extended to int8 in the B-stage, the
 * "8 * input_sum" term of the tiled correction (which compensates for
 * unsigned int4 nibbles in dotPacked4x8) cancels out and is not needed.
 *
 * Loop structure follows the NVIDIA double-buffered GEMM reference
 * (shmem_double_buf4.comp "store-first" variant; see gemm_double_buf.glsl in
 * test/custom_ops): prologue register prefetch, then per chunk
 * barrier -> prefetch next chunk -> int8 MMA on the current LDS slice ->
 * store temp into the other slice. One barrier per chunk; the prefetch is
 * pure loads, in flight during the math; quant unpack happens at the store
 * stage. The loop stays NESTED (groups x chunks, group epilog unconditional
 * at the group tail) — flattening it with a conditional coopmat epilog
 * crashes the Xclipse PAL compiler at large spec-resolved trip counts.
 *
 * Per-(group, N) weight sums/scales live in a SECOND ping-pong pair indexed
 * by group parity: the next group's values are prefetched into registers
 * and stored to the other wsum/wsc slice during the iteration that crosses
 * the group boundary, and the regular per-iteration barrier makes them
 * visible before that group's epilog runs. Per-row activation zp/scale
 * broadcasts are group-invariant and loaded once in the prologue.
 *
 * LDS layout for the MMA operands: K-slab split + ColumnMajor B + per-col
 * skew padding: the int8 WMMA matB lane layout wants 4 K-contiguous bytes
 * per lane, so a RowMajor B in LDS forces per-byte ds_load + v_perm repack
 * chains. ColumnMajor with a +1-uint skew per column gives one ds_load_b32
 * per lane with a bank-conflict-free col stride. Each uint holds 4 packed
 * int8.
 *
 * Tile hierarchy (yaml): MMA 16x16x16 int8, WG_TILE 128x64, WG_TILE_K = 32,
 * 4 subgroups x 64 threads. The double-buffered reference's subgroup-32
 * layout is NOT used: the Xclipse PAL compiler crashes in
 * vkCreateComputePipelines when int8 WMMA is compiled at forced subgroup
 * size 32 (fp16 WMMA at 32 is fine; see linear_qw_coopmat).
 *
 * Hard preconditions:
 *   M % WG_TILE_M == 0, N % WG_TILE_N == 0, K % WG_TILE_K == 0,
 *   INT4: group_size % WG_TILE_K == 0,
 *   device exposes coopmat<int8>x<int8>-><int32> at 16x16x16.
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

$if WEIGHT_NBITS == 4:
  #define WEIGHT_INT4

$if HAS_BIAS:
  #define HAS_BIAS

$if WEIGHT_STORAGE == "buffer":
  #define WEIGHT_BUFFER

layout(std430) buffer;

#include "common.glslh"

// Bindings — match add_linear_dqa_qw_node arg order:
//   output(0), fp_input(1), packed_int8_input(2), int_input_sums(3 - unused),
//   input_scales(4), input_zps(5), packed_weight(6), weight_sums(7),
//   weight_scales(8), bias(9).
${layout_declare_tensor(B, "w", "t_output",              "half", "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_input",               "half", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_packed_int8_input",   "int",  "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_int8_input_sums",     "int",  "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_int8_input_scales",   "half", "texture3d")}
${layout_declare_tensor(B, "r", "t_int8_input_zps",      "int8", "texture3d")}
${layout_declare_tensor(B, "r", "t_packed_weight",       "int",  WEIGHT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_sums",         "int",  "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_weight_scales",       "half", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias",                "half", "buffer", is_scalar_array=True)}

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

const uint A_SLAB_INT8     = WG_TILE_M * MMA_K;
const uint B_USEFUL_U32    = MMA_K / 4u;
const uint B_STRIDE_U32    = B_USEFUL_U32 + 1u; // +1 skew
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

// Per-WG-tile-row activation params (loaded ONCE at WG start; constant
// across groups).
shared int   izp_sh[WG_TILE_M];   // int32 (cast from int8 source) for broadcast
shared float ifs_sh[WG_TILE_M];   // float32 (cast from fp16 source) for broadcast

// Per-(group, output-channel) weight params, ping-ponged by group parity.
// (For per-channel INT8 only slice 0 is ever used.)
shared int   wsum_sh[2u * WG_TILE_N];
shared float wsc_sh[2u * WG_TILE_N];

#ifdef HAS_BIAS
shared float bias_sh[WG_TILE_N];
#endif

// Running fp32 accumulator (across all groups).
coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>
    result[MMAS_PER_SG_M][MMAS_PER_SG_N];

// Per-group int32 MMA accumulator.
coopmat<int32_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>
    accum_int32[MMAS_PER_SG_M][MMAS_PER_SG_N];

void main() {
  const uvec2 tileID = uvec2(gl_WorkGroupID.xy);
  const uvec2 warpInTile = uvec2(
      gl_SubgroupID % SG_GRID_X,
      gl_SubgroupID / SG_GRID_X);

  const uint K = uint(input_sizes.x);
  const uint N = uint(output_sizes.x);
  const uint N4 = (N + 3u) / 4u;
  const uint nblocks_x_A = (K + 3u) >> 2u;

#ifdef WEIGHT_INT4
  const uint num_groups = uint(num_groups_arg);
  const uint CHUNKS_PER_GROUP = uint(K4_per_group) * 4u / WG_TILE_K;
#else
  // Per-channel: a single quant "group" spanning all of K. The nested
  // groups x chunks loop below collapses to a flat chunk loop, the wsum/wsc
  // ping-pong never crosses a boundary, and the epilog runs exactly once.
  const uint num_groups = 1u;
  const uint CHUNKS_PER_GROUP = uint(num_groups_arg);
#endif
  const uint num_chunks = num_groups * CHUNKS_PER_GROUP;

  const uint tile_m_start = WG_TILE_M * tileID.y;
  const uint tile_n_start = WG_TILE_N * tileID.x;

  [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
    [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
      result[i][j] = coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(0.0);
      accum_int32[i][j] = coopmat<int32_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(0);
    }
  }

  // --- A staging thread map: one (m4, k4) ivec4 block per active thread ---
  // (4 M-rows x 4 K-positions; each block expands to 4 slab-major LDS uints.)
  const uint K_BLOCKS_PER_CHUNK = WG_TILE_K >> 2u;
  const uint A_ACTIVE_THREADS = (WG_TILE_M >> 2u) * K_BLOCKS_PER_CHUNK;
  const uint a_m_block = gl_LocalInvocationID.x / K_BLOCKS_PER_CHUNK;
  const uint a_k_block = gl_LocalInvocationID.x % K_BLOCKS_PER_CHUNK;
  const bool a_active = gl_LocalInvocationID.x < A_ACTIVE_THREADS;

#ifdef WEIGHT_INT4
  // --- B staging thread map: (block, col) slots; each slot extracts one
  //     ColumnMajor LDS uint (4 K-contiguous sign-extended int8) ---
  // INT4 weight block grid (see pack_q4_linear_weight.glsl): block (k4, n8)
  // covers K=[k4*4, k4*4+3] x N=[n8*8, n8*8+7]. Within a block, int32[r]
  // nibble col c maps to N = n8*8 + r + (c&1 ? 4 : 0), K = k4*4 + c/2 — one
  // (component, parity) pair yields exactly the 4 K-contiguous bytes of one
  // N column = one ColumnMajor LDS uint.
  const uint B_TOTAL_SLOTS = K_BLOCKS_PER_CHUNK * WG_TILE_N;
  const uint B_SLOTS_PER_THREAD = B_TOTAL_SLOTS / WG_SIZE;
  const uint N8_PER_TILE = WG_TILE_N >> 3u;
#else
  // --- B staging thread map: one (k4, n4) ivec4 block per active thread ---
  // INT8 weight block layout: wblk[n_in_blk] packs 4 K-contiguous bytes for
  // N-col (n4*4 + n_in_blk) — exactly one ColumnMajor LDS uint, written
  // as-is (no byte repack).
  const uint B_FETCH_SLOTS = K_BLOCKS_PER_CHUNK * (WG_TILE_N >> 2u);
  const uint N4_PER_TILE = WG_TILE_N >> 2u;
  const uint b_k4_in_chunk = gl_LocalInvocationID.x / N4_PER_TILE;
  const uint b_n_uint_col = gl_LocalInvocationID.x % N4_PER_TILE;
  const bool b_active = gl_LocalInvocationID.x < B_FETCH_SLOTS;
#endif

  // Prefetch temp registers.
  ivec4 temp_A;
#ifdef WEIGHT_INT4
  ivec4 temp_B[B_SLOTS_PER_THREAD];
  int   temp_wsum;
  float temp_wsc;
#else
  ivec4 temp_B;
#endif

  // =========================================================
  // PROLOGUE
  // =========================================================
  // One-time: per-row input zp + scale (texture3d, one m4-block of 4 rows per
  // texel) — constant across K groups.
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
  // Group 0 weight sums/scales -> slice 0.
  if (gl_LocalInvocationID.x < WG_TILE_N) {
    const uint n_idx = tile_n_start + gl_LocalInvocationID.x;
    f16vec4 sv = t_weight_scales[n_idx >> 2u];
    wsc_sh[gl_LocalInvocationID.x] = float(sv[n_idx & 3u]);
    wsum_sh[gl_LocalInvocationID.x] = t_weight_sums[n_idx];
  }
  memoryBarrierShared();
  barrier();

  // izp/ifs are per-row activation params, constant across K groups —
  // broadcast them into coopmats ONCE; the group epilog reuses them every
  // group (they depend only on the row block i, not on the group or j).
  coopmat<int32_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>
      izp_bcast[MMAS_PER_SG_M];
  coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>
      ifs_bcast[MMAS_PER_SG_M];
  [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
    const uint local_m_base = MMA_M * (MMAS_PER_SG_M * warpInTile.y + i);
    coopMatLoad(
        izp_bcast[i], izp_sh,
        local_m_base, /*stride=*/0u,
        gl_CooperativeMatrixLayoutColumnMajor);
    coopMatLoad(
        ifs_bcast[i], ifs_sh,
        local_m_base, /*stride=*/0u,
        gl_CooperativeMatrixLayoutColumnMajor);
  }

  // Prefetch chunk 0 into temp registers, then store to slice 0 (no barrier;
  // the first loop iteration's barrier publishes it).
  if (a_active) {
    const uint m4_global = (tile_m_start >> 2u) + a_m_block;
    temp_A = t_packed_int8_input[m4_global * nblocks_x_A + a_k_block];
  }
#ifdef WEIGHT_INT4
  [[unroll]] for (uint si = 0; si < B_SLOTS_PER_THREAD; ++si) {
    const uint slot = gl_LocalInvocationID.x + si * WG_SIZE;
    const uint block_in_chunk = slot >> 3u;
    const uint k4_blk = block_in_chunk / N8_PER_TILE;
    const uint n8_blk = (tile_n_start >> 3u) + (block_in_chunk % N8_PER_TILE);
#ifdef WEIGHT_BUFFER
    temp_B[si] = t_packed_weight[(n8_blk * nblocks_x_A) + k4_blk];
#else
    temp_B[si] = texelFetch(t_packed_weight, ivec2(k4_blk, n8_blk), 0);
#endif
  }
#else
  if (b_active) {
    const uint block_x_w = (tile_n_start >> 2u) + b_n_uint_col;
#ifdef WEIGHT_BUFFER
    temp_B = t_packed_weight[(b_k4_in_chunk * N4) + block_x_w];
#else
    temp_B = texelFetch(t_packed_weight, ivec2(block_x_w, b_k4_in_chunk), 0);
#endif
  }
#endif
  {
    // store chunk 0 -> slice 0
    if (a_active) {
      const uint slab_idx       = a_k_block / (MMA_K >> 2u);
      const uint k_uint_in_slab = a_k_block % (MMA_K >> 2u);
      const uint base_row = a_m_block * 4u;
      [[unroll]] for (uint m4i = 0; m4i < 4u; ++m4i) {
        Ash_int8[slab_idx * A_SLAB_U32 + (base_row + m4i) * A_STRIDE_U32 + k_uint_in_slab] =
            uint(temp_A[m4i]);
      }
    }
#ifdef WEIGHT_INT4
    [[unroll]] for (uint si = 0; si < B_SLOTS_PER_THREAD; ++si) {
      const uint slot = gl_LocalInvocationID.x + si * WG_SIZE;
      const uint block_in_chunk = slot >> 3u;
      const uint col_in_block   = slot & 7u;
      const uint k4_in_chunk    = block_in_chunk / N8_PER_TILE;
      const uint n8_in_tile     = block_in_chunk % N8_PER_TILE;
      const uint r      = col_in_block & 3u;
      const uint parity = col_in_block >> 2u;
      const int  w      = temp_B[si][r];
      const int  base   = int(4u * parity);
      const int v0 = (((w >> (base + 0))  & 0xF) - 8) & 0xFF;
      const int v1 = (((w >> (base + 8))  & 0xF) - 8) & 0xFF;
      const int v2 = (((w >> (base + 16)) & 0xF) - 8) & 0xFF;
      const int v3 = (((w >> (base + 24)) & 0xF) - 8) & 0xFF;
      const uint n_col      = n8_in_tile * 8u + r + parity * 4u;
      const uint slab_idx   = k4_in_chunk / (MMA_K >> 2u);
      const uint k4_in_slab = k4_in_chunk % (MMA_K >> 2u);
      Bsh_int8[slab_idx * B_SLAB_U32 + n_col * B_STRIDE_U32 + k4_in_slab] =
          uint(v0 | (v1 << 8) | (v2 << 16) | (v3 << 24));
    }
#else
    if (b_active) {
      const uint slab_idx   = b_k4_in_chunk / (MMA_K >> 2u);
      const uint k4_in_slab = b_k4_in_chunk % (MMA_K >> 2u);
      const uint n_col_base = b_n_uint_col * 4u;
      [[unroll]] for (uint n_in_blk = 0u; n_in_blk < 4u; ++n_in_blk) {
        Bsh_int8[slab_idx * B_SLAB_U32 + (n_col_base + n_in_blk) * B_STRIDE_U32 + k4_in_slab] =
            uint(temp_B[n_in_blk]);
      }
    }
#endif
  }

  // =========================================================
  // MAIN LOOP — nested groups x chunks (the flattened single loop with a
  // conditional coopmat epilog crashes the Xclipse PAL compiler at large
  // spec-resolved trip counts). One barrier per chunk. Chunk iteration
  // (global index `chunk`):
  //   1. barrier   — A/B slice (chunk%2) fully written; on the first chunk
  //                  of group g, wsum/wsc slice (g%2) is too.
  //   2. prefetch  — chunk+1 (A blocks, B blocks) into temp; when chunk+1
  //                  starts a new group, also its wsum/wsc element. Skipped
  //                  entirely on the final chunk.
  //   3. int8 MMA  — on slice (chunk%2) into accum_int32.
  //   4. store     — temp -> A/B slice ((chunk+1)%2), unpacking the weight;
  //                  on a group boundary, wsum/wsc -> slice ((g+1)%2).
  // The group epilog runs unconditionally at the tail of each group.
  // =========================================================
  uint chunk = 0;
  for (uint group_i = 0; group_i < num_groups; ++group_i) {
    for (uint inner = 0; inner < CHUNKS_PER_GROUP; ++inner, ++chunk) {
      const bool has_next = chunk + 1u < num_chunks;
      const bool group_crossing = has_next && (inner + 1u == CHUNKS_PER_GROUP);
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
#ifdef WEIGHT_INT4
        [[unroll]] for (uint si = 0; si < B_SLOTS_PER_THREAD; ++si) {
          const uint slot = gl_LocalInvocationID.x + si * WG_SIZE;
          const uint block_in_chunk = slot >> 3u;
          const uint k4_blk = (chunkK_nxt >> 2u) + block_in_chunk / N8_PER_TILE;
          const uint n8_blk = (tile_n_start >> 3u) + (block_in_chunk % N8_PER_TILE);
#ifdef WEIGHT_BUFFER
          temp_B[si] = t_packed_weight[(n8_blk * nblocks_x_A) + k4_blk];
#else
          temp_B[si] = texelFetch(t_packed_weight, ivec2(k4_blk, n8_blk), 0);
#endif
        }
        if (group_crossing && gl_LocalInvocationID.x < WG_TILE_N) {
          const uint n_idx = tile_n_start + gl_LocalInvocationID.x;
          f16vec4 sv = t_weight_scales[(group_i + 1u) * N4 + (n_idx >> 2u)];
          temp_wsc = float(sv[n_idx & 3u]);
          temp_wsum = t_weight_sums[(group_i + 1u) * N + n_idx];
        }
#else
        if (b_active) {
          const uint block_y_w = (chunkK_nxt >> 2u) + b_k4_in_chunk;
          const uint block_x_w = (tile_n_start >> 2u) + b_n_uint_col;
#ifdef WEIGHT_BUFFER
          temp_B = t_packed_weight[(block_y_w * N4) + block_x_w];
#else
          temp_B = texelFetch(t_packed_weight, ivec2(block_x_w, block_y_w), 0);
#endif
        }
#endif
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
#ifdef WEIGHT_INT4
        [[unroll]] for (uint si = 0; si < B_SLOTS_PER_THREAD; ++si) {
          const uint slot = gl_LocalInvocationID.x + si * WG_SIZE;
          const uint block_in_chunk = slot >> 3u;
          const uint col_in_block   = slot & 7u;
          const uint k4_in_chunk    = block_in_chunk / N8_PER_TILE;
          const uint n8_in_tile     = block_in_chunk % N8_PER_TILE;
          const uint r      = col_in_block & 3u;
          const uint parity = col_in_block >> 2u;
          const int  w      = temp_B[si][r];
          const int  base   = int(4u * parity);
          const int v0 = (((w >> (base + 0))  & 0xF) - 8) & 0xFF;
          const int v1 = (((w >> (base + 8))  & 0xF) - 8) & 0xFF;
          const int v2 = (((w >> (base + 16)) & 0xF) - 8) & 0xFF;
          const int v3 = (((w >> (base + 24)) & 0xF) - 8) & 0xFF;
          const uint n_col      = n8_in_tile * 8u + r + parity * 4u;
          const uint slab_idx   = k4_in_chunk / (MMA_K >> 2u);
          const uint k4_in_slab = k4_in_chunk % (MMA_K >> 2u);
          Bsh_int8[nxt_b + slab_idx * B_SLAB_U32 + n_col * B_STRIDE_U32 + k4_in_slab] =
              uint(v0 | (v1 << 8) | (v2 << 16) | (v3 << 24));
        }
        if (group_crossing && gl_LocalInvocationID.x < WG_TILE_N) {
          const uint wbase_nxt = ((group_i + 1u) % 2u) * WG_TILE_N;
          wsum_sh[wbase_nxt + gl_LocalInvocationID.x] = temp_wsum;
          wsc_sh[wbase_nxt + gl_LocalInvocationID.x] = temp_wsc;
        }
#else
        if (b_active) {
          const uint slab_idx   = b_k4_in_chunk / (MMA_K >> 2u);
          const uint k4_in_slab = b_k4_in_chunk % (MMA_K >> 2u);
          const uint n_col_base = b_n_uint_col * 4u;
          [[unroll]] for (uint n_in_blk = 0u; n_in_blk < 4u; ++n_in_blk) {
            Bsh_int8[nxt_b + slab_idx * B_SLAB_U32 + (n_col_base + n_in_blk) * B_STRIDE_U32 + k4_in_slab] =
                uint(temp_B[n_in_blk]);
          }
        }
#endif
      }
    }  // chunks

    // --- Group epilog: dequant accum_int32 -> result, reset accum ---
    {
      const uint wbase = (group_i % 2u) * WG_TILE_N;
      [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
        const uint local_n_base = MMA_N * (MMAS_PER_SG_N * warpInTile.x + j);

        coopmat<int32_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> wsum_bcast;
        coopMatLoad(
            wsum_bcast, wsum_sh,
            wbase + local_n_base, /*stride=*/0u,
            gl_CooperativeMatrixLayoutRowMajor);

        coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> wsc_bcast;
        coopMatLoad(
            wsc_bcast, wsc_sh,
            wbase + local_n_base, /*stride=*/0u,
            gl_CooperativeMatrixLayoutRowMajor);

        [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
          coopmat<int32_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> adjusted =
              accum_int32[i][j] - izp_bcast[i] * wsum_bcast;
          coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> adjusted_fp =
              coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(adjusted);
          coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> scales_outer =
              ifs_bcast[i] * wsc_bcast;
          result[i][j] += adjusted_fp * scales_outer;
          accum_int32[i][j] = coopmat<int32_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(0);
        }
      }
    }
  }  // groups

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
