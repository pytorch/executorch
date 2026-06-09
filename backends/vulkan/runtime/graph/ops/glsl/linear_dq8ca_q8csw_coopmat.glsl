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
 * Uses coopmat<int8> × coopmat<int8> → coopmat<int32> on the matrix unit
 * (RDNA3 V_WMMA_I32_16X16X16_IU8 — verified exposed via
 * queryCooperativeMatrixProperties on Radeon 780M, Mesa RADV).
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
 *   3. wsum / wsc / izp / ifs are all loaded ONCE per WG tile (not per-group).
 *
 * Tile hierarchy (mirrors linear_dq8ca_q4gsw_coopmat for direct comparison):
 *   MMA 16x16x16 int8, WG_TILE 64x64, WG_TILE_K = 32,
 *   4 subgroups × 64 threads = 256/WG.
 *
 * Hard preconditions:
 *   M % 64 == 0, N % 64 == 0, K % 32 == 0,
 *   subgroup_size == 64, device exposes coopmat<int8>×<int8>→<int32> at 16x16x16.
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

// LDS layout: K-slab split + ColumnMajor B + per-col skew padding on B.
//
// The WMMA wave64 lane layout for matrix B wants 4 K-contiguous bytes per lane
// (not 4 N-contiguous), so a RowMajor B in LDS forces one byte load per
// (lane, K-row) pair (a chain of ds_load_u8_d16 + v_perm_b32 repack). The
// layout below avoids that:
//  1. matA stays RowMajor (its lane layout wants 4 K-contiguous bytes per
//     lane — already what RowMajor gives us). Per-row stride 16B (no
//     skew needed: 2-way bank conflict, the wave64 minimum).
//  2. matB switches to ColumnMajor LDS — each N-col is 16 K-rows packed
//     contiguously. Stride between cols = 5 uints = 20 bytes (4 useful +
//     1 pad). The +1 uint skew makes col-stride coprime to 32 banks,
//     eliminating bank conflicts on both reads (coopMatLoad) and writes
//     (Stage B). Each lane still reads 4 K-contiguous bytes per
//     ds_load_b32, no v_perm_b32 repack.
//  3. Split LDS into MMA_K-sized K-slabs (WG_TILE_K=32 → 2 slabs) so each
//     slab's strides are short and 16-byte aligned for the A side.
const uint A_SLAB_INT8     = WG_TILE_M * MMA_K;          // 64 * 16 = 1024 int8/slab
const uint B_USEFUL_U32    = MMA_K / 4u;                 // 4 uints of K data per N-col
const uint B_STRIDE_U32    = B_USEFUL_U32 + 1u;          // 5 uints per col (4 useful + 1 skew)
const uint B_SLAB_U32      = WG_TILE_N * B_STRIDE_U32;   // 64 cols × 5 uints/col = 320 uints/slab
const uint NUM_K_SLABS     = WG_TILE_K / MMA_K;          // 2

const uint A_STRIDE_INT8   = MMA_K;                      // 16 int8 per A row (M-row stride)
const uint B_STRIDE_INT8   = B_STRIDE_U32 * 4u;          // 20 int8 per B col (incl. skew)

const uint A_SLAB_U32      = A_SLAB_INT8 / 4u;           // 256 uints/slab
const uint A_STRIDE_U32    = A_STRIDE_INT8 / 4u;         // 4 uints per A row

shared uint Ash_int8[NUM_K_SLABS * A_SLAB_U32];          // 512 uints = 2048 bytes
shared uint Bsh_int8[NUM_K_SLABS * B_SLAB_U32];          // 640 uints = 2560 bytes

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
  const uint M = uint(input_sizes.y);
  const uint N = uint(output_sizes.x);
  const uint N4 = (N + 3u) / 4u;
  const uint K4 = (K + 3u) / 4u;
  const uint NUM_K_CHUNKS = uint(k_chunks_arg);

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

  for (uint chunk_i = 0; chunk_i < NUM_K_CHUNKS; ++chunk_i) {
    const uint chunkK = chunk_i * WG_TILE_K;

    // --- Stage A: 4H4W packed int8 -> slab-major int8 in Ash_int8 ---
    // LDS layout: [slab][m_row][k_uint_in_slab] where slab is the
    // K-chunk of MMA_K=16 int8 (=4 uints). Each thread fetches one ivec4
    // (4 M-rows × 4 K-positions) and writes 4 uints, one per M-row, to
    // the appropriate slab + k_uint position.
    {
      const uint nblocks_x_A = (K + 3u) >> 2u;
      if (gl_LocalInvocationID.x < (WG_TILE_M >> 2u) * (WG_TILE_K >> 2u)) {
        const uint m_block_in_tile = gl_LocalInvocationID.x >> 3u;
        const uint k_block_in_chunk = gl_LocalInvocationID.x & 7u;
        const uint m4_global = (tile_m_start >> 2u) + m_block_in_tile;
        const uint k4_global = (chunkK >> 2u) + k_block_in_chunk;
        const ivec4 blk = t_packed_int8_input[m4_global * nblocks_x_A + k4_global];
        const uint base_row = m_block_in_tile * 4u;
        // k_block_in_chunk (0..7) splits across NUM_K_SLABS=2 slabs of 4 K-uints each.
        const uint slab_idx        = k_block_in_chunk >> 2u;        // 0 or 1
        const uint k_uint_in_slab  = k_block_in_chunk & 3u;         // 0..3
        const uint slab_base       = slab_idx * A_SLAB_U32;
        [[unroll]] for (uint m4i = 0; m4i < 4u; ++m4i) {
          Ash_int8[slab_base + (base_row + m4i) * A_STRIDE_U32 + k_uint_in_slab] = uint(blk[m4i]);
        }
      }
    }

    // --- Stage B: int8 weight -> ColumnMajor slab in Bsh_int8 ---
    // Source weight layout: each ivec4 at [k4, n4] packs 16 int8s as
    //   wblk[n_in_blk] = (K0, K1, K2, K3) packed (4 K-positions for one N-col).
    // ColumnMajor LDS layout: Bsh[slab][n_col][k_uint_in_col] where
    //   k_uint_in_col ∈ [0, 4) holds 4 packed K-bytes.
    // Critically, wblk[n_in_blk] IS exactly the 4-packed-K-bytes for one
    // N-col — we write it AS-IS to LDS with no byte unpack/repack. The
    // matB coopMatLoad then reads 4 K-contiguous bytes per lane in one
    // ds_load_b32 (no v_perm_b32 chain).
    {
      const uint fetch_slots = (WG_TILE_K >> 2u) * (WG_TILE_N >> 2u); // 8 * 16 = 128
      const uint n4_blocks_per_tile  = WG_TILE_N >> 2u;               // 16
      const uint nblocks_x_B = N4;
      if (gl_LocalInvocationID.x < fetch_slots) {
        const uint k4_in_chunk = gl_LocalInvocationID.x / n4_blocks_per_tile;
        const uint n_uint_col  = gl_LocalInvocationID.x % n4_blocks_per_tile;

        const uint block_y_w  = (chunkK >> 2u) + k4_in_chunk;
        const uint n_start_global = tile_n_start + n_uint_col * 4u;
        const uint block_x_w  = n_start_global >> 2u;

        ivec4 wblk;
#ifdef WEIGHT_BUFFER
        wblk = t_packed_int8_weight[(block_y_w * nblocks_x_B) + block_x_w];
#else
        wblk = texelFetch(t_packed_int8_weight, ivec2(block_x_w, block_y_w), 0);
#endif
        // ColumnMajor write: 4 N-cols at offsets [n_uint_col*4 .. n_uint_col*4+3],
        // each gets ONE uint (wblk[n_in_blk]) at slab position k4_in_slab.
        const uint slab_idx       = k4_in_chunk >> 2u;        // 0 or 1
        const uint k4_in_slab     = k4_in_chunk & 3u;         // 0..3 (which K4-block within slab)
        const uint slab_base      = slab_idx * B_SLAB_U32;
        const uint n_col_base     = n_uint_col * 4u;
        [[unroll]] for (uint n_in_blk = 0u; n_in_blk < 4u; ++n_in_blk) {
          const uint n_col = n_col_base + n_in_blk;
          // Bsh_int8[slab][n_col][k4_in_slab]; each entry = 4 packed K-bytes.
          Bsh_int8[slab_base + n_col * B_STRIDE_U32 + k4_in_slab] = uint(wblk[n_in_blk]);
        }
      }
    }

    barrier();

    // --- Inner K loop: coopmat<int8> x coopmat<int8> -> coopmat<int32> ---
    // Address LDS slabs. Each k iter consumes one slab of MMA_K=16
    // K-rows. coopMatLoad offset/stride are in units of the backing
    // array's element type (uint = 4 packed int8), NOT int8 elements.
    // matA is RowMajor with stride A_STRIDE_U32=4 uints (16 int8,
    // 16-byte aligned). matB is ColumnMajor with stride B_STRIDE_U32=5
    // uints (4 useful + 1 skew), coprime-to-32-banks on the LDS port side.
    [[unroll]] for (uint k = 0; k < NUM_K_SLABS; ++k) {
      const uint slab_a_base_u32 = k * A_SLAB_U32;
      const uint slab_b_base_u32 = k * B_SLAB_U32;

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

    barrier();
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
