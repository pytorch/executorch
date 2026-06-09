/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * KHR Cooperative Matrix variant of linear_dq8ca_q4gsw_tiled.
 *
 * Performs: out[M,N] = dequant(int8_act) * dequant(int4_w) (+ bias)
 *
 * Group epilog is coopmat-only: no shared-memory ping-pong, no scalar
 * correction loop.  The dequant + zero-point correction is expressed
 * entirely as coopmat element-wise arithmetic, using stride-0 row-major and
 * column-major coopMatLoad to broadcast per-row and per-column scalars into
 * 16x16 coopmat shapes.
 *
 * Math:
 *   accum_int32 = sum_k(int8_in_k * int4_signed_k)        // coopMatMulAdd
 *   adjusted    = accum_int32 - input_zp[m] * wsum_signed[group, n]
 *   delta_fp    = float(adjusted) * (input_scale[m] * weight_scale[group, n])
 *   result_fp  += delta_fp                                // accumulate across groups
 *
 * Because we sign-extend INT4 -> INT8 in the B-stage, the "8 * input_sum"
 * term in the existing tiled correction (which compensates for unsigned
 * int4 nibbles in dotPacked4x8) cancels out and is not needed here.
 *
 * Tile hierarchy (mirrors coopmat_mm / linear_q4gsw_coopmat):
 *   MMA 16x16x16 int8 (RDNA3 V_WMMA_I32_16X16X16_IU8 — verified exposed via
 *   queryCooperativeMatrixProperties).
 *   WG_TILE 64x64, WG_TILE_K = 32, 4 subgroups x 64 threads = 256/WG.
 *
 * Hard preconditions:
 *   M % 64 == 0, N % 64 == 0, K % 32 == 0, group_size % 32 == 0,
 *   subgroup_size == 64, device exposes coopmat<int8>x<int8>-><int32> at 16x16x16.
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
//   input_scales(4), input_zps(5), packed_int4_weight(6), weight_sums(7),
//   weight_scales(8), bias(9).
${layout_declare_tensor(B, "w", "t_output",              "half", "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_input",               "half", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_packed_int8_input",   "int",  "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_int8_input_sums",     "int",  "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_int8_input_scales",   "half", "texture3d")}
${layout_declare_tensor(B, "r", "t_int8_input_zps",      "int8", "texture3d")}
${layout_declare_tensor(B, "r", "t_packed_int4_weight",  "int",  WEIGHT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_sums",         "int",  "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_weight_scales",       "half", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias",                "half", "buffer", is_scalar_array=True)}

${layout_declare_ubo(B, "ivec4", "output_sizes")}
${layout_declare_ubo(B, "ivec4", "input_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "apply_bias",   "0")}
${layout_declare_spec_const(C, "int", "K4_per_group", "0")}
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

// LDS layout: K-slab split + ColumnMajor B + per-col skew padding, ported
// from linear_dq8ca_q8csw_coopmat (see that file for the full rationale):
// the wave64 int8 WMMA matB lane layout wants 4 K-contiguous bytes per lane,
// so a RowMajor B in LDS forces per-byte ds_load + v_perm repack chains.
// ColumnMajor with a +1-uint skew per column gives one ds_load_b32 per lane
// with a bank-conflict-free col stride. Each uint holds 4 packed int8.
const uint A_SLAB_INT8     = WG_TILE_M * MMA_K;          // 1024 int8/slab
const uint B_USEFUL_U32    = MMA_K / 4u;                 // 4 uints of K data per N-col
const uint B_STRIDE_U32    = B_USEFUL_U32 + 1u;          // 5 uints per col (4 useful + 1 skew)
const uint B_SLAB_U32      = WG_TILE_N * B_STRIDE_U32;   // 320 uints/slab
const uint NUM_K_SLABS     = WG_TILE_K / MMA_K;          // 2

const uint A_SLAB_U32      = A_SLAB_INT8 / 4u;           // 256 uints/slab
const uint A_STRIDE_U32    = MMA_K / 4u;                 // 4 uints per A row

shared uint Ash_int8[NUM_K_SLABS * A_SLAB_U32];          // 512 uints
shared uint Bsh_int8[NUM_K_SLABS * B_SLAB_U32];          // 640 uints

// Per-WG-tile-row activation params (loaded ONCE at WG start; constant across groups).
shared int   izp_sh[WG_TILE_M];   // int32 (cast from int8 source) for broadcast
shared float ifs_sh[WG_TILE_M];   // float32 (cast from fp16 source) for broadcast

// Per-(group, output-channel) weight params for the current group.
shared int   wsum_sh[WG_TILE_N];
shared float wsc_sh[WG_TILE_N];

#ifdef HAS_BIAS
shared float bias_sh[WG_TILE_N];
#endif

// Running fp32 accumulator (across all groups).
coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>
    result[MMAS_PER_SG_M][MMAS_PER_SG_N];

void main() {
  const uvec2 tileID = uvec2(gl_WorkGroupID.xy);
  const uvec2 warpInTile = uvec2(
      gl_SubgroupID % SG_GRID_X,
      gl_SubgroupID / SG_GRID_X);

  const uint K = uint(input_sizes.x);
  const uint M = uint(input_sizes.y);
  const uint N = uint(output_sizes.x);
  const uint N4 = (N + 3u) / 4u;

  const uint K_per_group = uint(K4_per_group) * 4u;
  const uint num_groups = uint(num_groups_arg);
  const uint CHUNKS_PER_GROUP = K_per_group / WG_TILE_K;

  const uint tile_m_start = WG_TILE_M * tileID.y;
  const uint tile_n_start = WG_TILE_N * tileID.x;

  // Initialize running fp32 result tile.
  [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
    [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
      result[i][j] = coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(0.0);
    }
  }

  // --- One-time stage: per-row input zp + scale (constant across K groups) ---
  // Source: texture3d, texelFetch(t_int8_input_scales, (m4, 0, 0)) = vec4(4 fp16),
  //         texelFetch(t_int8_input_zps,    (m4, 0, 0)) = ivec4(4 int8).
  // Each of the first WG_TILE_M/4 = 16 threads loads one m4-block (4 M-rows).
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

  for (uint group_i = 0; group_i < num_groups; ++group_i) {
    // --- Stage per-(group, N) weight scale + signed sum ---
    if (gl_LocalInvocationID.x < WG_TILE_N) {
      const uint n_idx = tile_n_start + gl_LocalInvocationID.x;
      const uint n4_idx = n_idx >> 2u;
      const uint n4_off = n_idx & 3u;
      f16vec4 sv = t_weight_scales[group_i * N4 + n4_idx];
      wsc_sh[gl_LocalInvocationID.x] = float(sv[n4_off]);
      wsum_sh[gl_LocalInvocationID.x] = t_weight_sums[group_i * N + n_idx];
    }
    memoryBarrierShared();
    barrier();

    // --- Reset per-group INT32 cooperative-matrix accumulator ---
    coopmat<int32_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>
        accum_int32[MMAS_PER_SG_M][MMAS_PER_SG_N];
    [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
      [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
        accum_int32[i][j] = coopmat<int32_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(0);
      }
    }

    for (uint inner = 0; inner < CHUNKS_PER_GROUP; ++inner) {
      const uint chunkK = group_i * K_per_group + inner * WG_TILE_K;

      // --- Stage A: 4H4W packed int8 -> slab-major int8 in Ash_int8 ---
      {
        const uint nblocks_x_A = (K + 3u) >> 2u;
        if (gl_LocalInvocationID.x < (WG_TILE_M >> 2u) * (WG_TILE_K >> 2u)) {
          const uint m_block_in_tile = gl_LocalInvocationID.x >> 3u;
          const uint k_block_in_chunk = gl_LocalInvocationID.x & 7u;
          const uint m4_global = (tile_m_start >> 2u) + m_block_in_tile;
          const uint k4_global = (chunkK >> 2u) + k_block_in_chunk;
          const ivec4 blk = t_packed_int8_input[m4_global * nblocks_x_A + k4_global];
          const uint base_row = m_block_in_tile * 4u;
          const uint slab_idx       = k_block_in_chunk >> 2u;   // 0 or 1
          const uint k_uint_in_slab = k_block_in_chunk & 3u;    // 0..3
          const uint slab_base      = slab_idx * A_SLAB_U32;
          [[unroll]] for (uint m4i = 0; m4i < 4u; ++m4i) {
            Ash_int8[slab_base + (base_row + m4i) * A_STRIDE_U32 + k_uint_in_slab] = uint(blk[m4i]);
          }
        }
      }

      // --- Stage B: INT4 -> sign-extended int8, ColumnMajor slab in Bsh_int8 ---
      // INT4 weight block grid (see pack_q4_linear_weight.glsl): block
      // (k4, n8) covers K=[k4*4, k4*4+3] x N=[n8*8, n8*8+7]; buffer pitch =
      // K4 blocks per n8 row, texture coord = ivec2(x=k4, y=n8). Within a
      // block, int32[r] nibble col c maps to N = n8*8 + r + (c&1 ? 4 : 0),
      // K = k4*4 + c/2 — so one (component, parity) pair yields exactly the
      // 4 K-contiguous bytes of one N column = one ColumnMajor LDS uint.
      {
        const uint total_uints = (WG_TILE_K >> 2u) * WG_TILE_N;  // 8 k4-blocks x 64 cols
        const uint nblocks_K_w = (K + 3u) >> 2u;
        for (uint slot = gl_LocalInvocationID.x; slot < total_uints; slot += WG_SIZE) {
          const uint block_in_chunk = slot >> 3u;        // 0..63
          const uint col_in_block   = slot & 7u;         // 0..7
          const uint k4_in_chunk    = block_in_chunk >> 3u;  // 0..7
          const uint n8_in_tile     = block_in_chunk & 7u;   // 0..7

          const uint k4_blk = (chunkK >> 2u) + k4_in_chunk;
          const uint n8_blk = (tile_n_start >> 3u) + n8_in_tile;

          ivec4 wblk;
#ifdef WEIGHT_BUFFER
          wblk = t_packed_int4_weight[(n8_blk * nblocks_K_w) + k4_blk];
#else
          wblk = texelFetch(t_packed_int4_weight, ivec2(k4_blk, n8_blk), 0);
#endif
          const uint r      = col_in_block & 3u;         // block component
          const uint parity = col_in_block >> 2u;        // 0 -> N+0..3, 1 -> N+4..7
          const int  w      = wblk[r];
          const int  base   = int(4u * parity);
          int v0 = (((w >> (base + 0))  & 0xF) - 8) & 0xFF;  // K = k4*4 + 0
          int v1 = (((w >> (base + 8))  & 0xF) - 8) & 0xFF;  // K = k4*4 + 1
          int v2 = (((w >> (base + 16)) & 0xF) - 8) & 0xFF;  // K = k4*4 + 2
          int v3 = (((w >> (base + 24)) & 0xF) - 8) & 0xFF;  // K = k4*4 + 3

          const uint n_col       = n8_in_tile * 8u + r + parity * 4u;
          const uint slab_idx    = k4_in_chunk >> 2u;    // 0 or 1
          const uint k4_in_slab  = k4_in_chunk & 3u;     // 0..3
          Bsh_int8[slab_idx * B_SLAB_U32 + n_col * B_STRIDE_U32 + k4_in_slab] =
              uint(v0 | (v1 << 8) | (v2 << 16) | (v3 << 24));
        }
      }

      barrier();

      // --- Inner K loop: coopmat<int8> x coopmat<int8> -> coopmat<int32> ---
      // Address LDS slabs; each k iter consumes one MMA_K slab. coopMatLoad
      // offset/stride are in units of the backing array's element type
      // (uint = 4 packed int8), NOT int8 elements. matA RowMajor (stride 4
      // uints, 16B aligned); matB ColumnMajor (stride 5 uints incl. skew).
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
    }  // CHUNKS_PER_GROUP

    // --- Group epilog (coopmat-only, no shared-memory ping-pong) ---
    // For each MMA tile in this thread:
    //   wsum_bcast  = broadcast wsum_sh[n] across rows (stride-0 RowMajor)
    //   wsc_bcast   = broadcast wsc_sh[n]  across rows (stride-0 RowMajor)
    //   (izp/ifs row broadcasts are group-invariant, loaded before the loop)
    //   adjusted    = accum_int32 - izp_bcast * wsum_bcast       (int32 element-wise)
    //   delta_fp    = float(adjusted) * (ifs_bcast * wsc_bcast)  (fp element-wise)
    //   result     += delta_fp
    [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
      const uint local_n_base = MMA_N * (MMAS_PER_SG_N * warpInTile.x + j);

      coopmat<int32_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> wsum_bcast;
      coopMatLoad(
          wsum_bcast, wsum_sh,
          local_n_base, /*stride=*/0u,
          gl_CooperativeMatrixLayoutRowMajor);

      coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> wsc_bcast;
      coopMatLoad(
          wsc_bcast, wsc_sh,
          local_n_base, /*stride=*/0u,
          gl_CooperativeMatrixLayoutRowMajor);

      [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
        coopmat<int32_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> adjusted =
            accum_int32[i][j] - izp_bcast[i] * wsum_bcast;

        coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> adjusted_fp =
            coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(adjusted);

        coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> scales_outer =
            ifs_bcast[i] * wsc_bcast;

        result[i][j] += adjusted_fp * scales_outer;
      }
    }
    // No barrier here — accum_int32 is per-subgroup, wsum_sh/wsc_sh stays
    // through to next group's reload (we barrier at the top of the next iter).
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
