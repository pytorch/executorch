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

// int8 row-major shared mem.  Each uint holds 4 packed int8.
const uint A_STRIDE_U32 = WG_TILE_K / 4u;
const uint B_STRIDE_U32 = WG_TILE_N / 4u;

shared uint Ash_int8[WG_TILE_M * A_STRIDE_U32];
shared uint Bsh_int8[WG_TILE_K * B_STRIDE_U32];

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
  const uint num_groups = K / K_per_group;
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

      // --- Stage A: 4H4W packed int8 -> row-major int8 in Ash_int8 ---
      {
        const uint nblocks_x_A = (K + 3u) >> 2u;
        if (gl_LocalInvocationID.x < (WG_TILE_M >> 2u) * (WG_TILE_K >> 2u)) {
          const uint m_block_in_tile = gl_LocalInvocationID.x >> 3u;
          const uint k_block_in_chunk = gl_LocalInvocationID.x & 7u;
          const uint m4_global = (tile_m_start >> 2u) + m_block_in_tile;
          const uint k4_global = (chunkK >> 2u) + k_block_in_chunk;
          const ivec4 blk = t_packed_int8_input[m4_global * nblocks_x_A + k4_global];
          const uint base_row = m_block_in_tile * 4u;
          const uint k_uint_col = k_block_in_chunk;
          [[unroll]] for (uint m4i = 0; m4i < 4u; ++m4i) {
            Ash_int8[(base_row + m4i) * A_STRIDE_U32 + k_uint_col] = uint(blk[m4i]);
          }
        }
      }

      // --- Stage B: INT4 -> sign-extended int8 in Bsh_int8 ---
      {
        const uint total_uints = WG_TILE_K * (WG_TILE_N / 4u);
        const uint nblocks_x_B = N >> 3u;
        for (uint slot = gl_LocalInvocationID.x; slot < total_uints; slot += WG_SIZE) {
          const uint k_row_in_chunk = slot / B_STRIDE_U32;
          const uint n_uint_col     = slot % B_STRIDE_U32;
          const uint k_row_global   = chunkK + k_row_in_chunk;
          const uint n_start_global = tile_n_start + n_uint_col * 4u;

          const uint block_y_w     = k_row_global >> 2u;
          const uint k_in_blk      = k_row_global & 3u;
          const uint block_x_w     = n_start_global >> 3u;
          const uint n_within_block = n_start_global & 7u;

          ivec4 wblk;
#ifdef WEIGHT_BUFFER
          wblk = t_packed_int4_weight[(block_y_w * nblocks_x_B) + block_x_w];
#else
          wblk = texelFetch(t_packed_int4_weight, ivec2(block_x_w, block_y_w), 0);
#endif
          const uint col_x = (n_within_block == 0u) ? (2u * k_in_blk) : (2u * k_in_blk + 1u);
          int v0 = (int(((wblk[0] >> int(4u * col_x)) & 0xF)) - 8) & 0xFF;
          int v1 = (int(((wblk[1] >> int(4u * col_x)) & 0xF)) - 8) & 0xFF;
          int v2 = (int(((wblk[2] >> int(4u * col_x)) & 0xF)) - 8) & 0xFF;
          int v3 = (int(((wblk[3] >> int(4u * col_x)) & 0xF)) - 8) & 0xFF;
          Bsh_int8[slot] = uint(v0 | (v1 << 8) | (v2 << 16) | (v3 << 24));
        }
      }

      barrier();

      // --- Inner K loop: coopmat<int8> x coopmat<int8> -> coopmat<int32> ---
      [[unroll]] for (uint k = 0; k < WG_TILE_K / MMA_K; ++k) {
        const uint k_start = MMA_K * k;

        coopmat<int8_t, gl_ScopeSubgroup, MMA_M, MMA_K, gl_MatrixUseA> matA[MMAS_PER_SG_M];
        [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
          const uint row_a = MMA_M * (MMAS_PER_SG_M * warpInTile.y + i);
          coopMatLoad(
              matA[i], Ash_int8,
              row_a * WG_TILE_K + k_start,
              WG_TILE_K,
              gl_CooperativeMatrixLayoutRowMajor);
        }

        coopmat<int8_t, gl_ScopeSubgroup, MMA_K, MMA_N, gl_MatrixUseB> matB;
        [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
          const uint col_b = MMA_N * (MMAS_PER_SG_N * warpInTile.x + j);
          coopMatLoad(
              matB, Bsh_int8,
              k_start * WG_TILE_N + col_b,
              WG_TILE_N,
              gl_CooperativeMatrixLayoutRowMajor);
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
    //   izp_bcast   = broadcast izp_sh[m]  across cols (stride-0 ColumnMajor)
    //   wsc_bcast   = broadcast wsc_sh[n]  across rows (stride-0 RowMajor)
    //   ifs_bcast   = broadcast ifs_sh[m]  across cols (stride-0 ColumnMajor)
    //   adjusted    = accum_int32 - izp_bcast * wsum_bcast       (int32 element-wise)
    //   delta_fp    = float(adjusted) * (ifs_bcast * wsc_bcast)  (fp element-wise)
    //   result     += delta_fp
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
          gi * N + gj, N,
          gl_CooperativeMatrixLayoutRowMajor);
    }
  }
}
