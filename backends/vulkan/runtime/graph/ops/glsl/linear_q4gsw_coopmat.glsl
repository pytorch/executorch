/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * KHR Cooperative Matrix variant of linear_q4gsw_tiled.
 *
 * Performs: out[M,N] = activation[M,K] * weight^T[N,K] (+ bias)
 *   where weight is INT4 group-symmetric quantized (group_size = 4 * K4_per_group).
 *
 * Inner-loop math is pure fp16 -> fp32 MMA via coopMatMulAdd. The per-group
 * weight scale is applied at SHARED-MEMORY STAGE TIME during the B-tile load:
 * each nibble is unpacked, sign-shifted by -8, cast to fp16, and multiplied
 * by the per-(group, output-channel) scale before it lands in Bsh. This keeps
 * the K-loop a clean fp16 MMA with no per-K-element scale fma.
 *
 * Tile hierarchy (mirrors coopmat_mm defaults):
 *   MMA_*         per-MMA-instruction shape (16x16x16 fp16)
 *   WG_TILE_*     output tile per workgroup (64x64; K-step 32)
 *   SG_GRID_*     subgroup grid inside workgroup (2x2 = 4 subgroups)
 *   SUBGROUP_SIZE hardware subgroup width (64 on RDNA3 / Adreno)
 *
 * Storage: activation/output forced to buffer; INT4 weight = texture2d or
 * buffer (yaml variant). DTYPE = half only.
 *
 * Hard preconditions (no shape/alignment checks inside the shader):
 *   M % WG_TILE_M == 0   (= 64)
 *   N % WG_TILE_N == 0   (= 64)
 *   K % WG_TILE_K == 0   (= 32)
 *   group_size % WG_TILE_K == 0   (so each group is an integer number of chunks)
 * Misaligned shapes silently miscompute / overrun — gate at dispatch time.
 */

#version 450 core

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : enable

#define PRECISION ${PRECISION}

$if HAS_BIAS:
  #define HAS_BIAS

$if WEIGHT_STORAGE == "buffer":
  #define WEIGHT_BUFFER

layout(std430) buffer;

#include "common.glslh"

// Bindings — match the order used by add_linear_qw_node so the dispatch
// site can reuse the same arg layout.
${layout_declare_tensor(B, "w", "t_output",              "half", "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_input",               "half", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_packed_int4_weight",  "int",  WEIGHT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_scales",       "half", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias",                "half", "buffer", is_scalar_array=True)}

${layout_declare_ubo(B, "ivec4", "output_sizes")}
${layout_declare_ubo(B, "ivec4", "input_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "apply_bias",   "0")}
${layout_declare_spec_const(C, "int", "K4_per_group", "0")}
// num_groups passed as a spec constant (not derived from the runtime sizes UBO):
// the Xclipse/AMD-PAL shader compiler crashes (null deref in vkCreateComputePipelines)
// when a loop containing coopMatMulAdd has a UBO-derived trip count.
${layout_declare_spec_const(C, "int", "num_groups_arg", "0")}
// Output width N for coopMatStore, as a spec constant: the same compiler
// MISCOMPILES coopMatStore whose offset/stride derive from a UBO value (only
// the first store per subgroup lands correctly; standalone repro cm_acc2).
${layout_declare_spec_const(C, "int", "out_N_arg", "0")}

// --- Tile geometry (from yaml; defaults match coopmat_mm) ---
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

shared uvec4 Ash[WG_TILE_M * A_STRIDE_VEC4];
shared uvec4 Bsh[WG_TILE_K * B_STRIDE_VEC4];
shared float16_t scales_sh[WG_TILE_N];
#ifdef HAS_BIAS
shared float bias_sh[WG_TILE_N];
#endif

// Fp32 accumulator coopmats (MMAS_PER_SG_M x MMAS_PER_SG_N per thread)
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
  const uint K4 = (K + 3u) / 4u;
  const uint N4 = (N + 3u) / 4u;

  const uint K_per_group = uint(K4_per_group) * 4u;
  const uint num_groups = uint(num_groups_arg);
  const uint CHUNKS_PER_GROUP = K_per_group / WG_TILE_K;

  const uint tile_m_start = WG_TILE_M * tileID.y;
  const uint tile_n_start = WG_TILE_N * tileID.x;

  // Initialize fp32 accumulators to zero.
  [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
    [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
      result[i][j] = coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(0.0);
    }
  }

  // Thread assignment for A tile staging (each thread writes one uvec4 = 8 fp16).
  // WG_TILE_K = 32 -> 4 uvec4 columns of A.  WG_SIZE = 256, WG_TILE_M = 64 ->
  // each thread handles exactly (256/64)=4 A-rows × (4/4)=1 col per outer K iter
  // ... actually 256 threads / 4 cols = 64 rows, matches WG_TILE_M=64. One pass.
  const uint INVS_PER_ROW_A = WG_TILE_K / FP16_PER_VEC4;  // = 4
  const uint a_col = gl_LocalInvocationID.x % INVS_PER_ROW_A;
  const uint a_row_offset = gl_LocalInvocationID.x / INVS_PER_ROW_A;

  // Thread assignment for B tile staging.  WG_TILE_N = 64 -> 8 uvec4 columns of B.
  // WG_SIZE = 256, 256/8 = 32 rows = WG_TILE_K, one pass.
  const uint INVS_PER_ROW_B = WG_TILE_N / FP16_PER_VEC4;  // = 8
  const uint b_col = gl_LocalInvocationID.x % INVS_PER_ROW_B;
  const uint b_row_offset = gl_LocalInvocationID.x / INVS_PER_ROW_B;

  // INT4 weight block grid (see pack_q4_linear_weight.glsl): block (k4, n8)
  // covers K=[k4*4, k4*4+3] x N=[n8*8, n8*8+7]; buffer pitch = K4 blocks per
  // n8 row, texture coord = ivec2(x=k4, y=n8).

  for (uint group_i = 0; group_i < num_groups; ++group_i) {
    // --- Stage per-group weight scales for this WG's N-tile into shared mem.
    //     WG_TILE_N=64 scales; WG_SIZE=256 threads — first 64 lanes load.
    if (gl_LocalInvocationID.x < WG_TILE_N) {
      const uint n_idx = tile_n_start + gl_LocalInvocationID.x;
      const uint n4_idx = n_idx >> 2u;
      const uint n4_off = n_idx & 3u;
      f16vec4 sv = t_weight_scales[group_i * N4 + n4_idx];
      scales_sh[gl_LocalInvocationID.x] = sv[n4_off];
    }
    memoryBarrierShared();
    barrier();

    for (uint inner = 0; inner < CHUNKS_PER_GROUP; ++inner) {
      const uint chunkK = group_i * K_per_group + inner * WG_TILE_K;

      // --- Stage A tile (fp16 activations) -> Ash ---
      {
        const uint row = tile_m_start + a_row_offset;
        const uint k_elem = chunkK + a_col * FP16_PER_VEC4;
        const uint k_hv4 = k_elem / 4u;
        f16vec4 v0 = t_input[row * K4 + k_hv4];
        f16vec4 v1 = t_input[row * K4 + k_hv4 + 1u];
        Ash[a_row_offset * A_STRIDE_VEC4 + a_col] = uvec4(
            packFloat2x16(v0.xy), packFloat2x16(v0.zw),
            packFloat2x16(v1.xy), packFloat2x16(v1.zw));
      }

      // --- Stage B tile from INT4 -> fp16 (with per-group scale) -> Bsh ---
      // Each thread fills one uvec4 = 8 fp16 weights at:
      //   K-row = chunkK + b_row_offset
      //   N range = tile_n_start + b_col*8 .. + b_col*8 + 7
      //
      // Within a packed ivec4 block, int32[r] packs 8 nibbles for 2 N values:
      //   col=2*k_in_block      -> N = n8_blk*8 + r,     K = k4_blk*4 + k_in_block
      //   col=2*k_in_block + 1  -> N = n8_blk*8 + r + 4, K = k4_blk*4 + k_in_block
      {
        const uint k_row = chunkK + b_row_offset;
        const uint n_start = tile_n_start + b_col * 8u;
        const uint k4_blk = k_row >> 2u;
        const uint k_in_block = k_row & 3u;
        const uint n8_blk = n_start >> 3u;

        ivec4 wblock;
#ifdef WEIGHT_BUFFER
        wblock = t_packed_int4_weight[(n8_blk * K4) + k4_blk];
#else
        wblock = texelFetch(t_packed_int4_weight, ivec2(k4_blk, n8_blk), 0);
#endif

        const uint col_lo = 2u * k_in_block;
        const uint col_hi = col_lo + 1u;

        // Dequant + apply per-group scale: w_fp = (nibble - 8) * scale
        f16vec4 v0;
        v0.x = float16_t(int(((wblock[0] >> (4 * col_lo)) & 0xF)) - 8)
             * scales_sh[b_col * 8u + 0u];
        v0.y = float16_t(int(((wblock[1] >> (4 * col_lo)) & 0xF)) - 8)
             * scales_sh[b_col * 8u + 1u];
        v0.z = float16_t(int(((wblock[2] >> (4 * col_lo)) & 0xF)) - 8)
             * scales_sh[b_col * 8u + 2u];
        v0.w = float16_t(int(((wblock[3] >> (4 * col_lo)) & 0xF)) - 8)
             * scales_sh[b_col * 8u + 3u];

        f16vec4 v1;
        v1.x = float16_t(int(((wblock[0] >> (4 * col_hi)) & 0xF)) - 8)
             * scales_sh[b_col * 8u + 4u];
        v1.y = float16_t(int(((wblock[1] >> (4 * col_hi)) & 0xF)) - 8)
             * scales_sh[b_col * 8u + 5u];
        v1.z = float16_t(int(((wblock[2] >> (4 * col_hi)) & 0xF)) - 8)
             * scales_sh[b_col * 8u + 6u];
        v1.w = float16_t(int(((wblock[3] >> (4 * col_hi)) & 0xF)) - 8)
             * scales_sh[b_col * 8u + 7u];

        Bsh[b_row_offset * B_STRIDE_VEC4 + b_col] = uvec4(
            packFloat2x16(v0.xy), packFloat2x16(v0.zw),
            packFloat2x16(v1.xy), packFloat2x16(v1.zw));
      }

      barrier();

      // --- Cooperative matrix MMA over WG_TILE_K ---
      [[unroll]] for (uint k = 0; k < WG_TILE_K / MMA_K; ++k) {
        const uint k_start = MMA_K * k;

        coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_K, gl_MatrixUseA> matA[MMAS_PER_SG_M];
        [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
          const uint row_a = MMA_M * (MMAS_PER_SG_M * warpInTile.y + i);
          coopMatLoad(
              matA[i], Ash,
              row_a * A_STRIDE_VEC4 + k_start / FP16_PER_VEC4,
              A_STRIDE_VEC4,
              gl_CooperativeMatrixLayoutRowMajor);
        }

        coopmat<float16_t, gl_ScopeSubgroup, MMA_K, MMA_N, gl_MatrixUseB> matB;
        [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
          const uint col_b = MMA_N * (MMAS_PER_SG_N * warpInTile.x + j) / FP16_PER_VEC4;
          coopMatLoad(
              matB, Bsh,
              k_start * B_STRIDE_VEC4 + col_b,
              B_STRIDE_VEC4,
              gl_CooperativeMatrixLayoutRowMajor);

          [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
            result[i][j] = coopMatMulAdd(matA[i], matB, result[i][j]);
          }
        }
      }

      barrier();
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
