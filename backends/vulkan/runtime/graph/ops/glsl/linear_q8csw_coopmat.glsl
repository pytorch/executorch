/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * KHR Cooperative Matrix variant of linear_q8csw_tiled (fp16 act x INT8
 * per-channel weight, weight-only quantization).
 *
 * Inner-loop math is pure fp16 -> fp32 MMA via coopMatMulAdd. The per-channel
 * weight scale is applied during the B-tile store to shared memory: each int8
 * weight is cast to fp16 and multiplied by the per-output-channel scale
 * before it lands in Bsh.
 *
 * Loop structure follows the NVIDIA double-buffered GEMM reference
 * (shmem_double_buf4.comp "store-first" variant; see gemm_double_buf.glsl in
 * test/custom_ops and the restructured linear_q4gsw_coopmat.glsl): prologue
 * register prefetch, one barrier per K-chunk, ping-pong LDS slices, INT8 ->
 * fp16 dequant at the store stage.
 *
 * Mirrors linear_q4gsw_coopmat (the int4 sibling) with two differences:
 *   1. B-stage reads int8 weight (no nibble unpack, no -8 bias).
 *   2. No per-group logic — per-channel weight quant has no groups, so each
 *      thread caches its 8 scales in 2 registers ONCE in the prologue.
 *
 * Tile hierarchy (yaml; mirrors the double-buffered reference): MMA 16x16x16
 * fp16, WG_TILE 128x128, WG_TILE_K = 16, 8 subgroups x 32 threads (subgroup
 * size 32 forced via the REQUIRED_SUBGROUP_SIZE annotation below).
 *
 * Hard preconditions: M % WG_TILE_M == 0, N % WG_TILE_N == 0,
 * K % WG_TILE_K == 0. The K-chunk loop bound (= K / WG_TILE_K) is passed as
 * a specialization constant (not derived from the sizes UBO) to avoid the
 * Xclipse/AMD-PAL shader-compiler crash on UBO-derived coopmat loop bounds.
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

$if HAS_BIAS:
  #define HAS_BIAS

$if WEIGHT_STORAGE == "buffer":
  #define WEIGHT_BUFFER

layout(std430) buffer;

#include "common.glslh"

// Bindings — match the order used by add_linear_qw_node (weight-only):
//   output(0), fp_input(1), packed_int8_weight(2), weight_scales(3), bias(4).
${layout_declare_tensor(B, "w", "t_output",              "half", "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_input",               "half", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_packed_int8_weight",  "int",  WEIGHT_STORAGE, is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight_scales",       "half", "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_bias",                "half", "buffer", is_scalar_array=True)}

${layout_declare_ubo(B, "ivec4", "output_sizes")}
${layout_declare_ubo(B, "ivec4", "input_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

${layout_declare_spec_const(C, "int", "apply_bias",   "0")}
// K4_per_group kept inert so the dispatcher's {apply_bias, K4_per_group, loop}
// spec list lines up; per-channel weight has no groups.
${layout_declare_spec_const(C, "int", "K4_per_group", "0")}
// K-chunk loop bound passed as a spec constant (see header note).
${layout_declare_spec_const(C, "int", "k_chunks_arg", "0")}
// Output width N for coopMatStore: the Xclipse compiler MISCOMPILES
// coopMatStore whose offset/stride derive from a UBO value (only the first
// store per subgroup lands correctly; standalone repro cm_acc2).
${layout_declare_spec_const(C, "int", "out_N_arg", "0")}

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

coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>
    result[MMAS_PER_SG_M][MMAS_PER_SG_N];

// Dequant 8 int8 weights (two ivec4 blocks, one K-row selected by shift)
// into 8 scaled fp16 weights (one Bsh uvec4).
uvec4 dequant_block_int8(
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

void main() {
  const uvec2 tileID = uvec2(gl_WorkGroupID.xy);
  const uvec2 warpInTile = uvec2(
      gl_SubgroupID % SG_GRID_X,
      gl_SubgroupID / SG_GRID_X);

  const uint K = uint(input_sizes.x);
  const uint K4 = (K + 3u) / 4u;
  const uint N4 = (uint(output_sizes.x) + 3u) / 4u;
  const uint num_chunks = uint(k_chunks_arg);

  const uint tile_m_start = WG_TILE_M * tileID.y;
  const uint tile_n_start = WG_TILE_N * tileID.x;

  [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
    [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
      result[i][j] = coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(0.0);
    }
  }

  const uint a_col = gl_LocalInvocationID.x % INVS_PER_ROW_A;
  const uint a_row_offset = gl_LocalInvocationID.x / INVS_PER_ROW_A;
  const uint b_col = gl_LocalInvocationID.x % INVS_PER_ROW_B;
  const uint b_row_offset = gl_LocalInvocationID.x / INVS_PER_ROW_B;

  // INT8 weight block layout: t_packed_int8_weight[k4 * N4 + n4] = ivec4
  // whose component n_in_blk packs 4 K-bytes (K of block k4) for N-col
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
    [[unroll]] for (uint p = 0; p < B_PASSES; ++p) {
      const uint k4 = (p * B_ROWS_PER_PASS + b_row_offset) >> 2u;
#ifdef WEIGHT_BUFFER
      temp_Ba[p] = t_packed_int8_weight[k4 * N4 + n4_a];
      temp_Bb[p] = t_packed_int8_weight[k4 * N4 + n4_a + 1u];
#else
      temp_Ba[p] = texelFetch(t_packed_int8_weight, ivec2(n4_a, k4), 0);
      temp_Bb[p] = texelFetch(t_packed_int8_weight, ivec2(n4_a + 1u, k4), 0);
#endif
    }
  }
  {
    [[unroll]] for (uint p = 0; p < A_PASSES; ++p) {
      Ash[(p * A_ROWS_PER_PASS + a_row_offset) * A_STRIDE_VEC4 + a_col] = temp_A[p];
    }
    [[unroll]] for (uint p = 0; p < B_PASSES; ++p) {
      Bsh[(p * B_ROWS_PER_PASS + b_row_offset) * B_STRIDE_VEC4 + b_col] =
          dequant_block_int8(temp_Ba[p], temp_Bb[p], b_shift, sc0, sc1);
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
      [[unroll]] for (uint p = 0; p < B_PASSES; ++p) {
        const uint k4 = (chunkK_nxt + p * B_ROWS_PER_PASS + b_row_offset) >> 2u;
#ifdef WEIGHT_BUFFER
        temp_Ba[p] = t_packed_int8_weight[k4 * N4 + n4_a];
        temp_Bb[p] = t_packed_int8_weight[k4 * N4 + n4_a + 1u];
#else
        temp_Ba[p] = texelFetch(t_packed_int8_weight, ivec2(n4_a, k4), 0);
        temp_Bb[p] = texelFetch(t_packed_int8_weight, ivec2(n4_a + 1u, k4), 0);
#endif
      }
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
        Bsh[nxt_base_B + (p * B_ROWS_PER_PASS + b_row_offset) * B_STRIDE_VEC4 + b_col] =
            dequant_block_int8(temp_Ba[p], temp_Bb[p], b_shift, sc0, sc1);
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

#ifdef HAS_BIAS
  if (apply_bias > 0) {
    for (uint t = gl_LocalInvocationID.x; t < WG_TILE_N; t += WG_SIZE) {
      bias_sh[t] = float(t_bias[tile_n_start + t]);
    }
    memoryBarrierShared();
    barrier();
  }
#endif

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
        coopMatLoad(bias_tile, bias_sh, local_n, /*stride=*/0u,
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
