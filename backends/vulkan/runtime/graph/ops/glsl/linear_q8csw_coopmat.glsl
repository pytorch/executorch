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
 * weight scale is applied at SHARED-MEMORY STAGE TIME during the B-tile load:
 * each int8 weight is cast to fp16 and multiplied by the per-output-channel
 * scale before it lands in Bsh. This keeps the K-loop a clean fp16 MMA.
 *
 * Mirrors linear_q4gsw_coopmat (the int4 sibling) with two differences:
 *   1. B-stage reads int8 weight (no nibble unpack, no -8 bias).
 *   2. No per-group loop — per-channel weight quant has no groups, so a single
 *      K-chunk loop runs the full accumulation; scales are staged ONCE.
 *
 * Tile hierarchy: MMA 16x16x16 fp16, WG_TILE 64x64, WG_TILE_K = 32,
 *   4 subgroups x 64 threads = 256/WG.
 *
 * Hard preconditions: M%64==0, N%64==0, K%32==0, subgroup_size==64.
 * The K-chunk loop bound (NUM_K_CHUNKS = K/WG_TILE_K) is passed as a
 * specialization constant (not derived from the sizes UBO) to avoid the
 * Xclipse/AMD-PAL shader-compiler crash on UBO-derived coopmat loop bounds.
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

shared uvec4 Ash[WG_TILE_M * A_STRIDE_VEC4];
shared uvec4 Bsh[WG_TILE_K * B_STRIDE_VEC4];
shared float16_t scales_sh[WG_TILE_N];
#ifdef HAS_BIAS
shared float bias_sh[WG_TILE_N];
#endif

coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>
    result[MMAS_PER_SG_M][MMAS_PER_SG_N];

void main() {
  const uvec2 tileID = uvec2(gl_WorkGroupID.xy);
  const uvec2 warpInTile = uvec2(
      gl_SubgroupID % SG_GRID_X,
      gl_SubgroupID / SG_GRID_X);

  const uint K = uint(input_sizes.x);
  const uint N = uint(output_sizes.x);
  const uint K4 = (K + 3u) / 4u;
  const uint N4 = (N + 3u) / 4u;
  const uint NUM_K_CHUNKS = uint(k_chunks_arg);

  const uint tile_m_start = WG_TILE_M * tileID.y;
  const uint tile_n_start = WG_TILE_N * tileID.x;

  [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
    [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
      result[i][j] = coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(0.0);
    }
  }

  const uint INVS_PER_ROW_A = WG_TILE_K / FP16_PER_VEC4;  // = 4
  const uint a_col = gl_LocalInvocationID.x % INVS_PER_ROW_A;
  const uint a_row_offset = gl_LocalInvocationID.x / INVS_PER_ROW_A;

  const uint INVS_PER_ROW_B = WG_TILE_N / FP16_PER_VEC4;  // = 8
  const uint b_col = gl_LocalInvocationID.x % INVS_PER_ROW_B;
  const uint b_row_offset = gl_LocalInvocationID.x / INVS_PER_ROW_B;

  // --- One-time stage: per-output-channel weight scales for this N-tile ---
  if (gl_LocalInvocationID.x < WG_TILE_N) {
    const uint n_idx = tile_n_start + gl_LocalInvocationID.x;
    const uint n4_idx = n_idx >> 2u;
    const uint n4_off = n_idx & 3u;
    f16vec4 sv = t_weight_scales[n4_idx];
    scales_sh[gl_LocalInvocationID.x] = sv[n4_off];
  }
  memoryBarrierShared();
  barrier();

  for (uint chunk_i = 0; chunk_i < NUM_K_CHUNKS; ++chunk_i) {
    const uint chunkK = chunk_i * WG_TILE_K;

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

    // --- Stage B tile from INT8 -> fp16 (per-channel scale) -> Bsh ---
    // Each thread fills one uvec4 = 8 fp16 weights at K-row = chunkK+b_row_offset,
    // N range = tile_n_start + b_col*8 .. +7.
    // INT8 weight block layout: t_packed_int8_weight[k4 * N4 + n4] = ivec4 whose
    // component n_in_blk packs 4 K-bytes (K of block k4) for N-col (n4*4+n_in_blk).
    {
      const uint k_row = chunkK + b_row_offset;
      const uint n_start = tile_n_start + b_col * 8u;
      const uint k4 = k_row >> 2u;
      const uint k_in_block = k_row & 3u;
      const uint n4_a = n_start >> 2u; // n_start is a multiple of 8 -> even

      ivec4 wa, wb;
#ifdef WEIGHT_BUFFER
      wa = t_packed_int8_weight[k4 * N4 + n4_a];
      wb = t_packed_int8_weight[k4 * N4 + n4_a + 1u];
#else
      wa = texelFetch(t_packed_int8_weight, ivec2(n4_a, k4), 0);
      wb = texelFetch(t_packed_int8_weight, ivec2(n4_a + 1u, k4), 0);
#endif

      const int shift = int(8u * k_in_block);
      f16vec4 v0;
      v0.x = float16_t(bitfieldExtract(wa.x, shift, 8)) * scales_sh[b_col * 8u + 0u];
      v0.y = float16_t(bitfieldExtract(wa.y, shift, 8)) * scales_sh[b_col * 8u + 1u];
      v0.z = float16_t(bitfieldExtract(wa.z, shift, 8)) * scales_sh[b_col * 8u + 2u];
      v0.w = float16_t(bitfieldExtract(wa.w, shift, 8)) * scales_sh[b_col * 8u + 3u];
      f16vec4 v1;
      v1.x = float16_t(bitfieldExtract(wb.x, shift, 8)) * scales_sh[b_col * 8u + 4u];
      v1.y = float16_t(bitfieldExtract(wb.y, shift, 8)) * scales_sh[b_col * 8u + 5u];
      v1.z = float16_t(bitfieldExtract(wb.z, shift, 8)) * scales_sh[b_col * 8u + 6u];
      v1.w = float16_t(bitfieldExtract(wb.w, shift, 8)) * scales_sh[b_col * 8u + 7u];

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
      coopMatStore(out_tile, t_output, gi * N_out + gj, N_out,
                   gl_CooperativeMatrixLayoutRowMajor);
    }
  }
}
