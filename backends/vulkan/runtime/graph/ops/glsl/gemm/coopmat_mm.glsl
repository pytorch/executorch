/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * KHR Cooperative Matrix MM kernel — unified linear + matmul.
 *
 * Variants (set in coopmat_mm.yaml):
 *   matmul_coopmat       row_major weight, no bias  (aten.mm runtime mat2)
 *   linear_coopmat       prepacked weight, no bias  (aten.linear)
 *   linear_coopmat_bias  prepacked weight, +bias    (aten.linear w/ bias)
 *
 * Computes: D = A * B[ + bias]
 *   A is [M, K] (row-major).
 *   B is either [K, N] row-major (matmul), or 4OC x 4IC blocked prepacked
 *   with t_weight[(k4 * N4 + n4) * 4 + dk] returning a vec4 of 4 N-elements
 *   at K-row k4*4+dk (linear).
 *   D is [M, N], buffer storage.
 *
 * fp16 x fp16 -> fp32 MMA. When DTYPE=half, inputs/outputs are native fp16
 * (no conversion, half the bandwidth). When DTYPE=float, inputs are fp32
 * with on-the-fly packHalf2x16 conversion at the shared-memory load.
 *
 * When HAS_BIAS, bias is staged once into shared memory and broadcast into
 * each accumulator tile (stride-0 coopMatLoad) before coopMatStore, so
 * t_output is write-only.
 *
 * Tile hierarchy (configurable via yaml; defaults shown for Adreno):
 *   MMA_*         per-MMA-instruction shape (16x16x16 fp16)
 *   WG_TILE_*     output tile produced per workgroup (64x64; K-step 32)
 *   SG_GRID_*     subgroup grid inside the workgroup (2x2 = 4 subgroups)
 *   SG_TILE_*     per-subgroup output tile (= WG_TILE / SG_GRID; 32x32)
 *   SUBGROUP_SIZE hardware subgroup width (64 on Adreno)
 *   WG_SIZE       threads per workgroup (= NUM_SUBGROUPS * SUBGROUP_SIZE)
 */

#version 450 core

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_control_flow_attributes : enable

#define PRECISION ${PRECISION}

$if DTYPE == "half":
  #define IS_FP16_INPUT
$if DTYPE == "float":
  #define IS_FP32_INPUT

$if HAS_BIAS:
  #define HAS_BIAS

$if WEIGHT_LAYOUT == "prepacked":
  #define WEIGHT_PREPACKED

layout(std430) buffer;

#include "common.glslh"

// Bindings: output(0), mat1(1), weight(2), [bias(3)]
${layout_declare_tensor(B, "w", "t_output", DTYPE, "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_mat1", DTYPE, "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_weight", DTYPE, "buffer", is_scalar_array=False)}
$if HAS_BIAS:
  ${layout_declare_tensor(B, "r", "t_bias", DTYPE, "buffer", is_scalar_array=True)}

// UBOs — N comes from out_sizes (linear) or mat2_sizes (matmul).
${layout_declare_ubo(B, "ivec4", "mat1_sizes")}
$if WEIGHT_LAYOUT == "prepacked":
  ${layout_declare_ubo(B, "ivec4", "out_sizes")}
$else:
  ${layout_declare_ubo(B, "ivec4", "mat2_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Cooperative-matrix instruction shape (must match a property enumerated by
// vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR for this device).
const uint MMA_M = ${MMA_M};
const uint MMA_N = ${MMA_N};
const uint MMA_K = ${MMA_K};

// Output tile produced per workgroup.
const uint WG_TILE_M = ${WG_TILE_M};
const uint WG_TILE_N = ${WG_TILE_N};
const uint WG_TILE_K = ${WG_TILE_K};

// Subgroup grid inside the workgroup.
const uint SG_GRID_X = ${SG_GRID_X};
const uint SG_GRID_Y = ${SG_GRID_Y};
const uint SUBGROUP_SIZE = ${SUBGROUP_SIZE};
const uint NUM_SUBGROUPS = SG_GRID_X * SG_GRID_Y;
const uint WG_SIZE = NUM_SUBGROUPS * SUBGROUP_SIZE;

// Derived: per-subgroup tile and MMAs per subgroup tile.
const uint SG_TILE_M = WG_TILE_M / SG_GRID_Y;
const uint SG_TILE_N = WG_TILE_N / SG_GRID_X;
const uint MMAS_PER_SG_M = SG_TILE_M / MMA_M;
const uint MMAS_PER_SG_N = SG_TILE_N / MMA_N;

// fp16: 8 elements per uvec4 (128-bit)
const uint FP16_PER_VEC4 = 8;

// Shared memory with skew padding
const uint A_STRIDE_VEC4 = (WG_TILE_K + FP16_PER_VEC4) / FP16_PER_VEC4;
const uint B_STRIDE_VEC4 = (WG_TILE_N + FP16_PER_VEC4) / FP16_PER_VEC4;

shared uvec4 Ash[WG_TILE_M * A_STRIDE_VEC4];
shared uvec4 Bsh[WG_TILE_K * B_STRIDE_VEC4];

#ifdef HAS_BIAS
// fp32 staging buffer so coopMatLoad can broadcast directly into the
// fp32 accumulator coopmat without a type conversion at the load.
shared float bias_sh[WG_TILE_N];
#endif

// Accumulator tiles (fp32)
coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> result[MMAS_PER_SG_M][MMAS_PER_SG_N];

#ifdef IS_FP32_INPUT
uvec2 f32x4_to_f16x4(vec4 v) {
    return uvec2(packHalf2x16(v.xy), packHalf2x16(v.zw));
}
#endif

void main() {
    const uvec2 tileID = uvec2(gl_WorkGroupID.xy);
    const uvec2 warpInTile = uvec2(
        gl_SubgroupID % SG_GRID_X,
        gl_SubgroupID / SG_GRID_X);

    const uint K = uint(mat1_sizes.x);
    const uint M = uint(mat1_sizes.y);
#ifdef WEIGHT_PREPACKED
    const uint N = uint(out_sizes.x);
#else
    const uint N = uint(mat2_sizes.x);
#endif
    const uint K4 = (K + 3u) / 4u;
    const uint N4 = (N + 3u) / 4u;

    // Defensive: skip workgroups whose tile is out of bounds. The C++ pick
    // function dispatches exactly num_tiles_n x num_tiles_m workgroups under
    // the alignment-gated (M%WG_TILE_M==0, N%WG_TILE_N==0) inputs, so this
    // never triggers today; it guards against a future dispatch error.
    const uint num_tiles_n = (N + WG_TILE_N - 1u) / WG_TILE_N;
    const uint num_tiles_m = (M + WG_TILE_M - 1u) / WG_TILE_M;
    if (tileID.x >= num_tiles_n || tileID.y >= num_tiles_m) {
        return;
    }

    [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
        [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
            result[i][j] = coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(0.0);
        }
    }

    // Thread assignment for A tile (WG_TILE_M rows x INVS_PER_ROW_A uvec4/row)
    const uint INVS_PER_ROW_A = WG_TILE_K / FP16_PER_VEC4;
    const uint a_col = gl_LocalInvocationID.x % INVS_PER_ROW_A;
    const uint a_row_offset = gl_LocalInvocationID.x / INVS_PER_ROW_A;

    // Thread assignment for B tile (WG_TILE_K rows x INVS_PER_ROW_B uvec4/row)
    const uint INVS_PER_ROW_B = WG_TILE_N / FP16_PER_VEC4;
    const uint b_col = gl_LocalInvocationID.x % INVS_PER_ROW_B;
    const uint b_row_offset = gl_LocalInvocationID.x / INVS_PER_ROW_B;

    const uint a_row_base = WG_TILE_M * tileID.y;
    const uint b_col_base = WG_TILE_N * tileID.x;

    for (uint chunkK = 0; chunkK < K; chunkK += WG_TILE_K) {

        // --- Load A tile -> shared (single pass) ---
        {
            uint row = a_row_base + a_row_offset;
            uint k_elem = chunkK + a_col * FP16_PER_VEC4;

#ifdef IS_FP16_INPUT
            uint k_hv4 = k_elem / 4;
            f16vec4 v0 = t_mat1[row * K4 + k_hv4];
            f16vec4 v1 = t_mat1[row * K4 + k_hv4 + 1];
            Ash[a_row_offset * A_STRIDE_VEC4 + a_col] = uvec4(
                packFloat2x16(v0.xy), packFloat2x16(v0.zw),
                packFloat2x16(v1.xy), packFloat2x16(v1.zw));
#else
            uint k_vec4 = k_elem / 4;
            vec4 v0 = t_mat1[row * K4 + k_vec4];
            vec4 v1 = t_mat1[row * K4 + k_vec4 + 1];
            uvec2 h0 = f32x4_to_f16x4(v0);
            uvec2 h1 = f32x4_to_f16x4(v1);
            Ash[a_row_offset * A_STRIDE_VEC4 + a_col] = uvec4(h0, h1);
#endif
        }

        // --- Load B tile -> shared (single pass) ---
        {
            uint k_row = chunkK + b_row_offset;
            uint n_elem = b_col_base + b_col * FP16_PER_VEC4;
            uint n4_0 = n_elem >> 2u;
#ifdef WEIGHT_PREPACKED
            // Prepacked: t_weight[(k4 * N4 + n4) * 4 + dk] yields vec4 of
            // 4 N-elements at K-row (k4*4+dk).
            uint k4 = k_row >> 2u;
            uint dk = k_row & 3u;
            uint b_idx0 = (k4 * N4 + n4_0) * 4u + dk;
            uint b_idx1 = (k4 * N4 + n4_0 + 1u) * 4u + dk;
#else
            // Row-major: t_weight[k_row * N4 + n4] yields vec4 of 4 N-elements.
            uint b_idx0 = k_row * N4 + n4_0;
            uint b_idx1 = k_row * N4 + n4_0 + 1u;
#endif

#ifdef IS_FP16_INPUT
            f16vec4 v0 = t_weight[b_idx0];
            f16vec4 v1 = t_weight[b_idx1];
            Bsh[b_row_offset * B_STRIDE_VEC4 + b_col] = uvec4(
                packFloat2x16(v0.xy), packFloat2x16(v0.zw),
                packFloat2x16(v1.xy), packFloat2x16(v1.zw));
#else
            vec4 v0 = t_weight[b_idx0];
            vec4 v1 = t_weight[b_idx1];
            uvec2 h0 = f32x4_to_f16x4(v0);
            uvec2 h1 = f32x4_to_f16x4(v1);
            Bsh[b_row_offset * B_STRIDE_VEC4 + b_col] = uvec4(h0, h1);
#endif
        }

        barrier();

        // --- Cooperative matrix MMA ---
        [[unroll]] for (uint k = 0; k < WG_TILE_K / MMA_K; ++k) {
            uint k_start = MMA_K * k;

            coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_K, gl_MatrixUseA> matA[MMAS_PER_SG_M];
            [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
                uint row_a = MMA_M * (MMAS_PER_SG_M * warpInTile.y + i);
                coopMatLoad(
                    matA[i], Ash,
                    row_a * A_STRIDE_VEC4 + k_start / FP16_PER_VEC4,
                    A_STRIDE_VEC4,
                    gl_CooperativeMatrixLayoutRowMajor);
            }

            coopmat<float16_t, gl_ScopeSubgroup, MMA_K, MMA_N, gl_MatrixUseB> matB;
            [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
                uint col_b = MMA_N * (MMAS_PER_SG_N * warpInTile.x + j) / FP16_PER_VEC4;
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
    // Stage one WG_TILE_N-wide row of bias into shared memory. The C++ dispatch
    // gate ensures N % WG_TILE_N == 0, so no per-element bounds check is needed.
    {
        const uint tile_n_start = WG_TILE_N * tileID.x;
        for (uint t = gl_LocalInvocationID.x; t < WG_TILE_N; t += WG_SIZE) {
            bias_sh[t] = float(t_bias[tile_n_start + t]);
        }
    }
    memoryBarrierShared();
    barrier();
#endif

    // --- Store result (with bias folded in pre-store, if present) ---
    [[unroll]] for (uint i = 0; i < MMAS_PER_SG_M; ++i) {
        [[unroll]] for (uint j = 0; j < MMAS_PER_SG_N; ++j) {
            uint gi = WG_TILE_M * tileID.y + MMA_M * (MMAS_PER_SG_M * warpInTile.y + i);
            uint gj = WG_TILE_N * tileID.x + MMA_N * (MMAS_PER_SG_N * warpInTile.x + j);

#ifdef HAS_BIAS
            // Stride-0 row-major load broadcasts MMA_N bias values across all
            // MMA_M rows of the accumulator tile.
            uint local_n = MMA_N * (MMAS_PER_SG_N * warpInTile.x + j);
            coopmat<float, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> bias_tile;
            coopMatLoad(
                bias_tile, bias_sh,
                local_n, /*stride=*/0u,
                gl_CooperativeMatrixLayoutRowMajor);
            result[i][j] += bias_tile;
#endif

#ifdef IS_FP16_INPUT
            coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator> out_tile =
                coopmat<float16_t, gl_ScopeSubgroup, MMA_M, MMA_N, gl_MatrixUseAccumulator>(result[i][j]);
            coopMatStore(
                out_tile, t_output,
                gi * N + gj, N,
                gl_CooperativeMatrixLayoutRowMajor);
#else
            coopMatStore(
                result[i][j], t_output,
                gi * N + gj, N,
                gl_CooperativeMatrixLayoutRowMajor);
#endif
        }
    }
}
