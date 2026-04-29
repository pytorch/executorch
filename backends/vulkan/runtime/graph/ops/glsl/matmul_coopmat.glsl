/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * KHR Cooperative Matrix matmul following matmul_vec conventions.
 * Drop-in replacement for matmul_vec when storage=buffer and device
 * supports GL_KHR_cooperative_matrix.
 *
 * Computes: D = A * B   (A: [M, K], B: [K, N], D: [M, N])
 *
 * fp16×fp16→fp32 MMA. When DTYPE=half, inputs are native fp16 (no
 * conversion, half the bandwidth). When DTYPE=float, inputs are fp32
 * with on-the-fly packHalf2x16 conversion.
 *
 * Output is always fp32 (fp32 accumulator → fp32 store) when DTYPE=float,
 * or fp16 when DTYPE=half.
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

layout(std430) buffer;

#include "common.glslh"

// Bindings — same order as matmul_vec: output(0), mat1(1), mat2(2)
// Output uses same dtype as input. Inputs use vec4 (non-scalar) for wide loads.
${layout_declare_tensor(B, "w", "t_output", DTYPE, "buffer", is_scalar_array=True)}
${layout_declare_tensor(B, "r", "t_mat1", DTYPE, "buffer", is_scalar_array=False)}
${layout_declare_tensor(B, "r", "t_mat2", DTYPE, "buffer", is_scalar_array=False)}

// UBOs — same as matmul_vec
${layout_declare_ubo(B, "ivec4", "mat1_sizes")}
${layout_declare_ubo(B, "ivec4", "mat2_sizes")}

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Tile dimensions
const uint lM = 16;
const uint lN = 16;
const uint lK = 16;
const uint TILE_M = 64;
const uint TILE_N = 64;
const uint TILE_K = 32;

// Workgroup: 4 subgroups in 2×2 grid, 64 threads each = 256 total
const uint WG_WIDTH = 2;
const uint WG_HEIGHT = 2;
const uint NUM_SUBGROUPS = 4;
const uint INVOCATIONS = 64 * NUM_SUBGROUPS;

// Result tiles per subgroup: 2×2
const uint C_ROWS = TILE_M / WG_HEIGHT / lM; // 2
const uint C_COLS = TILE_N / WG_WIDTH / lN;   // 2

// fp16: 8 elements per uvec4 (128-bit)
const uint FP16_PER_VEC4 = 8;

// Shared memory with skew padding
const uint A_STRIDE_VEC4 = (TILE_K + FP16_PER_VEC4) / FP16_PER_VEC4; // 5
const uint B_STRIDE_VEC4 = (TILE_N + FP16_PER_VEC4) / FP16_PER_VEC4; // 9

shared uvec4 Ash[TILE_M * A_STRIDE_VEC4];  // 5KB
shared uvec4 Bsh[TILE_K * B_STRIDE_VEC4];  // 4.5KB

// Accumulator tiles (fp32)
coopmat<float, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator> result[C_ROWS][C_COLS];

#ifdef IS_FP32_INPUT
uvec2 f32x4_to_f16x4(vec4 v) {
    return uvec2(packHalf2x16(v.xy), packHalf2x16(v.zw));
}
#endif

void main() {
    const uvec2 tileID = uvec2(gl_WorkGroupID.xy);
    const uvec2 warpInTile = uvec2(
        gl_SubgroupID % WG_WIDTH,
        gl_SubgroupID / WG_WIDTH);

    const uint K = uint(mat1_sizes.x);
    const uint M = uint(mat1_sizes.y);
    const uint N = uint(mat2_sizes.x);
    const uint K4 = (K + 3u) / 4u;
    const uint N4 = (N + 3u) / 4u;

    [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
            result[i][j] = coopmat<float, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator>(0.0);
        }
    }

    // Thread assignment for A tile (64 rows × 4 uvec4/row = single pass)
    const uint INVS_PER_ROW_A = TILE_K / FP16_PER_VEC4;  // 4
    const uint a_col = gl_LocalInvocationID.x % INVS_PER_ROW_A;
    const uint a_row_offset = gl_LocalInvocationID.x / INVS_PER_ROW_A;

    // Thread assignment for B tile (32 rows × 8 uvec4/row = single pass)
    const uint INVS_PER_ROW_B = TILE_N / FP16_PER_VEC4;  // 8
    const uint b_col = gl_LocalInvocationID.x % INVS_PER_ROW_B;
    const uint b_row_offset = gl_LocalInvocationID.x / INVS_PER_ROW_B;

    const uint a_row_base = TILE_M * tileID.y;
    const uint b_col_base = TILE_N * tileID.x;

    for (uint chunkK = 0; chunkK < K; chunkK += TILE_K) {

        // --- Load A tile → shared (single pass) ---
        {
            uint row = a_row_base + a_row_offset;
            uint k_elem = chunkK + a_col * FP16_PER_VEC4;

#ifdef IS_FP16_INPUT
            // Native fp16: two f16vec4 loads (64-bit each) → one uvec4 (128-bit)
            uint k_hv4 = k_elem / 4;
            f16vec4 v0 = t_mat1[row * K4 + k_hv4];
            f16vec4 v1 = t_mat1[row * K4 + k_hv4 + 1];
            Ash[a_row_offset * A_STRIDE_VEC4 + a_col] = uvec4(
                packHalf2x16(vec2(v0.xy)), packHalf2x16(vec2(v0.zw)),
                packHalf2x16(vec2(v1.xy)), packHalf2x16(vec2(v1.zw)));
#else
            // fp32 inputs: load two vec4 (8 fp32), convert to 8 fp16
            uint k_vec4 = k_elem / 4;
            vec4 v0 = t_mat1[row * K4 + k_vec4];
            vec4 v1 = t_mat1[row * K4 + k_vec4 + 1];
            uvec2 h0 = f32x4_to_f16x4(v0);
            uvec2 h1 = f32x4_to_f16x4(v1);
            Ash[a_row_offset * A_STRIDE_VEC4 + a_col] = uvec4(h0, h1);
#endif
        }

        // --- Load B tile → shared (single pass) ---
        {
            uint row = chunkK + b_row_offset;
            uint n_elem = b_col_base + b_col * FP16_PER_VEC4;

#ifdef IS_FP16_INPUT
            uint n_hv4 = n_elem / 4;
            f16vec4 v0 = t_mat2[row * N4 + n_hv4];
            f16vec4 v1 = t_mat2[row * N4 + n_hv4 + 1];
            Bsh[b_row_offset * B_STRIDE_VEC4 + b_col] = uvec4(
                packHalf2x16(vec2(v0.xy)), packHalf2x16(vec2(v0.zw)),
                packHalf2x16(vec2(v1.xy)), packHalf2x16(vec2(v1.zw)));
#else
            uint n_vec4 = n_elem / 4;
            vec4 v0 = t_mat2[row * N4 + n_vec4];
            vec4 v1 = t_mat2[row * N4 + n_vec4 + 1];
            uvec2 h0 = f32x4_to_f16x4(v0);
            uvec2 h1 = f32x4_to_f16x4(v1);
            Bsh[b_row_offset * B_STRIDE_VEC4 + b_col] = uvec4(h0, h1);
#endif
        }

        barrier();

        // --- Cooperative matrix MMA ---
        [[unroll]] for (uint k = 0; k < TILE_K / lK; ++k) {
            uint k_start = lK * k;

            coopmat<float16_t, gl_ScopeSubgroup, lM, lK, gl_MatrixUseA> matA[C_ROWS];
            [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
                uint row_a = lM * (C_ROWS * warpInTile.y + i);
                coopMatLoad(
                    matA[i], Ash,
                    row_a * A_STRIDE_VEC4 + k_start / FP16_PER_VEC4,
                    A_STRIDE_VEC4,
                    gl_CooperativeMatrixLayoutRowMajor);
            }

            coopmat<float16_t, gl_ScopeSubgroup, lK, lN, gl_MatrixUseB> matB;
            [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
                uint col_b = lN * (C_COLS * warpInTile.x + j) / FP16_PER_VEC4;
                coopMatLoad(
                    matB, Bsh,
                    k_start * B_STRIDE_VEC4 + col_b,
                    B_STRIDE_VEC4,
                    gl_CooperativeMatrixLayoutRowMajor);

                [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
                    result[i][j] = coopMatMulAdd(matA[i], matB, result[i][j]);
                }
            }
        }

        barrier();
    }

    // --- Store result ---
    [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
            uint gi = TILE_M * tileID.y + lM * (C_ROWS * warpInTile.y + i);
            uint gj = TILE_N * tileID.x + lN * (C_COLS * warpInTile.x + j);
#ifdef IS_FP16_INPUT
            // Convert fp32 accumulator to fp16 for fp16 output buffer
            coopmat<float16_t, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator> out_tile =
                coopmat<float16_t, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator>(result[i][j]);
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
