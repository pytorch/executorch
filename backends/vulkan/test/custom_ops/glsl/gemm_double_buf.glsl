/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Port of the NVIDIA double-buffered cooperative-matrix GEMM reference
 * (shmem_double_buf4.comp, the "store-first" variant) into the ExecuTorch
 * Vulkan shader system, for an apples-to-apples microbenchmark against
 * matmul_coopmat (coopmat_mm.glsl). The double-buffered loop structure —
 * prologue prefetch into temp registers, store-first + one barrier per
 * iteration, ping-pong shared-memory slices — is preserved verbatim.
 *
 * Structural adaptations only:
 *   - buffer_reference params -> standard SSBO bindings (D, A, B).
 *   - C input / alpha / beta dropped: computes D = A*B like matmul_coopmat.
 *   - fp32 accumulators converted to fp16 at the store, matching the half
 *     variant of matmul_coopmat (t_output is fp16).
 *   - B is row-major [K, N] only (the BColMajor=false path), matching the
 *     runtime-mat2 layout matmul_coopmat reads.
 *   - Tile geometry and the MMA shape are compile-time constants from the
 *     yaml. K and N arrive as spec constants — never from a UBO: the Xclipse
 *     driver crashes on UBO-derived coopmat loop bounds and miscompiles
 *     UBO-derived coopMatStore offsets/strides.
 *   - The reference's 8-subgroup x 32-thread workgroup is kept; the
 *     annotation below makes the runtime force subgroup size 32 at pipeline
 *     creation (Xclipse 970 supports sizes [32, 64]; default is 64).
 */

// REQUIRED_SUBGROUP_SIZE = 32

#version 450 core

#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_control_flow_attributes : enable

layout(std430) buffer;

// Bindings — match add_gemm_double_buf_node: output(0), mat1(1), mat2(2).
layout(set = 0, binding = 0) buffer restrict writeonly t_outputBuffer {
    float16_t t_output[]; // fp16 D [M, N]
};
layout(set = 0, binding = 1) buffer restrict readonly t_mat1Buffer {
    uvec4 t_mat1[]; // fp16 A [M, K] row-major, 8 elements per uvec4
};
layout(set = 0, binding = 2) buffer restrict readonly t_mat2Buffer {
    uvec4 t_mat2[]; // fp16 B [K, N] row-major, 8 elements per uvec4
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int K_arg = 0;
layout(constant_id = 4) const int N_arg = 0;

// MMA instruction shape (lM/lN/lK in the reference).
const uint lM = ${MMA_M};
const uint lN = ${MMA_N};
const uint lK = ${MMA_K};

// Output tile per workgroup and K-step per iteration.
const uint TILE_M = ${TILE_M};
const uint TILE_N = ${TILE_N};
const uint TILE_K = ${TILE_K};

const uint WORKGROUP_WIDTH_IN_SUBGROUPS = ${SG_GRID_X};
const uint WORKGROUP_HEIGHT_IN_SUBGROUPS = ${SG_GRID_Y};
const uint SUBGROUP_SIZE = ${SUBGROUP_SIZE};
const uint NUM_SUBGROUPS =
    WORKGROUP_WIDTH_IN_SUBGROUPS * WORKGROUP_HEIGHT_IN_SUBGROUPS;
const uint INVOCATIONS_PER_WORKGROUP = SUBGROUP_SIZE * NUM_SUBGROUPS;

// A tile is TILE_M rows x TILE_K columns (row-major); B tile is TILE_K rows
// x TILE_N columns (row-major).
const uint A_ROW_LEN = TILE_K;
const uint A_NUM_ROWS = TILE_M;
const uint B_ROW_LEN = TILE_N;
const uint B_NUM_ROWS = TILE_K;

// fp16: 8 elements per uvec4 (A_BITS = 16 in the reference).
const uint ELEMENTS_PER_VEC4 = 8;
const uint ROW_PAD_SH = ELEMENTS_PER_VEC4;

// One ping-pong slice of each shared-memory buffer (in uvec4 units).
const uint ASH_SLICE = A_NUM_ROWS * (A_ROW_LEN + ROW_PAD_SH) / ELEMENTS_PER_VEC4;
const uint BSH_SLICE = B_NUM_ROWS * (B_ROW_LEN + ROW_PAD_SH) / ELEMENTS_PER_VEC4;

// Double-buffered shared memory.
shared uvec4 Ash[2 * ASH_SLICE];
shared uvec4 Bsh[2 * BSH_SLICE];

const uint C_ROWS = TILE_M / WORKGROUP_HEIGHT_IN_SUBGROUPS / lM;
const uint C_COLS = TILE_N / WORKGROUP_WIDTH_IN_SUBGROUPS / lN;
coopmat<float, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator> result[C_ROWS][C_COLS];

void main()
{
    const uint K = uint(K_arg);
    const uint strideA = K;
    const uint strideB = uint(N_arg);
    const uint strideD = uint(N_arg);

    uvec2 tileID = uvec2(gl_WorkGroupID.xy);
    uvec2 warpInTile = uvec2(
        gl_SubgroupID % WORKGROUP_WIDTH_IN_SUBGROUPS,
        gl_SubgroupID / WORKGROUP_WIDTH_IN_SUBGROUPS);

    // Initialize result to zero
    [[unroll]] for (uint i = 0; i < C_ROWS; ++i)
        [[unroll]] for (uint j = 0; j < C_COLS; ++j)
            result[i][j] = coopmat<float, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator>(0.0);

    // Per-thread coordinates within a tile row; constant across all iterations.
    const uint INVS_PER_ROW_A = A_ROW_LEN / ELEMENTS_PER_VEC4;
    const uint atilek = ELEMENTS_PER_VEC4 * (gl_LocalInvocationID.x % INVS_PER_ROW_A);
    const uint INVS_PER_ROW_B = B_ROW_LEN / ELEMENTS_PER_VEC4;
    const uint btilej = ELEMENTS_PER_VEC4 * (gl_LocalInvocationID.x % INVS_PER_ROW_B);

    const uint STRIDE_A_SH = A_ROW_LEN + ROW_PAD_SH;
    const uint STRIDE_B_SH = B_ROW_LEN + ROW_PAD_SH;

    uvec4 temp_A[A_NUM_ROWS / (INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_A)];
    uvec4 temp_B[B_NUM_ROWS / (INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_B)];

    // =========================================================
    // PROLOGUE: prefetch tile 0 from global memory into temp registers,
    // =========================================================
    {
        uint gabase = strideA * (TILE_M * tileID.y);
        [[unroll]] for (uint i = 0; i < A_NUM_ROWS; i += INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_A) {
            uint atilei = i + gl_LocalInvocationID.x / INVS_PER_ROW_A;
            temp_A[i / (INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_A)] =
                t_mat1[(gabase + strideA * atilei + atilek) / ELEMENTS_PER_VEC4];
        }

        uint gbbase = TILE_N * tileID.x;
        [[unroll]] for (uint k = 0; k < B_NUM_ROWS; k += INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_B) {
            uint btilek = k + gl_LocalInvocationID.x / INVS_PER_ROW_B;
            temp_B[k / (INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_B)] =
                t_mat2[(gbbase + strideB * btilek + btilej) / ELEMENTS_PER_VEC4];
        }
    }
    // =========================================================
    // Second part of PROLOGUE: store to shared memory slice 0.
    // =========================================================
    {
        uint cur_base_A = 0;
        uint cur_base_B = 0;

        [[unroll]] for (uint i = 0; i < A_NUM_ROWS; i += INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_A) {
            uint si = i + gl_LocalInvocationID.x / INVS_PER_ROW_A;
            Ash[cur_base_A + (STRIDE_A_SH * si + atilek) / ELEMENTS_PER_VEC4] =
                temp_A[i / (INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_A)];
        }
        [[unroll]] for (uint k = 0; k < B_NUM_ROWS; k += INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_B) {
            uint sk = k + gl_LocalInvocationID.x / INVS_PER_ROW_B;
            Bsh[cur_base_B + (STRIDE_B_SH * sk + btilej) / ELEMENTS_PER_VEC4] =
                temp_B[k / (INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_B)];
        }
    }

    // =========================================================
    // MAIN LOOP — one barrier per iteration
    //
    // Each iteration:
    //   1. barrier() — make the cur slice visible to the math loop.
    //   2. Global prefetch of tile chunkK+TILE_K into temp.
    //   3. Math loop reading from slice cur.
    //   4. Store temp (tile for chunkK+TILE_K) -> slice nxt in shared memory.
    //      Different slices, no conflict with the ongoing math loop.
    // =========================================================
    uint chunkK;
    for (chunkK = 0; chunkK < K - TILE_K; chunkK += TILE_K) {
        // cur is the slice we read from this iteration.
        uint cur = (chunkK / TILE_K) % 2;
        uint cur_base_A = cur * ASH_SLICE;
        uint cur_base_B = cur * BSH_SLICE;
        // nxt is the slice we store to this iteration.
        uint nxt = ((chunkK + TILE_K) / TILE_K) % 2;
        uint nxt_base_A = nxt * ASH_SLICE;
        uint nxt_base_B = nxt * BSH_SLICE;

        // 1. --- barrier — cur slice fully written ---
        barrier();

        // 2. --- prefetch next tile from global memory -> temp ---
        {
            uint gabase = strideA * (TILE_M * tileID.y) + (chunkK + TILE_K);
            [[unroll]] for (uint i = 0; i < A_NUM_ROWS; i += INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_A) {
                uint atilei = i + gl_LocalInvocationID.x / INVS_PER_ROW_A;
                temp_A[i / (INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_A)] =
                    t_mat1[(gabase + strideA * atilei + atilek) / ELEMENTS_PER_VEC4];
            }

            uint gbbase = strideB * (chunkK + TILE_K) + TILE_N * tileID.x;
            [[unroll]] for (uint k = 0; k < B_NUM_ROWS; k += INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_B) {
                uint btilek = k + gl_LocalInvocationID.x / INVS_PER_ROW_B;
                temp_B[k / (INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_B)] =
                    t_mat2[(gbbase + strideB * btilek + btilej) / ELEMENTS_PER_VEC4];
            }
        }

        // 3. --- math loop using cur slice ---
        [[unroll]] for (uint k = 0; k < TILE_K / lK; ++k)
        {
            uint sk = lK * k;

            coopmat<float16_t, gl_ScopeSubgroup, lM, lK, gl_MatrixUseA> matA[C_ROWS];
            [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
                uint si = lM * (C_ROWS * warpInTile.y + i);
                coopMatLoad(matA[i], Ash,
                    cur_base_A + (STRIDE_A_SH * si + sk) / ELEMENTS_PER_VEC4,
                    STRIDE_A_SH / ELEMENTS_PER_VEC4,
                    gl_CooperativeMatrixLayoutRowMajor);
            }

            coopmat<float16_t, gl_ScopeSubgroup, lK, lN, gl_MatrixUseB> matB;
            [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
                uint sj = lN * (C_COLS * warpInTile.x + j);
                coopMatLoad(matB, Bsh,
                    cur_base_B + (STRIDE_B_SH * sk + sj) / ELEMENTS_PER_VEC4,
                    STRIDE_B_SH / ELEMENTS_PER_VEC4,
                    gl_CooperativeMatrixLayoutRowMajor);

                [[unroll]] for (uint i = 0; i < C_ROWS; ++i)
                    result[i][j] = coopMatMulAdd(matA[i], matB, result[i][j]);
            }
        }

        // 4. --- store temp (tile chunkK+TILE_K) -> nxt slice ---
        [[unroll]] for (uint i = 0; i < A_NUM_ROWS; i += INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_A) {
            uint si = i + gl_LocalInvocationID.x / INVS_PER_ROW_A;
            Ash[nxt_base_A + (STRIDE_A_SH * si + atilek) / ELEMENTS_PER_VEC4] =
                temp_A[i / (INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_A)];
        }
        [[unroll]] for (uint k = 0; k < B_NUM_ROWS; k += INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_B) {
            uint sk = k + gl_LocalInvocationID.x / INVS_PER_ROW_B;
            Bsh[nxt_base_B + (STRIDE_B_SH * sk + btilej) / ELEMENTS_PER_VEC4] =
                temp_B[k / (INVOCATIONS_PER_WORKGROUP / INVS_PER_ROW_B)];
        }
    }

    // exit from MAIN LOOP — last chunk

    uint cur = (chunkK / TILE_K) % 2;
    uint cur_base_A = cur * ASH_SLICE;
    uint cur_base_B = cur * BSH_SLICE;

    // --- barrier — cur slice fully written ---
    barrier();

    // --- math loop using cur slice ---
    [[unroll]] for (uint k = 0; k < TILE_K / lK; ++k)
    {
        uint sk = lK * k;

        coopmat<float16_t, gl_ScopeSubgroup, lM, lK, gl_MatrixUseA> matA[C_ROWS];
        [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
            uint si = lM * (C_ROWS * warpInTile.y + i);
            coopMatLoad(matA[i], Ash,
                cur_base_A + (STRIDE_A_SH * si + sk) / ELEMENTS_PER_VEC4,
                STRIDE_A_SH / ELEMENTS_PER_VEC4,
                gl_CooperativeMatrixLayoutRowMajor);
        }

        coopmat<float16_t, gl_ScopeSubgroup, lK, lN, gl_MatrixUseB> matB;
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
            uint sj = lN * (C_COLS * warpInTile.x + j);
            coopMatLoad(matB, Bsh,
                cur_base_B + (STRIDE_B_SH * sk + sj) / ELEMENTS_PER_VEC4,
                STRIDE_B_SH / ELEMENTS_PER_VEC4,
                gl_CooperativeMatrixLayoutRowMajor);

            [[unroll]] for (uint i = 0; i < C_ROWS; ++i)
                result[i][j] = coopMatMulAdd(matA[i], matB, result[i][j]);
        }
    }

    // Store D = A*B (fp32 accumulators -> fp16 output, no C/alpha/beta).
    [[unroll]] for (uint i = 0; i < C_ROWS; ++i) {
        uint gi = TILE_M * tileID.y + lM * (C_ROWS * warpInTile.y + i);
        [[unroll]] for (uint j = 0; j < C_COLS; ++j) {
            uint gj = TILE_N * tileID.x + lN * (C_COLS * warpInTile.x + j);
            coopmat<float16_t, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator> out_tile =
                coopmat<float16_t, gl_ScopeSubgroup, lM, lN, gl_MatrixUseAccumulator>(result[i][j]);
            coopMatStore(out_tile, t_output,
                strideD * gi + gj, strideD,
                gl_CooperativeMatrixLayoutRowMajor);
        }
    }
}
