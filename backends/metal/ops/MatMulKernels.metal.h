/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/metal/kernels/TileLoad.h>

#include <string>

namespace executorch {
namespace backends {
namespace metal_v2 {

//===----------------------------------------------------------------------===//
// matmulKernelSource() — the full Metal shader library backing the four
// matmul-family op classes (MatMulOp / AddMMOp / BAddBMMOp / BatchedMatMulOp).
// EXTRACTED INTO ITS OWN HEADER because the raw-string MSL is ~1k lines and
// drowns out the C++ host code in MatMulOp.mm.
//   §2a  applesimd helpers (frag_coord, swizzle_tile)
//   §2b  Epilogue functors (None, AxpbyBias)
//   §2c  storeFragWithEpilogue (per-lane bounds-safe store)
//   §2d  simdMMAKTile (one K-tile of MMA, mixed-precision)
//   §2e  matmul_naive / matmul_tiled (small/medium fallbacks)
//   §2f  matmul_simd_addmm_t (the unified SIMD-MMA workhorse — handles
//                                both un-fused matmul AND fused matmul+bias
//                                via use_out_source / do_axpby FCs)
//   §2g  Simdgroup helpers (used by gemv / gemv_t below)
//   §2h  gemv / gemv_t (M=1 / N=1 fast paths)
//   §2i  matmul_tensor_ops (Apple9+ tensor_ops::matmul2d)
//   §2j  bmm_<dtype> (small-batch naive batched)
//   §2k  template [[host_name]] instantiations (PSO entry points)
// Function constants (declared at the top of the MSL string, indices 0-4):
//   0: align_M       — M % BM == 0   (DCEs un-aligned safe-load paths)
//   1: align_N       — N % BN == 0
//   2: align_K       — K % BK == 0
//   3: use_out_source — true ⇒ apply bias epilogue; false ⇒ identity
//   4: do_axpby      — true ⇒ use kernel-passed alpha/beta; false ⇒ 1,1
// `inline` makes this safe to include from multiple TUs (each TU gets its
// own static-storage `source` since the function-local static is folded
// across instantiations, but the compiler+linker dedupe the inline body).
// Today only MatMulOp.mm includes this header.
//===----------------------------------------------------------------------===//
inline const std::string& matmulKernelSource() {
  static const std::string source = std::string(kTileLoadMetalSource) + R"(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant int TILE_SIZE = 32;

//===----------------------------------------------------------------------===//
// Naive kernel (fallback for small matrices or older devices)
//===----------------------------------------------------------------------===//

template<typename T>
kernel void matmul_naive(
    device const T* A [[buffer(0)]],
    device const T* B [[buffer(1)]],
    device T* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& K [[buffer(4)]],
    constant int& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
  int row = gid.y;
  int col = gid.x;
  if (row >= M || col >= N) return;

  T sum = T(0);
  for (int k = 0; k < K; k++) {
    sum += A[row * K + k] * B[k * N + col];
  }
  C[row * N + col] = sum;
}

//===----------------------------------------------------------------------===//
// Tiled kernel (medium matrices)
//===----------------------------------------------------------------------===//

template<typename T>
kernel void matmul_tiled(
    device const T* A [[buffer(0)]],
    device const T* B [[buffer(1)]],
    device T* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& K [[buffer(4)]],
    constant int& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]) {

  threadgroup T As[TILE_SIZE][TILE_SIZE + 1];
  threadgroup T Bs[TILE_SIZE][TILE_SIZE + 1];

  int row = tgid.y * TILE_SIZE + tid.y;
  int col = tgid.x * TILE_SIZE + tid.x;

  T sum = T(0);

  for (int tileK = 0; tileK < K; tileK += TILE_SIZE) {
    int aRow = row;
    int aCol = tileK + tid.x;
    As[tid.y][tid.x] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : T(0);

    int bRow = tileK + tid.y;
    int bCol = col;
    Bs[tid.y][tid.x] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : T(0);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 0; k < TILE_SIZE && (tileK + k) < K; k++) {
      sum += As[tid.y][k] * Bs[k][tid.x];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (row < M && col < N) {
    C[row * N + col] = sum;
  }
}

//===----------------------------------------------------------------------===//
// Cross-lane reduction helpers used by gemv / gemv_t and
// matmul_simd_splitk_accum.
//===----------------------------------------------------------------------===//

// Reduce a value across all 32 lanes of a simdgroup. Lane 0 holds the
// total; other lanes hold partial values (don't read).
template <typename T>
METAL_FUNC T simdReduceSum(T x) {
  x += simd_shuffle_down(x, 16);
  x += simd_shuffle_down(x, 8);
  x += simd_shuffle_down(x, 4);
  x += simd_shuffle_down(x, 2);
  x += simd_shuffle_down(x, 1);
  return x;
}

// Reduce across the slow axis of a (SM × SN) lane layout where SN is the
// fast (lane-contiguous) axis: lanes that share `sn` reduce together.
// `stride` is SN (initial shift); `log2_max` is the number of butterfly
// steps = log2(SM). After this, the SM=0 lanes (every `stride`-th lane
// starting at 0) hold the per-`sn` totals; other lanes hold partials.
// Example: SM=8, SN=4 → stride=4, log2_max=3 → shifts 4, 8, 16
// (32 lanes → 4 outputs in lanes 0..3).
template <typename T>
METAL_FUNC T simdReduceSumStrided(T x, ushort stride, ushort log2_max) {
  for (ushort i = 0; i < log2_max; ++i) {
    x += simd_shuffle_down(x, stride);
    stride <<= 1;
  }
  return x;
}

//===----------------------------------------------------------------------===//


template<typename T>
kernel void gemv(
    device const T* A [[buffer(0)]],
    device const T* x [[buffer(1)]],
    device T* y [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& K [[buffer(4)]],
    constant int& N [[buffer(5)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]) {

  int row = gid / 32;
  if (row >= M) return;

  T sum = T(0);
  for (int k = simd_lane; k < K; k += 32) {
    sum += A[row * K + k] * x[k];
  }
  sum = simdReduceSum(sum);
  if (simd_lane == 0) {
    y[row] = sum;
  }
}

//===----------------------------------------------------------------------===//
// GEMV transposed: y = A^T @ x, A is [K, N] row-major, x is [K], y is [N].
// Used when M==1 in matmul (autoregressive decode).
// Design follows MLX's GEMVTKernel (mlx/backend/metal/kernels/gemv_masked.h):
// Per-thread tile of TM K-rows × TN N-cols. Simdgroup is laid out as
// SM × SN lanes (SM*SN=32) splitting the K dimension SM ways and the N
// dimension SN ways. After the K loop, partial sums are reduced across
// the SM K-lanes via simd_shuffle_down (handled by simdReduceSumStrided).
//   tg layout       : 1 simdgroup = 32 threads
//   per-thread tile : TM=4 K-rows × TN=4 N-cols
//   simdgroup tile  : SM=8 K-lanes × SN=4 N-lanes -> 32*TM K-rows / iter,
//                     SN*TN=16 output cols / simdgroup
//   tg per N        : ceil(N / 16)
// Why TM×TN instead of "lane-per-col scalar":
//   - More work per thread (16 fmas/iter vs 1) -> better load amortization,
//     better ILP on the FMA pipeline.
//   - K split SM=8 ways within the simdgroup -> 8x less per-thread K work
//     for the same K, so scales to large K without becoming latency-bound.
//   - Same memory pattern (lanes within a simdgroup access consecutive N
//     cols for fixed K row) -> still fully coalesced.
// Accumulator promoted to float so reduction works for bf16 (Metal's
// simd_shuffle_down has no bfloat overload).
//===----------------------------------------------------------------------===//

template<typename T>
kernel void gemv_t(
    device const T* A [[buffer(0)]],
    device const T* x [[buffer(1)]],
    device T* y [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& K [[buffer(4)]],
    constant int& N [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]) {

  constexpr int SM = 8;
  constexpr int SN = 4;
  constexpr int TM = 4;
  constexpr int TN = 4;
  static_assert(SM * SN == 32, "simdgroup must have 32 lanes");
  constexpr int BLOCK_K = SM * TM;   // K rows consumed per outer iter (32)
  constexpr int COLS_PER_SG = SN * TN;  // output cols per simdgroup (16)

  // Lane decomposition: sn is fast (changes every lane), sm is slow.
  ushort sn = simd_lane % SN;
  ushort sm = simd_lane / SN;

  // Each tg owns COLS_PER_SG consecutive output columns. This thread's TN
  // contiguous cols start here.
  int col_base = int(tgid.x) * COLS_PER_SG + int(sn) * TN;
  if (col_base >= N) return;

  // Per-thread accumulators in float for accuracy + bf16-safe reduction.
  float results[TN] = {0.0f, 0.0f, 0.0f, 0.0f};

  // Determine in-bounds TN for THIS thread (uniform across the K loop).
  // Branch-free inner loop relies on this being checked once.
  int valid_tn = TN;
  if (col_base + TN > N) {
    valid_tn = N - col_base;  // 1..TN-1; we still wrote 'return' above for col_base >= N
  }

  // Whole BLOCK_K chunks (no per-K bounds check needed).
  int k_full = (K / BLOCK_K) * BLOCK_K;
  for (int k_block = 0; k_block < k_full; k_block += BLOCK_K) {
    int k_start = k_block + int(sm) * TM;

    float x_vals[TM];
    #pragma clang loop unroll(full)
    for (int tm = 0; tm < TM; ++tm) {
      x_vals[tm] = float(x[k_start + tm]);
    }
    if (valid_tn == TN) {
      // Hot path: full TN cols. Branch-free inner loop.
      #pragma clang loop unroll(full)
      for (int tm = 0; tm < TM; ++tm) {
        int kk = k_start + tm;
        #pragma clang loop unroll(full)
        for (int tn = 0; tn < TN; ++tn) {
          results[tn] += float(A[kk * N + col_base + tn]) * x_vals[tm];
        }
      }
    } else {
      // Edge tg: partial TN. Bounds-check tn; tm is fully in range.
      #pragma clang loop unroll(full)
      for (int tm = 0; tm < TM; ++tm) {
        int kk = k_start + tm;
        for (int tn = 0; tn < valid_tn; ++tn) {
          results[tn] += float(A[kk * N + col_base + tn]) * x_vals[tm];
        }
      }
    }
  }

  // K tail: remaining K rows < BLOCK_K (only when K is not a multiple of 32).
  if (k_full < K) {
    int k_start = k_full + int(sm) * TM;
    if (k_start < K) {
      int tm_max = min(TM, K - k_start);
      for (int tm = 0; tm < tm_max; ++tm) {
        float xv = float(x[k_start + tm]);
        for (int tn = 0; tn < valid_tn; ++tn) {
          results[tn] += float(A[(k_start + tm) * N + col_base + tn]) * xv;
        }
      }
    }
  }

  // Reduce across SM K-lanes (different sm, same sn). After this, lanes
  // with sm == 0 hold the total; other sm lanes hold garbage.
  #pragma clang loop unroll(full)
  for (int tn = 0; tn < TN; ++tn) {
    results[tn] = simdReduceSumStrided(results[tn], ushort(SN), ushort(3));
  }

  // First K-lane writes the in-bounds cols.
  if (sm == 0) {
    for (int tn = 0; tn < valid_tn; ++tn) {
      y[col_base + tn] = T(results[tn]);
    }
  }
}

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// matmul_simd_splitk_accum — partition reduction kernel
// MLX-faithful transcription of `gemm_splitk_accum` (no-axpby) from
//   mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_splitk.h:168-193
// One thread per output element. Reads partial[0..P-1, m, n], sums in
// fp32, stores out[m, n] in T. We dispatch with grid (N, M, 1) and block
// (32, 1, 1) — Metal divides into TGs as needed. Each thread is identified
// by (gid.x = n, gid.y = m).
//===----------------------------------------------------------------------===//

template <typename T>
[[kernel, max_total_threads_per_threadgroup(32)]]
kernel void matmul_simd_splitk_accum(
    device const float* partial [[buffer(0)]],
    device T* out [[buffer(1)]],
    constant int& M [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& split_k_partitions [[buffer(4)]],
    constant int& split_k_partition_stride [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
  if (int(gid.x) >= N || int(gid.y) >= M) return;

  const int idx = int(gid.y) * N + int(gid.x);
  float acc = 0.0f;
  int offset = idx;
  for (int p = 0; p < split_k_partitions; p++) {
    acc += partial[offset];
    offset += split_k_partition_stride;
  }
  out[idx] = T(acc);
}


template [[host_name("matmul_naive_f32")]] kernel void matmul_naive<float>(device const float*, device const float*, device float*, constant int&, constant int&, constant int&, uint2);
template [[host_name("matmul_naive_f16")]] kernel void matmul_naive<half>(device const half*, device const half*, device half*, constant int&, constant int&, constant int&, uint2);
template [[host_name("matmul_naive_bf16")]] kernel void matmul_naive<bfloat>(device const bfloat*, device const bfloat*, device bfloat*, constant int&, constant int&, constant int&, uint2);

template [[host_name("matmul_tiled_f32")]] kernel void matmul_tiled<float>(device const float*, device const float*, device float*, constant int&, constant int&, constant int&, uint2, uint2, uint2);
template [[host_name("matmul_tiled_f16")]] kernel void matmul_tiled<half>(device const half*, device const half*, device half*, constant int&, constant int&, constant int&, uint2, uint2, uint2);
template [[host_name("matmul_tiled_bf16")]] kernel void matmul_tiled<bfloat>(device const bfloat*, device const bfloat*, device bfloat*, constant int&, constant int&, constant int&, uint2, uint2, uint2);

// (Big-source instantiations of the dead matmul_simd_addmm_t /
// matmul_simd_splitk_partial / matmul_simd_splitk_nax_partial / matmul_mlx_*
// templates were removed when the matmul family migrated to MLX 0.31.2's
// per-shape JIT (ops/mlx_jit/). The template bodies above are retained as
// historical reference; with no instantiations they compile to nothing in
// the metallib.)

//===----------------------------------------------------------------------===//
// Phase B: split-K accum kernel instantiations.
//===----------------------------------------------------------------------===//

template [[host_name("matmul_simd_splitk_accum_f32")]]
kernel void matmul_simd_splitk_accum<float>(
    device const float*, device float*,
    constant int&, constant int&, constant int&, constant int&, uint2);
template [[host_name("matmul_simd_splitk_accum_f16")]]
kernel void matmul_simd_splitk_accum<half>(
    device const float*, device half*,
    constant int&, constant int&, constant int&, constant int&, uint2);
template [[host_name("matmul_simd_splitk_accum_bf16")]]
kernel void matmul_simd_splitk_accum<bfloat>(
    device const float*, device bfloat*,
    constant int&, constant int&, constant int&, constant int&, uint2);

template [[host_name("gemv_f32")]] kernel void gemv<float>(device const float*, device const float*, device float*, constant int&, constant int&, constant int&, uint, uint);
template [[host_name("gemv_f16")]] kernel void gemv<half>(device const half*, device const half*, device half*, constant int&, constant int&, constant int&, uint, uint);
// NOTE: no gemv_bf16: Metal's simd_shuffle_down used inside gemv<T> has no
// bfloat overload (only float/half/int), and instantiating gemv<bfloat>
// would fail to compile the whole shader source — taking down every other
// _bf16 kernel with it. MatMulOp::selectKernel only picks GEMV when N==1,
// which our test models don't hit. If you need bf16 GEMV, refactor the
// kernel to promote to float for the simd reduction.

template [[host_name("gemv_t_f32")]] kernel void gemv_t<float>(device const float*, device const float*, device float*, constant int&, constant int&, constant int&, uint3, uint);
template [[host_name("gemv_t_f16")]] kernel void gemv_t<half>(device const half*, device const half*, device half*, constant int&, constant int&, constant int&, uint3, uint);
template [[host_name("gemv_t_bf16")]] kernel void gemv_t<bfloat>(device const bfloat*, device const bfloat*, device bfloat*, constant int&, constant int&, constant int&, uint3, uint);
// gemv_t_bf16 works because the float accumulator + simdReduceSumStrided
// promotes to float for the cross-lane reduction (Metal's
// simd_shuffle_down has no bfloat overload).

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//

// (Big-source instantiations of the dead matmul_simd_addmm_t / matmul_simd_splitk_partial /
// matmul_simd_splitk_nax_partial templates were removed when the matmul family
// migrated to MLX 0.31.2 per-shape JIT (ops/mlx_jit/). The template bodies are
// retained above as historical reference; with no instantiations they compile
// to nothing in the metallib.)


// NAX split-K partial: handled by MatMulOp::dispatchSplitKNAX through
// ops/mlx_jit/. The MLX `gemm_splitk_nax` template
// (steel/gemm/kernels/steel_gemm_splitk_nax.h) is JIT-compiled per
// (tile × dtype × align) shape on demand.

//===----------------------------------------------------------------------===//
// Naive batched matmul fallback (small problems where SIMD MMA has poor
// occupancy). [B, M, K] @ [B, K, N] -> [B, M, N], one thread per output.
//===----------------------------------------------------------------------===//
template<typename T>
kernel void bmm(
    device const T* A [[buffer(0)]],
    device const T* B [[buffer(1)]],
    device T* C [[buffer(2)]],
    constant int& batch [[buffer(3)]],
    constant int& M [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant int& N [[buffer(6)]],
    constant int& A_batch_stride [[buffer(7)]],
    constant int& B_batch_stride [[buffer(8)]],
    constant int& C_batch_stride [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]) {
  int col = gid.x;
  int row = gid.y;
  int b = gid.z;
  if (row >= M || col >= N || b >= batch) return;
  device const T* A_b = A + b * A_batch_stride;
  device const T* B_b = B + b * B_batch_stride;
  device T* C_b = C + b * C_batch_stride;
  T sum = T(0);
  for (int k = 0; k < K; k++) sum += A_b[row * K + k] * B_b[k * N + col];
  C_b[row * N + col] = sum;
}

template [[host_name("bmm_f32")]] kernel void bmm<float>(device const float*, device const float*, device float*, constant int&, constant int&, constant int&, constant int&, constant int&, constant int&, constant int&, uint3);
template [[host_name("bmm_f16")]] kernel void bmm<half>(device const half*, device const half*, device half*, constant int&, constant int&, constant int&, constant int&, constant int&, constant int&, constant int&, uint3);
template [[host_name("bmm_bf16")]] kernel void bmm<bfloat>(device const bfloat*, device const bfloat*, device bfloat*, constant int&, constant int&, constant int&, constant int&, constant int&, constant int&, constant int&, uint3);
)";
  return source;
}

}  // namespace metal_v2
}  // namespace backends
}  // namespace executorch
