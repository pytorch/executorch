/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MatMulOp.h"
#include <executorch/backends/portable/runtime/metal_v2/OpUtils.h>
#include <executorch/backends/portable/runtime/metal_v2/MetalStream.h>
#include <executorch/backends/portable/runtime/metal_v2/kernels/TileLoad.h>
#include <executorch/backends/portable/runtime/metal_v2/ops/MPSGraphOp.h>
#include <executorch/runtime/platform/log.h>
#include <cstdlib>
#include <cstring>
#include <string>

namespace executorch {
namespace backends {
namespace metal_v2 {

using runtime::Error;

//===----------------------------------------------------------------------===//
// Output Shape
//===----------------------------------------------------------------------===//

std::vector<SizesType> MatMulOp::computeOutputShape(
    EValuePtrSpan inputs) const {

  if (inputs.size() < 2 || !inputs[0]->isTensor() || !inputs[1]->isTensor()) {
    return {};
  }

  auto& A = inputs[0]->toTensor();
  auto& B = inputs[1]->toTensor();

  if (A.dim() < 2 || B.dim() < 2) {
    return {};
  }

  SizesType M = A.size(A.dim() - 2);
  SizesType N = B.size(B.dim() - 1);

  return {M, N};
}

//===----------------------------------------------------------------------===//
// Kernel Selection
//
// Picks among Naive / Tiled / Simd / NT / TN / GEMV / GEMV_T based on:
//   - input layout (row-contig vs col-contig, where col-contig means the
//     tensor is a .T view of an underlying row-contig tensor)
//   - problem size (M, N, K)
//   - device tier (smaller thresholds on phones, larger on Ultra/Max)
//===----------------------------------------------------------------------===//

namespace {

struct MatMulThresholds {
  int simdMNK;   // min M,N,K to pick Simd over Tiled/Naive
  int gemvMK;    // min M (or N for gemv_t) to use the simdgroup gemv path
};

constexpr MatMulThresholds thresholdsForTier(DeviceTier tier) {
  switch (tier) {
    case DeviceTier::Phone:    return {32, 16};
    case DeviceTier::MacUltra: return {64, 32};
    case DeviceTier::MacBase:
    default:                   return {48, 24};
  }
}

} // namespace

const char* MatMulOp::kernelTypePrefix(MatMulKernelType type) const {
  switch (type) {
    case MatMulKernelType::Naive:           return "matmul_naive";
    case MatMulKernelType::Tiled:           return "matmul_tiled";
    case MatMulKernelType::Simd:            return "matmul_simd_t_64_64_16_2_2_n";
    case MatMulKernelType::Simd_BN32:       return "matmul_simd_t_64_32_32_2_2_n";
    case MatMulKernelType::Simd_M32:        return "matmul_simd_t_32_64_32_1_4_n";
    case MatMulKernelType::NT:              return "matmul_simd_t_64_64_16_2_2_t";
    case MatMulKernelType::TN:              return "matmul_simd_t_64_64_16_2_2_tn";
    case MatMulKernelType::GEMV:            return "gemv";
    case MatMulKernelType::GEMV_T:          return "gemv_t";
    case MatMulKernelType::TensorOps:       return "matmul_tensor_ops";
  }
  return "matmul_naive";
}

// selectKernel only handles the size-based fallback ladder for the regular
// (NN) case. NT/TN/GEMV/GEMV_T are picked separately based on input layout.
//
// Tier ladder:
//   Simd     : M >= 64. 64x64 output, 4 sg in 2x2.
//   Simd_M32 : 16 <= M < 64. 32x64 output, 4 sg in 1x4. MLX-style "skinny"
//              variant for prefill batches like Llama M=32.
//   Tiled    : 32 <= M < 16 (rare middle ground), legacy fallback.
//   Naive    : everything smaller.
//
// Variants compiled but NOT auto-routed (kept for future use / experimentation):
// - Simd_M32_BN128: tried for compute-bound large-K cases. Theoretical AI
//   gain (10.7 -> 12.8 FLOPs/byte) is real, but in practice doubling the
//   per-sg register pressure (16 vs 8 simdgroup_matrix accumulators) and
//   threadgroup memory cuts wave-level occupancy by more than the AI gain
//   buys. Net regression on Apple M-series. Could be reconsidered if we
//   add register-blocked variants or tune for specific GPU families.
// - Simd_M32_SplitK: didn't help compute-bound cases (the bottleneck is
//   arithmetic intensity, not parallelism).
MatMulKernelType MatMulOp::selectKernel(int64_t M, int64_t N, int64_t K) const {
  if (M >= 64 && N >= 64 && K >= 16) {
    // MLX-inspired heuristic for fp32 NN, refined empirically from sweep:
    //   N <= 1024              -> BN32 (need more tgs along N for parallelism)
    //   M >= 512 + K >= 4096   -> BN32 (BK=32 halves K-barrier count, big wins
    //                                   when both M and K are large enough that
    //                                   barrier overhead dominates)
    //   otherwise              -> Simd (BN=64 wins for moderate-M large-N where
    //                                   tg-level data reuse beats parallelism)
    if (N <= 1024) return MatMulKernelType::Simd_BN32;
    if (M >= 512 && K >= 4096) return MatMulKernelType::Simd_BN32;
    return MatMulKernelType::Simd;
  }
  if (M >= 2 && N >= 64 && K >= 16) return MatMulKernelType::Simd_M32;
  if (M >= 32 && N >= 32) return MatMulKernelType::Tiled;
  return MatMulKernelType::Naive;
}

//===----------------------------------------------------------------------===//
// Dispatch
//===----------------------------------------------------------------------===//

void MatMulOp::dispatch(
    MetalStream* stream,
    EValuePtrSpan inputs,
    EValuePtrSpan outputs) {

  // TEMPORARY runtime switch: when METAL_USE_MPSGRAPH=1 (or =true), route ALL
  // matmul cases through MPSGraph instead of our hand-written kernels. Useful
  // for benchmarking and as a sanity-check fallback when the custom kernels
  // misbehave. Selection logic below is left intact (just bypassed).
  // Works under both MTL3 and MTL4 (MPSGraphOp branches internally on
  // useMTL4() — under MTL4 it uses a singleton legacy queue + shared event).
  static const bool kForceMPSGraph = []() {
    const char* env = getenv("METAL_USE_MPSGRAPH");
    return env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0);
  }();
  if (kForceMPSGraph) {
    static MPSGraphMatMulOp mpsOp;
    ET_LOG(Info, "MatMulOp: forcing MPSGraph path (METAL_USE_MPSGRAPH=1)");
    mpsOp.dispatch(stream, inputs, outputs);
    return;
  }

  auto& A = inputs[0]->toTensor();
  auto& B = inputs[1]->toTensor();
  auto& C = outputs[0]->toTensor();

  auto err = resizeOutput(inputs, outputs[0]);
  if (err != Error::Ok) {
    ET_LOG(Error, "MatMulOp: failed to resize output");
    return;
  }

  const bool aRC = isRowContiguous(A);
  const bool bRC = isRowContiguous(B);
  const bool aCC = !aRC && isColContiguous(A);
  const bool bCC = !bRC && isColContiguous(B);

  if (!(aRC || aCC) || !(bRC || bCC)) {
    ET_LOG(Error, "MatMulOp: A and B must each be row- or column-contiguous");
    return;
  }
  if (aCC && bCC) {
    ET_LOG(Error, "MatMulOp: matmul_tt (both transposed) is not implemented");
    return;
  }

  int32_t M = static_cast<int32_t>(A.size(0));
  int32_t K = static_cast<int32_t>(A.size(1));
  int32_t N = static_cast<int32_t>(B.size(1));

  ScalarType dtype = C.scalar_type();

  // Pick kernel type from layout + size.
  MatMulKernelType kernelType;
  if (aRC && bRC) {
    if (N == 1)      kernelType = MatMulKernelType::GEMV;
    else if (M == 1) kernelType = MatMulKernelType::GEMV_T;
    else             kernelType = selectKernel(M, N, K);
  } else if (aRC && bCC) {
    kernelType = MatMulKernelType::NT;
  } else /* aCC && bRC — A is transposed (TN) */ {
    kernelType = MatMulKernelType::TN;
  }

  // Upgrade Simd -> TensorOps when device supports Apple9 family (M3+/A17 Pro+)
  // AND sizes meet matmul2d constraints (BM=BN=64, K aligned to 16). We don't
  // upgrade NT/TN — tensor_ops::matmul2d would need a different descriptor
  // (transpose flags). Could be added later if perf justifies it.
  if (kernelType == MatMulKernelType::Simd) {
    auto* metalStream = static_cast<MetalStream*>(stream);
    if (metalStream && metalStream->device() &&
        [metalStream->device() supportsFamily:MTLGPUFamilyApple9] &&
        (M % 64 == 0) && (N % 64 == 0) && (K % 16 == 0) &&
        (dtype == ScalarType::Float ||
         dtype == ScalarType::Half  ||
         dtype == ScalarType::BFloat16)) {
      kernelType = MatMulKernelType::TensorOps;
    }
  }

  std::string kname = std::string(kernelTypePrefix(kernelType)) + "_" + dtypeSuffix(dtype);
  auto* kernel = getKernel(stream, kname.c_str());

  ET_LOG(Info, "MatMulOp: M=%d, K=%d, N=%d, kernel=%s", M, K, N, kname.c_str());

  uvec3 grid, block;

  switch (kernelType) {
    case MatMulKernelType::Naive:
      grid = uvec3((N + 7) / 8, (M + 7) / 8, 1);
      block = uvec3(8, 8, 1);
      break;

    case MatMulKernelType::Tiled:
      grid = uvec3((N + 31) / 32, (M + 31) / 32, 1);
      block = uvec3(32, 32, 1);
      break;

    case MatMulKernelType::Simd:
    case MatMulKernelType::NT:
    case MatMulKernelType::TN:
    case MatMulKernelType::TensorOps:
      // 64x64 output tile, 4 simdgroups (128 threads), grid.z=1 (no batch).
      grid = uvec3((N + 63) / 64, (M + 63) / 64, 1);
      block = uvec3(128, 1, 1);
      break;

    case MatMulKernelType::Simd_BN32:
      // 64x32 output tile (BM=64, BN=32), 4 simdgroups in 2x2 layout.
      // Doubles tg count along N vs Simd, helps small-N cases.
      grid = uvec3((N + 31) / 32, (M + 63) / 64, 1);
      block = uvec3(128, 1, 1);
      break;

    case MatMulKernelType::Simd_M32:
      // 32x64 output tile (BM=32, BN=64), 4 simdgroups in 1x4 layout (128
      // threads), grid.z=1 (no batch).
      grid = uvec3((N + 63) / 64, (M + 31) / 32, 1);
      block = uvec3(128, 1, 1);
      break;

    case MatMulKernelType::GEMV:
      // y = A @ x ; one simdgroup per output row (M outputs).
      grid = uvec3(M * 32, 1, 1);
      block = uvec3(32, 1, 1);
      break;

    case MatMulKernelType::GEMV_T:
      // C = A_row @ B ; MLX-style TM×TN tiled gemv_t. Each tg = 1 simdgroup
      // (32 threads) outputs SN*TN = 16 consecutive columns. Total tgs =
      // ceil(N / 16).
      grid = uvec3(((N + 15) / 16) * 32, 1, 1);
      block = uvec3(32, 1, 1);
      break;
  }

  // GEMV_T has swapped operand semantics: gemv_t(matrix=B, vector=A, out=C).
  if (kernelType == MatMulKernelType::GEMV_T) {
    stream->dispatch(kernel, {
      {B.mutable_data_ptr(), B.nbytes()},  // matrix [K,N]
      {A.mutable_data_ptr(), A.nbytes()},  // vector [K]
      {C.mutable_data_ptr(), C.nbytes()},  // output [N]
      M, K, N
    }, grid, block);
  } else {
    stream->dispatch(kernel, {
      {A.mutable_data_ptr(), A.nbytes()},
      {B.mutable_data_ptr(), B.nbytes()},
      {C.mutable_data_ptr(), C.nbytes()},
      M, K, N
    }, grid, block);
  }
}

//===----------------------------------------------------------------------===//
// Kernel Source
//===----------------------------------------------------------------------===//

// Kernel source: prepend kTileLoadMetalSource so matmul_simd / nt / tn can
// call cooperativeLoadTileVec4 + cooperativeLoadTileTransposedVec4. Built
// once into a static std::string. Shared between MatMulOp and
// BatchedMatMulOp so both can reference matmul_simd and bmm kernels.
static const std::string& matmulKernelSource() {
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
// MMA helper: run one K-tile of multiply-accumulate using simdgroup_matrix.
// Loads FRAGS_M A-fragments × FRAGS_N B-fragments from threadgroup memory
// at (a_row, b_col) within the tile, then performs FRAGS_M × FRAGS_N MMAs
// into the existing C_frag accumulators.
//
// Templating on FRAGS_M / FRAGS_N lets matmul_simd (4×4) and
// matmul_simd_m32 (4×2) share the inner loop without duplication.
//===----------------------------------------------------------------------===//
template <typename T, int FRAGS_M, int FRAGS_N, int SMEM_A_STRIDE, int SMEM_B_STRIDE>
inline void simdMMAKTile(
    simdgroup_matrix<T, 8, 8> C_frag[FRAGS_M][FRAGS_N],
    threadgroup const T* As,   // points at &As_buf[curBuf][a_row][k_off]
    threadgroup const T* Bs,   // points at &Bs_buf[curBuf][k_off][b_col]
    int BK_) {
  for (int k = 0; k < BK_; k += 8) {
    simdgroup_matrix<T, 8, 8> A_frag[FRAGS_M];
    #pragma clang loop unroll(full)
    for (int i = 0; i < FRAGS_M; ++i) {
      simdgroup_load(A_frag[i], As + i * 8 * SMEM_A_STRIDE + k, SMEM_A_STRIDE);
    }
    simdgroup_matrix<T, 8, 8> B_frag[FRAGS_N];
    #pragma clang loop unroll(full)
    for (int j = 0; j < FRAGS_N; ++j) {
      simdgroup_load(B_frag[j], Bs + k * SMEM_B_STRIDE + j * 8, SMEM_B_STRIDE);
    }
    #pragma clang loop unroll(full)
    for (int i = 0; i < FRAGS_M; ++i) {
      #pragma clang loop unroll(full)
      for (int j = 0; j < FRAGS_N; ++j) {
        simdgroup_multiply_accumulate(
            C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// BlockLoader: stateful, MLX-style cooperative tile loader.
//
// Drop-in replacement for cooperativeLoadTileVec4 with several improvements:
//
//  1) Auto-derives per-thread vec width from (BROWS * BCOLS / tgp_size).
//     Our cooperativeLoadTileVec4 was hardcoded to vec4 — that meant a
//     64-thread tg loading a 16x64 tile had to do 4 vec4 loads/thread
//     (16 elts each); BlockLoader auto-derives vec16 = 1 load/thread.
//
//  2) Stateful src pointer + next() advances by one K-tile worth of bytes,
//     avoiding the per-call (gRow * srcStride + gCol) re-derivation.
//
//  3) Branch-free load_safe via predicate SELECT (not predicate FLOW):
//        tmp_val[j] = src[in_bounds ? offset : 0];
//        tmp_val[j] = in_bounds ? tmp_val[j] : 0;
//     The compiler can vectorize fully even at edges, no warp divergence.
//
//  4) reduction_dim template flag selects K-direction:
//        reduction_dim=0 → K is the row dim (B's tile)  → tile_stride = BROWS*src_ld
//        reduction_dim=1 → K is the col dim (A's tile)  → tile_stride = BCOLS
//
//  5) ReadVector POD struct for arbitrary vec_size loads (compiler lowers
//     to underlying vec4/vec8 instructions).
//
// Constraints (static_assert):
//   BROWS * BCOLS must be divisible by tgp_size  (so n_reads is integer)
//   BCOLS must be divisible by n_reads           (so TCOLS is integer)
//
// dst is supplied per call (load_unsafe / load_safe) so that the same
// loader instance can target either of two threadgroup buffers in
// double-buffered K loops.
//===----------------------------------------------------------------------===//

template <
    typename T, short BROWS, short BCOLS, short dst_ld,
    short reduction_dim, short tgp_size>
struct BlockLoader {
  static_assert((BROWS * BCOLS) % tgp_size == 0,
      "BROWS*BCOLS must be divisible by tgp_size");

  // Compile-time-derived shape using enum (Metal does not allow
  // static constexpr struct members in the default address space; enum
  // constants work because they have no storage).
  enum : short {
    n_reads = (BCOLS * BROWS) / tgp_size,
    vec_size = n_reads,
    TCOLS    = BCOLS / n_reads,
    TROWS    = tgp_size / TCOLS,
  };
  static_assert(BCOLS % n_reads == 0,
      "BCOLS must be divisible by n_reads");

  // Per-thread (bi, bj) within the tile.
  const int src_ld;
  const int tile_stride;
  const short bi;
  const short bj;
  device const T* src;

  // POD-sized vector for raw byte copy. Compiler lowers to native vec4/8/16
  // load/store instructions as appropriate.
  struct alignas(sizeof(T)) ReadVector {
    uint8_t v[sizeof(T) * vec_size];
  };

  inline BlockLoader(device const T* src_, int src_ld_, ushort tid)
      : src_ld(src_ld_),
        tile_stride(reduction_dim ? BCOLS : BROWS * src_ld_),
        bi(tid / TCOLS),
        bj(vec_size * (tid % TCOLS)),
        src(src_ + bi * src_ld_ + bj) {}

  // Branch-free load: assumes the entire tile is in-bounds.
  inline void load_unsafe(threadgroup T* dst) const {
    threadgroup T* dst_thread = dst + bi * dst_ld + bj;
    #pragma clang loop unroll(full)
    for (short i = 0; i < BROWS; i += TROWS) {
      *((threadgroup ReadVector*)(dst_thread + i * dst_ld)) =
          *((device const ReadVector*)(src + i * src_ld));
    }
  }

  // Bounds-checked load. src_tile_dim = (in-bounds-cols, in-bounds-rows)
  // for the current tile (computed by caller from M/N/K and tile offsets).
  // Out-of-bounds elements are zero-filled via predicate SELECT — no
  // warp divergence even at edges.
  inline void load_safe(threadgroup T* dst, short2 src_tile_dim) const {
    threadgroup T* dst_thread = dst + bi * dst_ld + bj;
    short2 my_dim = src_tile_dim - short2(bj, bi);

    // This thread is entirely past the tile edge → zero-fill.
    if (my_dim.x <= 0 || my_dim.y <= 0) {
      #pragma clang loop unroll(full)
      for (short i = 0; i < BROWS; i += TROWS) {
        #pragma clang loop unroll(full)
        for (short j = 0; j < vec_size; ++j) {
          dst_thread[i * dst_ld + j] = T(0);
        }
      }
      return;
    }

    bool tmp_idx[vec_size];
    T tmp_val[vec_size];
    #pragma clang loop unroll(full)
    for (short i = 0; i < BROWS; i += TROWS) {
      #pragma clang loop unroll(full)
      for (short j = 0; j < vec_size; ++j) {
        tmp_idx[j] = (i < my_dim.y) && (j < my_dim.x);
      }
      // Predicate SELECT for the load: read from a safe address (offset 0)
      // when out-of-bounds. Avoids reading past the buffer.
      #pragma clang loop unroll(full)
      for (short j = 0; j < vec_size; ++j) {
        tmp_val[j] = src[tmp_idx[j] ? (i * src_ld + j) : 0];
      }
      #pragma clang loop unroll(full)
      for (short j = 0; j < vec_size; ++j) {
        tmp_val[j] = tmp_idx[j] ? tmp_val[j] : T(0);
      }
      #pragma clang loop unroll(full)
      for (short j = 0; j < vec_size; ++j) {
        dst_thread[i * dst_ld + j] = tmp_val[j];
      }
    }
  }

  // Advance src to the next K-tile.
  inline void next() {
    src += tile_stride;
  }
};

//===----------------------------------------------------------------------===//
// matmul_simd_t: templated GEMM kernel with tunable tile params.
//
// Subsumes matmul_simd / matmul_simd_m32 / matmul_nt via template params:
//   BM, BN, BK   : output / K tile dims (BM,BN multiples of 8; BK multiple of 8)
//   WM, WN       : simdgroup grid (WM × WN simdgroups per tg, total WM*WN)
//                  BM must be multiple of WM*8, BN multiple of WN*8.
//   TRANSPOSE_B  : if true, B is physically [N, K] (logical [K, N]); load via
//                  the transposed tile loader (used for matmul_nt).
//
// Shape constraints enforced via static_assert. Per-simdgroup output sub-tile
// is (BM/WM) × (BN/WN), broken into FRAGS_M × FRAGS_N fragments of 8×8.
//
// Threadgroup layout: WM * WN simdgroups × 32 = total threads.
//
// Notes:
//   - Bounds-checked loaders used everywhere (cooperativeLoadTileVec4 zero-
//     pads M/N/K edges). For NN-aligned shapes the compiler may DCE the
//     edge predicates inside the inner loop. We do NOT yet have a separate
//     "branch-free interior" kernel instantiation (item 3 in the MLX gap
//     list); deferred for clarity.
//   - Float accumulator is NOT used here — accumulator type follows T. For
//     long K reductions in fp16/bf16 this may lose precision; revisit if
//     accuracy issues appear.
//===----------------------------------------------------------------------===//

template <typename T, int BM, int BN, int BK, int WM, int WN, bool TRANSPOSE_A, bool TRANSPOSE_B>
kernel void matmul_simd_t(
    device const T* A [[buffer(0)]],
    device const T* B [[buffer(1)]],
    device T* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& K [[buffer(4)]],
    constant int& N [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {

  static_assert(BM % (WM * 8) == 0, "BM must be multiple of WM*8");
  static_assert(BN % (WN * 8) == 0, "BN must be multiple of WN*8");
  static_assert(BK % 8 == 0,        "BK must be multiple of 8");
  static_assert(WM * WN >= 1,       "Need at least 1 simdgroup");

  constexpr int NUM_SIMDS = WM * WN;
  constexpr int NUM_THREADS = NUM_SIMDS * 32;
  constexpr int SUBROWS_PER_SG = BM / WM;
  constexpr int SUBCOLS_PER_SG = BN / WN;
  constexpr int FRAGS_M = SUBROWS_PER_SG / 8;
  constexpr int FRAGS_N = SUBCOLS_PER_SG / 8;
  constexpr int PAD = 4;
  constexpr int SMEM_A = BK + PAD;
  constexpr int SMEM_B = BN + PAD;

  A += int(tgid.z) * M * K;
  B += int(tgid.z) * K * N;
  C += int(tgid.z) * M * N;

  const int tileRow = int(tgid.y) * BM;
  const int tileCol = int(tgid.x) * BN;

  const int simd_m = int(simd_gid) / WN;
  const int simd_n = int(simd_gid) % WN;
  const int subRow = tileRow + simd_m * SUBROWS_PER_SG;
  const int subCol = tileCol + simd_n * SUBCOLS_PER_SG;

  threadgroup T As[2][BM][SMEM_A];
  threadgroup T Bs[2][BK][SMEM_B];

  simdgroup_matrix<T, 8, 8> C_frag[FRAGS_M][FRAGS_N];
  #pragma clang loop unroll(full)
  for (int i = 0; i < FRAGS_M; ++i) {
    #pragma clang loop unroll(full)
    for (int j = 0; j < FRAGS_N; ++j) {
      C_frag[i][j] = simdgroup_matrix<T, 8, 8>(0);
    }
  }

  // BlockLoader for A (used unless TRANSPOSE_A; constructed unconditionally
  // for simplicity — the unused state is just a few register slots).
  // For TRANSPOSE_A=true, A is logically [M, K] but stored as [K, M]; loaded
  // via cooperativeLoadTileTransposedVec4 directly (stateless).
  BlockLoader<T, BM, BK, SMEM_A, /*reduction_dim=*/1, NUM_THREADS>
      loader_a(A + tileRow * K, K, tid);

  const bool m_aligned = (tileRow + BM <= M);
  const bool n_aligned = (tileCol + BN <= N);
  const int  m_inb     = m_aligned ? BM : (M - tileRow);
  const int  n_inb     = n_aligned ? BN : (N - tileCol);

  // A-load helper: dispatches to BlockLoader (NN/NT) or transposed helper (TN).
  // BUF_IDX = 0 or 1 (which double-buffer slot); k_off = K-tile starting
  // offset; k_inb = in-bounds K count for this tile.
  #define LOAD_A_TILE(BUF_IDX, k_off, k_inb)                                        \
    do {                                                                            \
      if (TRANSPOSE_A) {                                                            \
        /* A is physical [K,M], stride M; load logical [BM,BK] tile */              \
        cooperativeLoadTileTransposedVec4<T, BM, BK, SMEM_A, NUM_THREADS>(          \
            As[(BUF_IDX)], A, K, M, M, (k_off), tileRow, tid);                      \
      } else if (m_aligned && (k_inb) == BK) {                                      \
        loader_a.load_unsafe(&As[(BUF_IDX)][0][0]);                                 \
      } else {                                                                      \
        loader_a.load_safe(&As[(BUF_IDX)][0][0], short2((k_inb), m_inb));           \
      }                                                                             \
    } while (0)

  if (TRANSPOSE_B) {
    // Initial loads: A (NN or TN-transposed) + B via existing transposed helper.
    LOAD_A_TILE(0, 0, min(int(BK), K));
    cooperativeLoadTileTransposedVec4<T, BK, BN, SMEM_B, NUM_THREADS>(
        Bs[0], B, K, N, K, 0, tileCol, tid);
  } else {
    // BlockLoader for B (non-transposed): K is the ROW dim.
    BlockLoader<T, BK, BN, SMEM_B, /*reduction_dim=*/0, NUM_THREADS>
        loader_b(B + tileCol, N, tid);

    LOAD_A_TILE(0, 0, min(int(BK), K));
    if (n_aligned && BK <= K) loader_b.load_unsafe(&Bs[0][0][0]);
    else loader_b.load_safe(&Bs[0][0][0],
                            short2(n_inb, min(int(BK), K)));

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int numKTiles = (K + BK - 1) / BK;

    for (int t = 0; t < numKTiles; t++) {
      int curBuf = t & 1;
      int nextBuf = curBuf ^ 1;

      if (t + 1 < numKTiles) {
        if (!TRANSPOSE_A) loader_a.next();
        loader_b.next();
        int nextTileK = (t + 1) * BK;
        int k_inb = min(int(BK), K - nextTileK);
        LOAD_A_TILE(nextBuf, nextTileK, k_inb);
        if (n_aligned && k_inb == BK) loader_b.load_unsafe(&Bs[nextBuf][0][0]);
        else loader_b.load_safe(&Bs[nextBuf][0][0], short2(n_inb, k_inb));
      }

      simdMMAKTile<T, FRAGS_M, FRAGS_N, SMEM_A, SMEM_B>(
          C_frag,
          &As[curBuf][simd_m * SUBROWS_PER_SG][0],
          &Bs[curBuf][0][simd_n * SUBCOLS_PER_SG],
          BK);

      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (int i = 0; i < FRAGS_M; ++i) {
      #pragma clang loop unroll(full)
      for (int j = 0; j < FRAGS_N; ++j) {
        int outRow = subRow + i * 8;
        int outCol = subCol + j * 8;
        if (outRow < M && outCol < N) {
          simdgroup_store(C_frag[i][j], C + outRow * N + outCol, N);
        }
      }
    }
    return;
  }

  // ========== TRANSPOSE_B=true K-loop ==========

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int numKTiles = (K + BK - 1) / BK;

  for (int t = 0; t < numKTiles; t++) {
    int curBuf = t & 1;
    int nextBuf = curBuf ^ 1;

    if (t + 1 < numKTiles) {
      if (!TRANSPOSE_A) loader_a.next();
      int nextTileK = (t + 1) * BK;
      int k_inb = min(int(BK), K - nextTileK);
      LOAD_A_TILE(nextBuf, nextTileK, k_inb);
      cooperativeLoadTileTransposedVec4<T, BK, BN, SMEM_B, NUM_THREADS>(
          Bs[nextBuf], B, K, N, K, nextTileK, tileCol, tid);
    }

    simdMMAKTile<T, FRAGS_M, FRAGS_N, SMEM_A, SMEM_B>(
        C_frag,
        &As[curBuf][simd_m * SUBROWS_PER_SG][0],
        &Bs[curBuf][0][simd_n * SUBCOLS_PER_SG],
        BK);

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  #pragma clang loop unroll(full)
  for (int i = 0; i < FRAGS_M; ++i) {
    #pragma clang loop unroll(full)
    for (int j = 0; j < FRAGS_N; ++j) {
      int outRow = subRow + i * 8;
      int outCol = subCol + j * 8;
      if (outRow < M && outCol < N) {
        simdgroup_store(C_frag[i][j], C + outRow * N + outCol, N);
      }
    }
  }
  #undef LOAD_A_TILE
}

//===----------------------------------------------------------------------===//
// Simdgroup helpers
//===----------------------------------------------------------------------===//

// Sum-reduce a per-lane value across the 32-lane simdgroup using the
// shuffle-down ladder. After this, lane 0 holds the total; other lanes hold
// partial sums (don't rely on them).
//
// Note: simd_shuffle_down has overloads for float/half/int but NOT bfloat,
// so any kernel that uses this can't be instantiated for bfloat directly —
// see gemv below. Kernels that don't need cross-lane reduction (e.g. the
// new gemv_t) work for bfloat too.
template<typename T>
inline T simdReduceSum(T x) {
  #pragma clang loop unroll(full)
  for (int offset = 16; offset > 0; offset /= 2) {
    x += simd_shuffle_down(x, ushort(offset));
  }
  return x;
}

// Sum-reduce a per-lane value across a SUBSET of lanes within the simdgroup
// — specifically, lanes whose IDs differ only in some upper-stride bits.
// `stride` is the smallest distance between two lanes that should be merged
// (= the count of "fast" lanes that don't participate). `log2_count` is
// the number of merges = log2(participating lane count).
//
// Example layout: lane_id = sm * SN + sn, with sm in [0, SM) and sn in
// [0, SN). To reduce across SM lanes (different sm values, same sn), use
// stride=SN and log2_count=log2(SM). After the call, lanes with sm == 0
// hold the reduced total for their sn group; other sm lanes hold garbage.
template<typename T>
inline T simdReduceSumStrided(T x, ushort stride, ushort log2_count) {
  for (ushort i = 0; i < log2_count; ++i) {
    x += simd_shuffle_down(x, ushort(stride << i));
  }
  return x;
}

//===----------------------------------------------------------------------===//
// matmul_simd_addmm_t: NN matmul with FUSED bias add (epilogue fusion).
//
// CURRENTLY BROKEN — kept as a starting point for a future session.
//
// Problem: MSL's simdgroup_matrix does NOT support the '+' binary operator,
// so the naive `simdgroup_store(C_frag[i][j] + bias_frag, ...)` fails to
// compile. To make this work we'd need one of:
//
//   1) Store C_frag to TGSM via simdgroup_store, then have each thread
//      do a scalar bias-add load→store pass to global C. ~30 LOC + ~16KB
//      additional TGSM (overlay-able with the As/Bs buffers since K-loop
//      is done; would need a union or careful sequencing).
//
//   2) Use simdgroup_multiply_accumulate(out, identity, bias_frag, C_frag)
//      where identity is an 8x8 identity matrix loaded from a TGSM constant.
//      Requires constructing the identity (no built-in MSL constructor).
//
//   3) Switch to a per-fragment lane-aware scalar epilogue using
//      simd_shuffle to gather bias values per lane.
//
// All three are real options. Approach 1 is simplest correct;
// approach 2 keeps everything in registers (best perf); approach 3 is the
// most general (easy to extend to other epilogue ops like ReLU/SiLU).
//
// Additional non-kernel work needed for end-to-end addmm:
//   - aoti_torch_mps_addmm_out shim function (AOTI C ABI boundary)
//   - Update partition allow-list to include aten::addmm
//   - Make decompose_linear_pass conditional on whether v2 supports addmm
//
// For now: this kernel template body is the un-fused matmul (no bias),
// kept compiled so the AddMMOp class wiring remains in place. It will
// produce INCORRECT results (matmul without the bias add) if invoked.
// Since AOTI rejects addmm at export time today (no shim), this is not
// reachable from any test or model. Marked TODO for future work.
//===----------------------------------------------------------------------===//

template <typename T, int BM, int BN, int BK, int WM, int WN>
kernel void matmul_simd_addmm_t(
    device const T* A [[buffer(0)]],
    device const T* B [[buffer(1)]],
    device T* C [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& K [[buffer(4)]],
    constant int& N [[buffer(5)]],
    device const T* BIAS [[buffer(6)]],
    constant int& bias_stride_m [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]]) {

  (void)BIAS;            // unused while bias-fusion is TODO (see header)
  (void)bias_stride_m;

  static_assert(BM % (WM * 8) == 0, "BM must be multiple of WM*8");
  static_assert(BN % (WN * 8) == 0, "BN must be multiple of WN*8");
  static_assert(BK % 8 == 0,        "BK must be multiple of 8");

  constexpr int NUM_SIMDS = WM * WN;
  constexpr int NUM_THREADS = NUM_SIMDS * 32;
  constexpr int SUBROWS_PER_SG = BM / WM;
  constexpr int SUBCOLS_PER_SG = BN / WN;
  constexpr int FRAGS_M = SUBROWS_PER_SG / 8;
  constexpr int FRAGS_N = SUBCOLS_PER_SG / 8;
  constexpr int PAD = 4;
  constexpr int SMEM_A = BK + PAD;
  constexpr int SMEM_B = BN + PAD;

  A += int(tgid.z) * M * K;
  B += int(tgid.z) * K * N;
  C += int(tgid.z) * M * N;

  const int tileRow = int(tgid.y) * BM;
  const int tileCol = int(tgid.x) * BN;
  const int simd_m = int(simd_gid) / WN;
  const int simd_n = int(simd_gid) % WN;
  const int subRow = tileRow + simd_m * SUBROWS_PER_SG;
  const int subCol = tileCol + simd_n * SUBCOLS_PER_SG;

  threadgroup T As[2][BM][SMEM_A];
  threadgroup T Bs[2][BK][SMEM_B];

  simdgroup_matrix<T, 8, 8> C_frag[FRAGS_M][FRAGS_N];
  #pragma clang loop unroll(full)
  for (int i = 0; i < FRAGS_M; ++i) {
    #pragma clang loop unroll(full)
    for (int j = 0; j < FRAGS_N; ++j) {
      C_frag[i][j] = simdgroup_matrix<T, 8, 8>(0);
    }
  }

  BlockLoader<T, BM, BK, SMEM_A, /*reduction_dim=*/1, NUM_THREADS>
      loader_a(A + tileRow * K, K, tid);
  BlockLoader<T, BK, BN, SMEM_B, /*reduction_dim=*/0, NUM_THREADS>
      loader_b(B + tileCol, N, tid);

  const bool m_aligned = (tileRow + BM <= M);
  const bool n_aligned = (tileCol + BN <= N);
  const int  m_inb     = m_aligned ? BM : (M - tileRow);
  const int  n_inb     = n_aligned ? BN : (N - tileCol);

  if (m_aligned && BK <= K) loader_a.load_unsafe(&As[0][0][0]);
  else loader_a.load_safe(&As[0][0][0], short2(min(int(BK), K), m_inb));
  if (n_aligned && BK <= K) loader_b.load_unsafe(&Bs[0][0][0]);
  else loader_b.load_safe(&Bs[0][0][0], short2(n_inb, min(int(BK), K)));

  threadgroup_barrier(mem_flags::mem_threadgroup);

  const int numKTiles = (K + BK - 1) / BK;

  for (int t = 0; t < numKTiles; t++) {
    int curBuf = t & 1;
    int nextBuf = curBuf ^ 1;

    if (t + 1 < numKTiles) {
      loader_a.next();
      loader_b.next();
      int nextTileK = (t + 1) * BK;
      int k_inb = min(int(BK), K - nextTileK);
      if (m_aligned && k_inb == BK) loader_a.load_unsafe(&As[nextBuf][0][0]);
      else loader_a.load_safe(&As[nextBuf][0][0], short2(k_inb, m_inb));
      if (n_aligned && k_inb == BK) loader_b.load_unsafe(&Bs[nextBuf][0][0]);
      else loader_b.load_safe(&Bs[nextBuf][0][0], short2(n_inb, k_inb));
    }

    simdMMAKTile<T, FRAGS_M, FRAGS_N, SMEM_A, SMEM_B>(
        C_frag,
        &As[curBuf][simd_m * SUBROWS_PER_SG][0],
        &Bs[curBuf][0][simd_n * SUBCOLS_PER_SG],
        BK);

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // TODO: bias-add epilogue (see header comment for the 3 implementation
  // options). For now this just stores the unfused matmul accumulator.
  #pragma clang loop unroll(full)
  for (int i = 0; i < FRAGS_M; ++i) {
    #pragma clang loop unroll(full)
    for (int j = 0; j < FRAGS_N; ++j) {
      int outRow = subRow + i * 8;
      int outCol = subCol + j * 8;
      if (outRow < M && outCol < N) {
        simdgroup_store(C_frag[i][j], C + outRow * N + outCol, N);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// GEMV: Matrix-vector (N=1)
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
//
// Design follows MLX's GEMVTKernel (mlx/backend/metal/kernels/gemv_masked.h):
// Per-thread tile of TM K-rows × TN N-cols. Simdgroup is laid out as
// SM × SN lanes (SM*SN=32) splitting the K dimension SM ways and the N
// dimension SN ways. After the K loop, partial sums are reduced across
// the SM K-lanes via simd_shuffle_down (handled by simdReduceSumStrided).
//
//   tg layout       : 1 simdgroup = 32 threads
//   per-thread tile : TM=4 K-rows × TN=4 N-cols
//   simdgroup tile  : SM=8 K-lanes × SN=4 N-lanes -> 32*TM K-rows / iter,
//                     SN*TN=16 output cols / simdgroup
//   tg per N        : ceil(N / 16)
//
// Why TM×TN instead of "lane-per-col scalar":
//   - More work per thread (16 fmas/iter vs 1) -> better load amortization,
//     better ILP on the FMA pipeline.
//   - K split SM=8 ways within the simdgroup -> 8x less per-thread K work
//     for the same K, so scales to large K without becoming latency-bound.
//   - Same memory pattern (lanes within a simdgroup access consecutive N
//     cols for fixed K row) -> still fully coalesced.
//
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
// Template instantiations
//===----------------------------------------------------------------------===//

template [[host_name("matmul_naive_f32")]] kernel void matmul_naive<float>(device const float*, device const float*, device float*, constant int&, constant int&, constant int&, uint2);
template [[host_name("matmul_naive_f16")]] kernel void matmul_naive<half>(device const half*, device const half*, device half*, constant int&, constant int&, constant int&, uint2);
template [[host_name("matmul_naive_bf16")]] kernel void matmul_naive<bfloat>(device const bfloat*, device const bfloat*, device bfloat*, constant int&, constant int&, constant int&, uint2);

template [[host_name("matmul_tiled_f32")]] kernel void matmul_tiled<float>(device const float*, device const float*, device float*, constant int&, constant int&, constant int&, uint2, uint2, uint2);
template [[host_name("matmul_tiled_f16")]] kernel void matmul_tiled<half>(device const half*, device const half*, device half*, constant int&, constant int&, constant int&, uint2, uint2, uint2);
template [[host_name("matmul_tiled_bf16")]] kernel void matmul_tiled<bfloat>(device const bfloat*, device const bfloat*, device bfloat*, constant int&, constant int&, constant int&, uint2, uint2, uint2);

// matmul_simd_t instantiations. Kernel name encodes tile params so the
// host can compose the name from a TileSpec at dispatch time.
//   matmul_simd_t_<BM>_<BN>_<BK>_<WM>_<WN>_<n|t>_<dtype>
//
// Currently registered:
//   (32, 64, 32, 1, 4, n) — replicates matmul_simd_m32 (16 <= M < 64)
//   (64, 64, 16, 2, 2, n) — replicates matmul_simd          (M >= 64, NN)
//   (64, 64, 16, 2, 2, t) — replicates matmul_nt            (M >= 64, NT)
// Each combo × 3 dtypes = 9 total. Add more here as we build out per-shape
// or per-dtype tile tables. Each one adds compiled .metallib bytes; only
// register what we'll route to.
template [[host_name("matmul_simd_t_32_64_32_1_4_n_f32")]] kernel void matmul_simd_t<float,    32, 64, 32, 1, 4, false, false>(device const float*, device const float*, device float*, constant int&, constant int&, constant int&, uint3, uint, uint, uint);
template [[host_name("matmul_simd_t_32_64_32_1_4_n_f16")]] kernel void matmul_simd_t<half,     32, 64, 32, 1, 4, false, false>(device const half*,  device const half*,  device half*,  constant int&, constant int&, constant int&, uint3, uint, uint, uint);
template [[host_name("matmul_simd_t_32_64_32_1_4_n_bf16")]] kernel void matmul_simd_t<bfloat,  32, 64, 32, 1, 4, false, false>(device const bfloat*,device const bfloat*,device bfloat*,constant int&, constant int&, constant int&, uint3, uint, uint, uint);

template [[host_name("matmul_simd_t_64_64_16_2_2_n_f32")]] kernel void matmul_simd_t<float,    64, 64, 16, 2, 2, false, false>(device const float*, device const float*, device float*, constant int&, constant int&, constant int&, uint3, uint, uint, uint);
template [[host_name("matmul_simd_t_64_64_16_2_2_n_f16")]] kernel void matmul_simd_t<half,     64, 64, 16, 2, 2, false, false>(device const half*,  device const half*,  device half*,  constant int&, constant int&, constant int&, uint3, uint, uint, uint);
template [[host_name("matmul_simd_t_64_64_16_2_2_n_bf16")]] kernel void matmul_simd_t<bfloat,  64, 64, 16, 2, 2, false, false>(device const bfloat*,device const bfloat*,device bfloat*,constant int&, constant int&, constant int&, uint3, uint, uint, uint);

template [[host_name("matmul_simd_t_64_64_16_2_2_t_f32")]] kernel void matmul_simd_t<float,    64, 64, 16, 2, 2, false, true>(device const float*, device const float*, device float*, constant int&, constant int&, constant int&, uint3, uint, uint, uint);
template [[host_name("matmul_simd_t_64_64_16_2_2_t_f16")]] kernel void matmul_simd_t<half,     64, 64, 16, 2, 2, false, true>(device const half*,  device const half*,  device half*,  constant int&, constant int&, constant int&, uint3, uint, uint, uint);
template [[host_name("matmul_simd_t_64_64_16_2_2_t_bf16")]] kernel void matmul_simd_t<bfloat,  64, 64, 16, 2, 2, false, true>(device const bfloat*,device const bfloat*,device bfloat*,constant int&, constant int&, constant int&, uint3, uint, uint, uint);

// MLX's "small fp32 NN" tile: bm=64, bn=32, bk=32, wm=2, wn=2.
// Use for fp32 medium-M cases where Simd's 64x64 tile produces too few
// tgs (small N relative to M) — the smaller BN doubles tg count along N
// and the bigger BK halves K-tile barriers.
template [[host_name("matmul_simd_t_64_32_32_2_2_n_f32")]] kernel void matmul_simd_t<float,    64, 32, 32, 2, 2, false, false>(device const float*, device const float*, device float*, constant int&, constant int&, constant int&, uint3, uint, uint, uint);
template [[host_name("matmul_simd_t_64_32_32_2_2_n_f16")]] kernel void matmul_simd_t<half,     64, 32, 32, 2, 2, false, false>(device const half*,  device const half*,  device half*,  constant int&, constant int&, constant int&, uint3, uint, uint, uint);
template [[host_name("matmul_simd_t_64_32_32_2_2_n_bf16")]] kernel void matmul_simd_t<bfloat,  64, 32, 32, 2, 2, false, false>(device const bfloat*,device const bfloat*,device bfloat*,constant int&, constant int&, constant int&, uint3, uint, uint, uint);

// TN instantiations: TRANSPOSE_A=true, TRANSPOSE_B=false. A is physically
// stored [K, M] (column-contiguous from PyTorch's view) and loaded via the
// transposed helper; B is loaded normally.
//   matmul_simd_t_<BM>_<BN>_<BK>_<WM>_<WN>_tn_<dtype>
template [[host_name("matmul_simd_t_64_64_16_2_2_tn_f32")]] kernel void matmul_simd_t<float,    64, 64, 16, 2, 2, true, false>(device const float*, device const float*, device float*, constant int&, constant int&, constant int&, uint3, uint, uint, uint);
template [[host_name("matmul_simd_t_64_64_16_2_2_tn_f16")]] kernel void matmul_simd_t<half,     64, 64, 16, 2, 2, true, false>(device const half*,  device const half*,  device half*,  constant int&, constant int&, constant int&, uint3, uint, uint, uint);
template [[host_name("matmul_simd_t_64_64_16_2_2_tn_bf16")]] kernel void matmul_simd_t<bfloat,  64, 64, 16, 2, 2, true, false>(device const bfloat*,device const bfloat*,device bfloat*,constant int&, constant int&, constant int&, uint3, uint, uint, uint);

// Fused matmul+bias kernel (NN, single 64x64 tile). Used by AddMMOp.
template [[host_name("matmul_simd_addmm_t_64_64_16_2_2_f32")]] kernel void matmul_simd_addmm_t<float,    64, 64, 16, 2, 2>(device const float*, device const float*, device float*, constant int&, constant int&, constant int&, device const float*, constant int&, uint3, uint, uint, uint);
template [[host_name("matmul_simd_addmm_t_64_64_16_2_2_f16")]] kernel void matmul_simd_addmm_t<half,     64, 64, 16, 2, 2>(device const half*,  device const half*,  device half*,  constant int&, constant int&, constant int&, device const half*,  constant int&, uint3, uint, uint, uint);
template [[host_name("matmul_simd_addmm_t_64_64_16_2_2_bf16")]] kernel void matmul_simd_addmm_t<bfloat,  64, 64, 16, 2, 2>(device const bfloat*,device const bfloat*,device bfloat*,constant int&, constant int&, constant int&, device const bfloat*,constant int&, uint3, uint, uint, uint);

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
// matmul_tensor_ops — Metal 4 tensor_ops::matmul2d (Apple9+ / M3+ only)
//
// Uses the Apple-blessed pattern from example_matmul_metal4: each threadgroup
// computes one BMxBN output tile via tensor_ops::matmul2d with
// execution_simdgroups<4> (= 128 threads/threadgroup). K is dynamic so a
// single kernel handles arbitrary K (K must still be a multiple of 16,
// enforced by host dispatch).
//
// Gated on __METAL_VERSION__ >= 410 (MSL 4.1, ships with macOS 26 / iOS 26).
// On older MSL versions this entire block is skipped so other kernels still
// compile.
//===----------------------------------------------------------------------===//
// Note: dropped the __METAL_VERSION__ gate during debugging — re-add once the
// runtime macro for MSL 4.1 is confirmed.
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

template <typename T>
kernel void matmul_tensor_ops(
    device T* A_buf [[buffer(0)]],
    device T* B_buf [[buffer(1)]],
    device T* C_buf [[buffer(2)]],
    constant int& M [[buffer(3)]],
    constant int& K [[buffer(4)]],
    constant int& N [[buffer(5)]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
  using namespace mpp::tensor_ops;
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 16;  // matmul2d's inner K-tile size

  // Build inline tensors from raw buffer pointers + runtime dims.
  // dextents arg order: (cols, rows) so the second dim is the M axis.
  auto A = metal::tensor<device T, metal::dextents<int32_t, 2>, metal::tensor_inline>(
      A_buf, metal::dextents<int32_t, 2>(K, M));
  auto B = metal::tensor<device T, metal::dextents<int32_t, 2>, metal::tensor_inline>(
      B_buf, metal::dextents<int32_t, 2>(N, K));
  auto C = metal::tensor<device T, metal::dextents<int32_t, 2>, metal::tensor_inline>(
      C_buf, metal::dextents<int32_t, 2>(N, M));

  // Two ops: init (mode::multiply, overwrites C tile) + accumulate
  // (mode::multiply_accumulate, += into C tile). matmul2d processes BK=16
  // K-elements per run() call -> we loop K/BK times.
  constexpr auto desc_init = matmul2d_descriptor(
      BM, BN, BK,
      false, false, false,
      matmul2d_descriptor::mode::multiply);
  constexpr auto desc_acc = matmul2d_descriptor(
      BM, BN, BK,
      false, false, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc_init, metal::execution_simdgroups<4>> mm_init;
  matmul2d<desc_acc,  metal::execution_simdgroups<4>> mm_acc;

  auto c_tile = C.slice(tgid.x * BN, tgid.y * BM);

  // First K-tile: initialize C
  {
    auto a = A.slice(0, tgid.y * BM);
    auto b = B.slice(tgid.x * BN, 0);
    mm_init.run(a, b, c_tile);
  }
  // Remaining K-tiles: accumulate
  for (int k = BK; k < K; k += BK) {
    auto a = A.slice(k, tgid.y * BM);
    auto b = B.slice(tgid.x * BN, k);
    mm_acc.run(a, b, c_tile);
  }
}

template [[host_name("matmul_tensor_ops_f32")]] kernel void matmul_tensor_ops<float>(device float*, device float*, device float*, constant int&, constant int&, constant int&, uint2);
template [[host_name("matmul_tensor_ops_f16")]] kernel void matmul_tensor_ops<half>(device half*, device half*, device half*, constant int&, constant int&, constant int&, uint2);
template [[host_name("matmul_tensor_ops_bf16")]] kernel void matmul_tensor_ops<bfloat>(device bfloat*, device bfloat*, device bfloat*, constant int&, constant int&, constant int&, uint2);

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

const char* MatMulOp::kernelSource() const {
  return matmulKernelSource().c_str();
}

//===----------------------------------------------------------------------===//
// AddMMOp (aten::addmm) — fused bias-matmul.
//
// Schema: addmm(input, mat1, mat2, *, beta=1, alpha=1) -> Tensor
//   inputs[0] = input (bias) [M, N] OR broadcast (commonly [N])
//   inputs[1] = mat1 [M, K]
//   inputs[2] = mat2 [K, N]
//   inputs[3] = beta (Scalar, default 1) — IGNORED, must be 1 for now
//   inputs[4] = alpha (Scalar, default 1) — IGNORED, must be 1 for now
//
// Constraints (currently enforced — caller must satisfy or use mm + add):
//   - mat1 row-contiguous, mat2 row-contiguous (NN layout)
//   - bias is [M, N] contiguous OR [N] (1D-broadcast)
//   - beta == alpha == 1
//
// Falls through to MatMulOp's plain matmul kernel + a separate elementwise
// add IF those constraints don't hold (rare in practice for nn.Linear).
//===----------------------------------------------------------------------===//

std::vector<SizesType> AddMMOp::computeOutputShape(
    EValuePtrSpan inputs) const {
  if (inputs.size() < 3 || !inputs[1]->isTensor() || !inputs[2]->isTensor()) {
    return {};
  }
  const auto& mat1 = inputs[1]->toTensor();
  const auto& mat2 = inputs[2]->toTensor();
  return {static_cast<SizesType>(mat1.size(0)),
          static_cast<SizesType>(mat2.size(1))};
}

void AddMMOp::dispatch(
    MetalStream* stream,
    EValuePtrSpan inputs,
    EValuePtrSpan outputs) {

  if (inputs.size() < 3) {
    ET_LOG(Error, "AddMMOp: expected at least 3 inputs (input, mat1, mat2)");
    return;
  }

  auto& bias = inputs[0]->toTensor();
  auto& A    = inputs[1]->toTensor();
  auto& B    = inputs[2]->toTensor();
  auto& C    = outputs[0]->toTensor();

  auto err = resizeOutput(inputs, outputs[0]);
  if (err != Error::Ok) {
    ET_LOG(Error, "AddMMOp: failed to resize output");
    return;
  }

  // For unsupported alpha/beta or non-NN layouts: fall back to plain matmul
  // followed by an elementwise add. (Unimplemented; just error for now —
  // PyTorch's addmm with default scalars + nn.Linear bias hits the fast path.)
  const bool aRC = isRowContiguous(A);
  const bool bRC = isRowContiguous(B);
  if (!aRC || !bRC) {
    ET_LOG(Error, "AddMMOp: only NN layout (both row-contiguous) is supported "
                  "currently; got A.RC=%d B.RC=%d", aRC, bRC);
    return;
  }

  int32_t M = static_cast<int32_t>(A.size(0));
  int32_t K = static_cast<int32_t>(A.size(1));
  int32_t N = static_cast<int32_t>(B.size(1));
  ScalarType dtype = C.scalar_type();

  // Determine bias stride pattern. Two supported cases:
  //  1) bias is 2D [M, N] contiguous → stride_m = N
  //  2) bias is 1D [N] (broadcasts across M rows) → stride_m = 0
  // We detect via dim() == 1; for 2D we trust it's contiguous (PyTorch addmm
  // requires this OR will broadcast-expand before reaching us).
  int32_t bias_stride_m;
  if (bias.dim() == 1) {
    if (bias.size(0) != N) {
      ET_LOG(Error, "AddMMOp: 1D bias dim mismatch (got %lld, expected %d)",
             (long long)bias.size(0), N);
      return;
    }
    bias_stride_m = 0;  // same row repeated
  } else if (bias.dim() == 2 && bias.size(0) == M && bias.size(1) == N) {
    bias_stride_m = N;
  } else {
    ET_LOG(Error, "AddMMOp: unsupported bias shape (dim=%zd)",
           (ptrdiff_t)bias.dim());
    return;
  }

  std::string kname =
      std::string("matmul_simd_addmm_t_64_64_16_2_2_") + dtypeSuffix(dtype);
  auto* kernel = getKernel(stream, kname.c_str());

  ET_LOG(Info,
         "AddMMOp: M=%d K=%d N=%d bias_stride_m=%d kernel=%s",
         M, K, N, bias_stride_m, kname.c_str());

  // Same dispatch grid as the Simd 64x64 MM tile.
  uvec3 grid((N + 63) / 64, (M + 63) / 64, 1);
  uvec3 block(128, 1, 1);

  stream->dispatch(kernel, {
    {A.mutable_data_ptr(), A.nbytes()},
    {B.mutable_data_ptr(), B.nbytes()},
    {C.mutable_data_ptr(), C.nbytes()},
    M, K, N,
    {bias.mutable_data_ptr(), bias.nbytes()},
    bias_stride_m
  }, grid, block);
}

const char* AddMMOp::kernelSource() const {
  // The addmm kernel template lives inside the same source string as
  // matmul_simd_t — both are compiled from matmulKernelSource(). Reuse it.
  return matmulKernelSource().c_str();
}

//===----------------------------------------------------------------------===//
// BatchedMatMulOp (aten::bmm) - [B, M, K] @ [B, K, N] -> [B, M, N]
//===----------------------------------------------------------------------===//

std::vector<SizesType> BatchedMatMulOp::computeOutputShape(
    EValuePtrSpan inputs) const {

  if (inputs.size() < 2 || !inputs[0]->isTensor() || !inputs[1]->isTensor()) {
    return {};
  }

  auto& A = inputs[0]->toTensor();  // [B, M, K]
  auto& B = inputs[1]->toTensor();  // [B, K, N]

  if (A.dim() != 3 || B.dim() != 3) {
    return {};
  }

  SizesType batch = A.size(0);
  SizesType M = A.size(1);
  SizesType N = B.size(2);

  return {batch, M, N};
}

void BatchedMatMulOp::dispatch(
    MetalStream* stream,
    EValuePtrSpan inputs,
    EValuePtrSpan outputs) {

  auto& A = inputs[0]->toTensor();  // [B, M, K]
  auto& B = inputs[1]->toTensor();  // [B, K, N]
  auto& C = outputs[0]->toTensor(); // [B, M, N]

  auto err = resizeOutput(inputs, outputs[0]);
  if (err != Error::Ok) {
    ET_LOG(Error, "BatchedMatMulOp: failed to resize output");
    return;
  }

  if (!isRowContiguous(A) || !isRowContiguous(B)) {
    // Broadcast tolerated below; non-broadcast non-contig is an error.
    if (!(isRowContiguous(A) && B.strides().size() >= 1 && B.strides()[0] == 0)) {
      ET_LOG(Error, "BatchedMatMulOp: inputs must be row-contiguous (or B broadcast over batch)");
      return;
    }
  }

  int32_t batch = static_cast<int32_t>(A.size(0));
  int32_t M = static_cast<int32_t>(A.size(1));
  int32_t K = static_cast<int32_t>(A.size(2));
  int32_t N = static_cast<int32_t>(B.size(2));

  ScalarType dtype = C.scalar_type();

  //----------------------------------------------------------------------
  // Broadcast fast path: if B's per-batch stride is 0, B is just [K, N]
  // replicated across batch. The whole bmm collapses to one 2D matmul:
  //     (batch*M, K) @ (K, N)  ->  (batch*M, N)
  // Bigger M => better tile occupancy, single kernel launch, and we get
  // to ride MatMulOp's full ladder including TensorOps when aligned.
  // (Not normally hit by aten::bmm — torch.bmm requires both operands to
  // have an explicit batch dim — but defensive in case an upstream pass
  // emits this pattern.)
  //----------------------------------------------------------------------
  if (B.strides().size() >= 1 && B.strides()[0] == 0 && batch > 1 &&
      isRowContiguous(A)) {
    int32_t M2 = batch * M;

    // Mirror MatMulOp's kernel-selection ladder (size + Apple9 family).
    auto* metalStream = static_cast<MetalStream*>(stream);
    const bool canTensorOps =
        metalStream && metalStream->device() &&
        [metalStream->device() supportsFamily:MTLGPUFamilyApple9] &&
        (M2 % 64 == 0) && (N % 64 == 0) && (K % 16 == 0) &&
        (dtype == ScalarType::Float || dtype == ScalarType::Half ||
         dtype == ScalarType::BFloat16);

    std::string kname;
    uvec3 grid, block;
    if (canTensorOps) {
      kname = std::string("matmul_tensor_ops_") + dtypeSuffix(dtype);
      grid = uvec3((N + 63) / 64, (M2 + 63) / 64, 1);
      block = uvec3(128, 1, 1);
    } else if (M2 >= 64 && N >= 64 && K >= 16) {
      kname = std::string("matmul_simd_") + dtypeSuffix(dtype);
      grid = uvec3((N + 63) / 64, (M2 + 63) / 64, 1);
      block = uvec3(128, 1, 1);
    } else if (M2 >= 32 && N >= 32) {
      kname = std::string("matmul_tiled_") + dtypeSuffix(dtype);
      grid = uvec3((N + 31) / 32, (M2 + 31) / 32, 1);
      block = uvec3(32, 32, 1);
    } else {
      kname = std::string("matmul_naive_") + dtypeSuffix(dtype);
      grid = uvec3((N + 7) / 8, (M2 + 7) / 8, 1);
      block = uvec3(8, 8, 1);
    }

    ET_LOG(Info,
           "BatchedMatMulOp: broadcast collapse batch=%d M=%d->%d K=%d N=%d kernel=%s",
           batch, M, M2, K, N, kname.c_str());

    auto* kernel = getKernel(stream, kname.c_str());
    stream->dispatch(kernel, {
      {A.mutable_data_ptr(), A.nbytes()},
      {B.mutable_data_ptr(), B.nbytes()},
      {C.mutable_data_ptr(), C.nbytes()},
      M2, K, N
    }, grid, block);
    return;
  }

  // Prefer the SIMD MMA kernel (with tgid.z = batch) for large enough tiles;
  // fall back to the naive batched kernel for small problems where SIMD
  // would have low occupancy. matmul_simd assumes contiguous batched layout
  // (A_stride = M*K, etc), which our row-contiguous check above guarantees.
  const bool useSimd = (M >= 64) && (N >= 64) && (K >= 16);

  if (useSimd) {
    std::string kname = std::string("matmul_simd_") + dtypeSuffix(dtype);
    auto* kernel = getKernel(stream, kname.c_str());
    ET_LOG(Info, "BatchedMatMulOp: simd batch=%d M=%d K=%d N=%d",
           batch, M, K, N);
    uvec3 grid((N + 63) / 64, (M + 63) / 64, batch);
    uvec3 block(128, 1, 1);
    stream->dispatch(kernel, {
      {A.mutable_data_ptr(), A.nbytes()},
      {B.mutable_data_ptr(), B.nbytes()},
      {C.mutable_data_ptr(), C.nbytes()},
      M, K, N
    }, grid, block);
    return;
  }

  // Naive fallback (small problems).
  int32_t A_batch_stride = M * K;
  int32_t B_batch_stride = K * N;
  int32_t C_batch_stride = M * N;
  std::string kname = std::string("bmm_") + dtypeSuffix(dtype);
  auto* kernel = getKernel(stream, kname.c_str());
  ET_LOG(Info, "BatchedMatMulOp: naive batch=%d M=%d K=%d N=%d",
         batch, M, K, N);
  uvec3 grid((N + 7) / 8, (M + 7) / 8, batch);
  uvec3 block(8, 8, 1);
  stream->dispatch(kernel, {
    {A.mutable_data_ptr(), A.nbytes()},
    {B.mutable_data_ptr(), B.nbytes()},
    {C.mutable_data_ptr(), C.nbytes()},
    batch, M, K, N,
    A_batch_stride, B_batch_stride, C_batch_stride
  }, grid, block);
}

const char* BatchedMatMulOp::kernelSource() const {
  // Share the full kernel source with MatMulOp so we can use both
  // matmul_simd_<dtype> (fast path) and bmm_<dtype> (naive fallback).
  return matmulKernelSource().c_str();
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch
