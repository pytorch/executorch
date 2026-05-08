/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MatMulOp.h"
#include <executorch/backends/metal/ops/MatMulKernels.metal.h>
#include <executorch/backends/metal/ops/MatMulCommon.h>
#include <executorch/backends/metal/ops/MatMulMlxJit.h>
#include <executorch/backends/metal/ops/mlx_jit/KernelLoader.h>
#include <executorch/backends/metal/ops/registry/OpUtils.h>
#include <executorch/backends/metal/core/MetalStream.h>
#include <executorch/backends/metal/core/MetalDeviceInfo.h>
#include <executorch/backends/metal/kernels/TileLoad.h>
#include <executorch/backends/metal/ops/MPSGraphOp.h>
#include <executorch/runtime/platform/log.h>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>

namespace executorch {
namespace backends {
namespace metal_v2 {

using runtime::Error;
//===----------------------------------------------------------------------===//
// File layout (~2k lines):
//   §1  HOST-SIDE OP CLASSES                                  (this section)
//        §1a  shared host helpers (FCs, tile pickers)         lines ~60-150
//        §1b  MatMulOp     — aten::mm / linear (un-fused)     lines ~155-370
//        §1c  AddMMOp      — aten::addmm (matmul + bias)
//        §1d  BAddBMMOp    — aten::baddbmm (3D matmul + bias)
//        §1e  BatchedMatMulOp — aten::bmm
//   §2  KERNEL SOURCE STRING (matmulKernelSource)             lines ~370-1440
//        Single Metal raw-string literal compiled at runtime via
//        MetalKernelCompiler. Internal layout (free helpers, then kernels):
//          §2a  applesimd::frag_coord / swizzle_tile          (Apple lane mapping)
//          §2b  Epilogue functors (None, AxpbyBias)
//          §2c  storeFragWithEpilogue (per-lane bounds-safe store)
//          §2d  simdMMAKTile (one K-tile of MMA, mixed-precision)
//          §2e  BlockLoader (cooperative tile load; from TileLoad.h)
//          §2f  matmul_naive / matmul_tiled                   (small/medium fallbacks)
//          §2g  matmul_simd_addmm_t                           (the unified SIMD-MMA kernel —
//                handles both un-fused matmul AND fused matmul+bias via the
//                use_out_source / do_axpby function constants — see §2g header
//                comment for the full FC matrix)
//          §2h  gemv / gemv_t                                  (M=1 / N=1 fast paths)
//          §2i  matmul_tensor_ops                              (Apple9+ tensor_ops::matmul2d)
//          §2j  bmm_<dtype>                                    (small-batch naive batched)
//          §2k  template [[host_name]] instantiations          (PSO entry points)
//   §3  KERNEL-SOURCE ACCESSORS  (kernelSource() one-liners)   lines ~1440 onward
// All four op classes share the same matmulKernelSource(); the unified
// matmul_simd_addmm_t is the workhorse — its (align_M, align_N, align_K,
// use_out_source, do_axpby) function-constant tuple specializes it per
// dispatch (~32 PSO variants per (tile, layout, dtype)).
// Function constants (declared at top of MSL string, indices 0-4):
//   0: align_M       — M % BM == 0   (DCEs un-aligned safe-load paths)
//   1: align_N       — N % BN == 0
//   2: align_K       — K % BK == 0
//   3: use_out_source — true ⇒ apply bias epilogue; false ⇒ identity
//   4: do_axpby      — true ⇒ use kernel-passed alpha/beta; false ⇒ alpha=beta=1
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Output Shape
//===----------------------------------------------------------------------===//

std::vector<SizesType> MatMulOp::computeOutputShape(
    ::executorch::runtime::Span<::executorch::runtime::EValue*> inputs) const {

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
// Picks among Naive / Tiled / Simd / NT / TN / GEMV / GEMV_T based on:
//   - input layout (row-contig vs col-contig, where col-contig means the
//     tensor is a .T view of an underlying row-contig tensor)
//   - problem size (M, N, K)
//   - device tier (smaller thresholds on phones, larger on Ultra/Max)
//===----------------------------------------------------------------------===//

std::string MatMulOp::kernelTypePrefix(MatMulKernelType type) const {
  // Returns the local big-source kernel-name prefix for the kernel types
  // that still use the big-source path (Naive / Tiled / GEMV / GEMV_T).
  // The Simd-MMA family (Simd*/NT/TN) and the NAX/SplitK paths route
  // through MLX JIT directly and don't use this prefix.
  switch (type) {
    case MatMulKernelType::Naive:           return "matmul_naive";
    case MatMulKernelType::Tiled:           return "matmul_tiled";
    case MatMulKernelType::GEMV:            return "gemv";
    case MatMulKernelType::GEMV_T:          return "gemv_t";
    default:
      // Simd*/NT/TN/Mlx_Dense_NAX/Simd_SplitK/SplitK_NAX should be
      // dispatched through their MLX-JIT helpers without ever consulting
      // this map. If we end up here, the dispatch switch is incomplete.
      ET_CHECK_MSG(false, "MatMulOp::kernelTypePrefix: unhandled type %d "
                          "(should have been routed through MLX JIT)",
                   int(type));
      return "matmul_naive";
  }
}

// selectKernel only handles the size-based fallback ladder for the regular
// (NN) case. NT/TN/GEMV/GEMV_T are picked separately based on input layout.
// Tier ladder:
//   Simd     : M >= 64. 64x64 output, 4 sg in 2x2.
//   Simd_M32 : 16 <= M < 64. 32x64 output, 4 sg in 1x4. MLX-style "skinny"
//              variant for prefill batches like Llama M=32.
//   Tiled    : 32 <= M < 16 (rare middle ground), local-kernel fallback.
//   Naive    : everything smaller.
// Variants compiled but NOT auto-routed (kept for future use / experimentation):
// - Simd_M32_BN128: tried for compute-bound large-K cases. Theoretical AI
//   gain (10.7 -> 12.8 FLOPs/byte) is real, but in practice doubling the
//   per-sg register pressure (16 vs 8 simdgroup_matrix accumulators) and
//   threadgroup memory cuts wave-level occupancy by more than the AI gain
//   buys. Net regression on Apple M-series. Could be reconsidered if we
//   add register-blocked variants or tune for specific GPU families.
// - Simd_M32_SplitK: didn't help compute-bound cases (the bottleneck is
//   arithmetic intensity, not parallelism).
MatMulKernelType MatMulOp::selectKernel(int64_t M, int64_t N, int64_t K,
                                        ScalarType dtype) const {
  // M==1 fast path → gemv_t.
  if (M == 1 && N >= 1 && K >= 1) {
    return MatMulKernelType::GEMV_T;
  }

  // Below the simd-MMA path's lower bound: keep the tiled / naive ladder.
  if (M < 64 || N < 64 || K < 16) {
    if (M >= 2 && N >= 64 && K >= 16) return MatMulKernelType::Simd_M32;
    if (M >= 32 && N >= 32) return MatMulKernelType::Tiled;
    return MatMulKernelType::Naive;
  }

  // MLX-style SIMD split-K precondition (mlx/backend/metal/matmul.cpp:935-940):
  //   batch == 1                                     (MatMulOp is non-batched)
  //   _tm·_tn ≤ min_tmn_threshold                    (output tile count)
  //   _tk ≥ 8                                        (≥ 8 K-blocks of bk=16 → K ≥ 128)
  //   K ≥ max(M, N)                                  (deep-K regime)
  // We also gate K%16==0 because we only emit the K-aligned partial-kernel
  // variants (MLX uses a separate residual loop for the last partition;
  // we skip that to keep PSO count down).
  // Threshold per MLX: 2048 on Max/Ultra ('s'/'d'), 1024 on medium / phone.
  // Phase C: NAX split-K precondition (mlx/backend/metal/matmul.cpp:962-966):
  //   half/bf16 only (fp32 only with TF32, which we don't enable)
  //   M·N ≥ 4 194 304   (≥ 2048·2048)
  //   K ≥ 10240
  //   K ≥ 3·max(M, N)
  //   batch == 1
  //   Apple9+ (M3+) family — checked at dispatch time.
  // NAX is more specific than SIMD split-K so check it first.
  {
    const auto preTier = MetalDeviceInfo::tier();

    // NAX split-K (Apple9+ check happens at dispatch via supportsFamily;
    // here we only filter by shape + dtype). MLX 0.31.2 also gates fp32
    // NAX behind env::enable_tf32() (truncated mantissa); we mirror that
    // via MLX_ENABLE_TF32=1.
    const bool nax_dtype_ok = (dtype == ScalarType::Half ||
                               dtype == ScalarType::BFloat16 ||
                               (dtype == ScalarType::Float && tf32Enabled()));
    const int64_t mn = M * N;
    if (nax_dtype_ok &&
        mn >= int64_t(2048) * 2048 &&
        K >= 10240 &&
        K >= 3 * std::max(M, N)) {
      // Apple9 device check happens in dispatch — for non-Apple9 we'll
      // fall back below. Do the supportsFamily check here too to avoid
      // returning SplitK_NAX on unsupported HW.
      auto* metalStream = static_cast<MetalStream*>(MetalStream::get());
      if (metalStream && metalStream->device() &&
          [metalStream->device() supportsFamily:MTLGPUFamilyApple9]) {
        return MatMulKernelType::SplitK_NAX;
      }
    }

    // Mlx_Dense_NAX (MLX 0.31.2 gemm_fused_nax) — fires when NAX
    // precondition holds AND split-K-NAX precondition didn't. MLX 0.31.2
    // routes most matmuls through this dense NAX path (matmul.cpp:983-1009).
    // NN layout only — NT/TN dense NAX not ported.
    {
      if (nax_dtype_ok) {
        auto* metalStream = static_cast<MetalStream*>(MetalStream::get());
        if (metalStream && metalStream->device() &&
            [metalStream->device() supportsFamily:MTLGPUFamilyApple9]) {
          return MatMulKernelType::Mlx_Dense_NAX;
        }
      }
    }

    const int min_tmn_threshold =
        (preTier == DeviceTier::MacUltra) ? 2048 : 1024;
    const int64_t _tm = (M + 16 - 1) / 16;
    const int64_t _tn = (N + 16 - 1) / 16;
    const int64_t _tk = K / 16;
    if (_tm * _tn <= min_tmn_threshold && _tk >= 8 &&
        K >= std::max(M, N) && (K % 16) == 0) {
      return MatMulKernelType::Simd_SplitK;
    }
  }

  // MLX's GEMM_TPARAM_MACRO transcribed verbatim from
  // mlx/backend/metal/matmul.cpp:88-169. NN-only (NT/TN go through
  // separate code paths above this function), so MLX's `nt` branches are
  // dead-coded out. Device-class mapping:
  //   'g'/'p' (small Apple Silicon) → DeviceTier::Phone
  //   'd'     (Mac Max/Ultra)       → DeviceTier::MacUltra
  //   else    (M-base/Pro, 's', 'c')→ DeviceTier::MacBase  (defaults only)
  // Tile-substitution: we don't have MLX's (32, 64, 16, 1, 2) tile, so
  // when MLX would pick it (large 'd' nn deep-K half/bf, small 'd' fp32
  // nt) we fall back to Simd_M32 = (32, 64, 32, 1, 4). Same BM/BN, larger
  // BK + double WN — closest available shape.
  const bool fp32 = (dtype == ScalarType::Float);
  const auto tier = MetalDeviceInfo::tier();

  // MLX defaults → Simd (64, 64, 16, 2, 2).
  MatMulKernelType k = MatMulKernelType::Simd;

  if (tier == DeviceTier::Phone) {
    if (!fp32) k = MatMulKernelType::Simd_W12;            // 64,64,16,1,2
    // else fp32 nn → defaults (Simd).
  } else if (tier == DeviceTier::MacUltra) {
    const int64_t area = M * N;
    if (area >= (int64_t(1) << 20)) {
      // large
      if (!fp32) {
        if (2 * std::max(M, N) > K) {
          k = MatMulKernelType::Simd_W12;                 // 64,64,16,1,2
        } else {
          k = MatMulKernelType::Simd_W12_M32;             // 32,64,16,1,2 (real, was Simd_M32 substitute)
        }
      }
      // else fp32 large 'd' → defaults (Simd).
    } else {
      // smaller
      if (!fp32) {
        k = MatMulKernelType::Simd_W12;                   // 64,64,16,1,2
      } else {
        k = MatMulKernelType::Simd_BN32;                  // 64,32,32,2,2
      }
    }
  }
  // else: DeviceTier::MacBase (medium) → defaults (Simd).
  return k;
}

//===----------------------------------------------------------------------===//
// SIMD split-K dispatch helper.
// MLX-faithful translation of `steel_matmul_splitk_axpby` in
//   mlx/backend/metal/matmul.cpp:529-680
// (specifically the non-axpby case: alpha=1, beta=0, no bias).
// Two-stage dispatch:
//   1. Partial:  matmul_simd_splitk_partial_<bm>_<bn>_16_2_2_<mna>_<dtype>
//                Grid = (tn, tm, P), block = (32, 2, 2) = 128 threads.
//                Writes to a freshly-allocated [P, M, N] fp32 buffer.
//   2. Accum:    matmul_simd_splitk_accum_<dtype>
//                Grid = (N, M, 1), block = (32, 1, 1).
//                Reduces partial[0..P-1, m, n] → out[m, n] in T.
// Tile sizes per MLX (matmul.cpp:550-557):
//   bm = M < 40 ? 16 : 32,   bn = N < 40 ? 16 : 32,   bk = 16
// Partition count (matmul.cpp:560-561):
//   P = clamp(next_pow2(_tk / (_tm * _tn)), 2, 32)
// Caller has already validated split-K eligibility via selectKernel.
//===----------------------------------------------------------------------===//
namespace {
inline int next_pow2(int v) {
  if (v <= 1) return 1;
  int p = 1;
  while (p < v) p <<= 1;
  return p;
}
}  // namespace

namespace {
// Reuse the shared MLX-JIT helpers (struct mirrors + helpers). These used
// to live inline in this anon namespace but were moved to MatMulMlxJit.h
// when AddMM-family ops needed them too (Tier 4). The using-declarations
// below keep all this file's call sites working without textual changes.
using mlx_jit_helpers::GEMMParamsHost;
using mlx_jit_helpers::GEMMSplitKParamsHost;
using mlx_jit_helpers::GEMMAddMMParamsHost;
using mlx_jit_helpers::toJitDtype;
using mlx_jit_helpers::buildBaseName;
using mlx_jit_helpers::buildFusedHashSuffix;
using mlx_jit_helpers::buildSplitKNaxHashSuffix;
using mlx_jit_helpers::makeMlxFusedFCs;
using mlx_jit_helpers::makeMlxSplitKNaxFCs;

// Dispatch the SIMD-MMA split-K *partial* via MLX 0.31.2's `gemm_splitk`
// template (steel_gemm_splitk.h). The accum is intentionally kept on our
// local `matmul_simd_splitk_accum_*` kernel — MLX's gemm_splitk_accum is
// designed for `dispatch_threads` (Metal's exact-thread-count dispatch),
// while MetalStream only exposes `dispatch_threadgroups`. Under
// dispatch_threadgroups the MLX accum's `[[thread_position_in_grid]]`
// would OOB whenever N % 32 != 0 (no host-side bounds-check); routing
// only the partial avoids that landmine for now.
// Caller pre-computes (bm, bn, P, partition_size, partition_stride,
// gemm_k_iterations) from MLX's heuristic in matmul.cpp:550-561.
// Allocates + frees the fp32 partial buffer internally.
inline void dispatchSimdSplitKPartialViaMlxJit(
    MetalStream* stream,
    const executorch::aten::Tensor& A,
    const executorch::aten::Tensor& B,
    void* partial_ptr, size_t partial_bytes,
    int32_t M, int32_t K, int32_t N,
    int bm, int bn, int bk, int wm, int wn,
    int P,
    int split_k_partition_size,
    int split_k_partition_stride,
    int gemm_k_iterations,
    bool mn_aligned, bool k_aligned,
    executorch::aten::ScalarType dtype) {
  const auto jdtIn = toJitDtype(dtype);
  const auto jdtOut = mlx_jit::JitDtype::Float32;
  const char* aTname = mlx_jit::typeToName(jdtIn);
  const char* outTname = mlx_jit::typeToName(jdtOut);

  // MLX-style kernel name. mn_aligned and k_aligned are TEMPLATE args
  // (not FCs) for gemm_splitk, so they MUST be in the cache key —
  // distinct (mn × k_aligned) combos compile to separate libraries.
  std::ostringstream kn;
  kn << "steel_gemm_splitk_nn_" << aTname << "_" << outTname
     << "_bm" << bm << "_bn" << bn << "_bk" << bk
     << "_wm" << wm << "_wn" << wn
     << "_mn_aligned_" << (mn_aligned ? 't' : 'n')
     << "_k_aligned_" << (k_aligned ? 't' : 'n');
  const std::string baseName = kn.str();

  ET_LOG(Debug,
         "MatMulOp[simd_splitk/mlx_jit]: M=%d K=%d N=%d dtype=%s "
         "tile=(%d,%d,%d,%d,%d) P=%d mn_aligned=%d k_aligned=%d kname=%s",
         M, K, N, dtypeSuffix(dtype), bm, bn, bk, wm, wn, P,
         int(mn_aligned), int(k_aligned), baseName.c_str());

  auto pso = mlx_jit::shared(stream->compiler())
                 .getSplitKKernel(baseName, jdtIn, jdtOut,
                                  /*ta=*/false, /*tb=*/false,
                                  bm, bn, bk, wm, wn,
                                  mn_aligned, k_aligned);
  ET_CHECK_MSG(pso != nil,
               "MatMulOp::dispatchSimdSplitKPartialViaMlxJit: "
               "getSplitKKernel returned nil for '%s'", baseName.c_str());

  // GEMMSpiltKParams (MLX's struct, typo preserved). NN row-major
  // layout: lda=K, ldb=N, ldc=N. swizzle_log=0 for SIMD split-K
  // (matches matmul.cpp's steel_matmul_splitk_axpby which doesn't
  // swizzle the partial-kernel grid).
  const int _tn = (N + bn - 1) / bn;
  const int _tm = (M + bm - 1) / bm;
  GEMMSplitKParamsHost params{
      /*M=*/M, /*N=*/N, /*K=*/K,
      /*lda=*/K, /*ldb=*/N, /*ldc=*/N,
      /*tiles_n=*/_tn, /*tiles_m=*/_tm,
      /*split_k_partitions=*/P,
      /*split_k_partition_stride=*/split_k_partition_stride,
      /*split_k_partition_size=*/split_k_partition_size,
      /*swizzle_log=*/0,
      /*gemm_k_iterations_aligned=*/gemm_k_iterations,
  };

  // partial = OUTPUT of split-K phase A. Use hazard-aware setOutputBuffer
  // so phase B (which reads `partial`) sees a RAW edge.
  auto bo = stream->allocator().bufferForPtr(partial_ptr, partial_bytes);

  // Grid + block per matmul.cpp's steel_matmul_splitk_axpby
  // (3D grid, 32 × WN × WM block).
  stream->recorder().beginDispatch(pso)
      .setInput(0, A.const_data_ptr(), A.nbytes())
      .setInput(1, B.const_data_ptr(), B.nbytes())
      .setOutputBuffer(2, bo.mtl, bo.offset, partial_bytes)
      .setBytes(3, &params, sizeof(params))
      .run(uvec3{uint32_t(_tn), uint32_t(_tm), uint32_t(P)},
           uvec3{32u, uint32_t(wn), uint32_t(wm)});
}

//===----------------------------------------------------------------------===//
// Tier 1 — Simd* family routed through MLX 0.31.2 `gemm` template via
// per-shape JIT (steel/gemm/kernels/steel_gemm_fused.h). The host-side
// kernel selection (which (BM, BN, BK, WM, WN) tile + which kernelType)
// is unchanged — we just swap the kernel-acquisition + binding layer.
//===----------------------------------------------------------------------===//

// Mapping from our MatMulKernelType enum → the (BM, BN, BK, WM, WN) tile
// each variant has historically been hardcoded to. Centralized so callers
// can't drift.
struct SimdTileShape {
  int BM, BN, BK, WM, WN;
};

inline SimdTileShape simdTileFor(MatMulKernelType kt) {
  switch (kt) {
    case MatMulKernelType::Simd:        return {64, 64, 16, 2, 2};
    case MatMulKernelType::Simd_BN32:   return {64, 32, 32, 2, 2};
    case MatMulKernelType::Simd_M32:    return {32, 64, 32, 1, 4};
    case MatMulKernelType::Simd_W12:    return {64, 64, 16, 1, 2};
    case MatMulKernelType::Simd_W12_M32:return {32, 64, 16, 1, 2};
    // NT/TN historically use the (64,64,16,2,2) tile too. Tier 2 will
    // route them through the JIT path; until then this mapping is unused
    // for the transposed cases.
    case MatMulKernelType::NT:          return {64, 64, 16, 2, 2};
    case MatMulKernelType::TN:          return {64, 64, 16, 2, 2};
    default: return {64, 64, 16, 2, 2};
  }
}

// Dispatch a SIMD-MMA dense GEMM via MLX 0.31.2's `gemm` template
// (steel_gemm_fused.h). Shape-and-tile selection is the caller's
// responsibility (matches MatMulOp's existing per-kernelType tile
// hardcoding); this helper only swaps in MLX's kernel + ABI.
// Caller already knows (BM, BN, BK, WM, WN) and transpose flags.
// FCs match MLX upstream slot numbers (10/100/110/200/201/202). use_out_source
// + do_axpby + has_batch are fixed false here (un-fused matmul, no bias,
// non-batched). The split-K variant uses a different helper.
inline void dispatchSimdViaMlxJit(
    MetalStream* stream,
    const executorch::aten::Tensor& A,
    const executorch::aten::Tensor& B,
    executorch::aten::Tensor& C,
    int32_t M, int32_t K, int32_t N,
    int BM, int BN, int BK, int WM, int WN,
    int32_t swizzle_log,
    bool transpose_a, bool transpose_b,
    executorch::aten::ScalarType dtype) {
  const auto jdt = toJitDtype(dtype);
  const char* tname = mlx_jit::typeToName(jdt);
  const std::string baseName = buildBaseName(
      "steel_gemm_fused", transpose_a, transpose_b,
      tname, tname, BM, BN, BK, WM, WN);

  const bool has_batch = false;
  const bool use_out_source = false;
  const bool do_axpby = false;
  const bool align_M = (M % BM) == 0;
  const bool align_N = (N % BN) == 0;
  const bool align_K = (K % BK) == 0;

  const std::string hashName = baseName + buildFusedHashSuffix(
      has_batch, use_out_source, do_axpby, align_M, align_N, align_K);
  const auto fcs = makeMlxFusedFCs(
      has_batch, use_out_source, do_axpby, align_M, align_N, align_K);

  ET_LOG(Debug,
         "MatMulOp[simd/mlx_jit]: M=%d K=%d N=%d dtype=%s "
         "tile=(%d,%d,%d,%d,%d) ta=%d tb=%d swizzle_log=%d kname=%s",
         M, K, N, dtypeSuffix(dtype), BM, BN, BK, WM, WN,
         transpose_a, transpose_b, swizzle_log, baseName.c_str());

  auto pso = mlx_jit::shared(stream->compiler())
                 .getDenseGemmKernel(baseName, hashName, fcs, jdt,
                                     transpose_a, transpose_b,
                                     BM, BN, BK, WM, WN);
  ET_CHECK_MSG(pso != nil,
               "MatMulOp::dispatchSimdViaMlxJit: getDenseGemmKernel returned "
               "nil for '%s'", baseName.c_str());

  // GEMMParams layout. Leading dimensions follow MLX's convention:
  //   transpose_a=false → A is row-major [M, K] → lda = K
  //   transpose_a=true  → A.T is row-major [K, M] → lda = M
  //   transpose_b=false → B is row-major [K, N] → ldb = N
  //   transpose_b=true  → B.T is row-major [N, K] → ldb = K
  //   D is always row-major [M, N] → ldd = N
  const int lda = transpose_a ? M : K;
  const int ldb = transpose_b ? K : N;
  const int ldd = N;
  const int tilesN = (N + BN - 1) / BN;
  const int tilesM = (M + BM - 1) / BM;

  GEMMParamsHost params{
      /*M=*/M, /*N=*/N, /*K=*/K,
      /*lda=*/lda, /*ldb=*/ldb, /*ldd=*/ldd,
      /*tiles_n=*/tilesN, /*tiles_m=*/tilesM,
      /*batch_stride_a=*/0, /*batch_stride_b=*/0, /*batch_stride_d=*/0,
      /*swizzle_log=*/swizzle_log,
      /*gemm_k_iterations_aligned=*/(K / BK),
      /*batch_ndim=*/0,
  };

  // Grid: swizzled (tn, tm, batch=1) per MLX (matmul.cpp:300-306).
  const int tile = 1 << swizzle_log;
  const int swizzled_tn = tilesN * tile;
  const int swizzled_tm = (tilesM + tile - 1) / tile;

  // Bindings (skipping FC-gated slots 2/5/6/7).
  stream->recorder().beginDispatch(pso)
      .setInput(0, A.const_data_ptr(), A.nbytes())
      .setInput(1, B.const_data_ptr(), B.nbytes())
      // slot 2 (C / out_source) FC-gated to use_out_source=false → unused.
      .setOutput(3, C.mutable_data_ptr(), C.nbytes())
      .setBytes(4, &params, sizeof(params))
      // slots 5/6/7 FC-gated → unused.
      .run(uvec3{uint32_t(swizzled_tn), uint32_t(swizzled_tm), 1u},
           uvec3{32u, uint32_t(WN), uint32_t(WM)});
}

}  // namespace

//===----------------------------------------------------------------------===//
// dispatchSplitK — SIMD split-K dispatch helper.
// MLX-faithful translation of `steel_matmul_splitk_axpby` in
//   mlx/backend/metal/matmul.cpp:529-680
// (specifically the non-axpby case: alpha=1, beta=0, no bias).
//===----------------------------------------------------------------------===//

void MatMulOp::dispatchSplitK(
    MetalStream* stream,
    const executorch::aten::Tensor& A,
    const executorch::aten::Tensor& B,
    executorch::aten::Tensor& C,
    int32_t M, int32_t K, int32_t N,
    executorch::aten::ScalarType dtype) {
  const int bm = (M < 40) ? 16 : 32;
  const int bn = (N < 40) ? 16 : 32;
  const int bk = 16;
  const int wm = 2;
  const int wn = 2;
  const int _tm = (M + bm - 1) / bm;
  const int _tn = (N + bn - 1) / bn;
  const int _tk = K / bk;
  int P = std::min(std::max(2, next_pow2(_tk / std::max(1, _tm * _tn))), 32);
  // P must divide _tk evenly so each partition gets ≥ 1 K-iter; clamp down.
  while (P > 1 && (_tk / P) < 1) P /= 2;
  const int gemm_k_iterations = _tk / P;
  const int split_k_partition_size = gemm_k_iterations * bk;
  const int split_k_partition_stride = M * N;
  const bool mn_aligned = (M % bm == 0) && (N % bn == 0);
  const bool k_aligned = (K % bk == 0);

  // Allocate the fp32 partial buffer.
  const size_t partial_bytes = size_t(P) * size_t(M) * size_t(N) * sizeof(float);
  void* partial_ptr = stream->allocator().alloc(partial_bytes);
  ET_CHECK_MSG(partial_ptr != nullptr,
               "MatMulOp::dispatchSplitK: alloc(%zu) failed (P=%d M=%d N=%d)",
               partial_bytes, P, M, N);

  // ---- Partial dispatch via MLX `gemm_splitk` template (JIT) ----
  dispatchSimdSplitKPartialViaMlxJit(
      stream, A, B, partial_ptr, partial_bytes,
      M, K, N, bm, bn, bk, wm, wn,
      P, split_k_partition_size, split_k_partition_stride,
      gemm_k_iterations, mn_aligned, k_aligned, dtype);

  // ---- Accum dispatch ----
  {
    char kname[64];
    std::snprintf(kname, sizeof(kname),
                  "matmul_simd_splitk_accum_%s", dtypeSuffix(dtype));
    auto* kernel = getKernel(stream, kname, /*fcs=*/nullptr);
    ET_CHECK_MSG(kernel != nullptr,
                 "MatMulOp::dispatchSplitK: failed to load kernel '%s'", kname);

    // partial = INPUT to phase B (the accum kernel reads partial sums
    // produced by phase A). Hazard-aware setInputBuffer records the
    // RAW dependency on the parent MTLBuffer.
    auto bo = stream->allocator().bufferForPtr(partial_ptr, partial_bytes);

    // One thread per output element. Round N up to a multiple of 32 to
    // fit a (32, 1, 1) TG; in-kernel bounds-check guards the tail.
    const uint32_t tgN = uint32_t((N + 31) / 32);
    stream->recorder().beginDispatch(kernel)
        .setInputBuffer(0, bo.mtl, bo.offset, partial_bytes)
        .setOutput(1, C.mutable_data_ptr(), C.nbytes())
        .setBytes<int32_t>(2, M)
        .setBytes<int32_t>(3, N)
        .setBytes<int32_t>(4, P)
        .setBytes<int32_t>(5, split_k_partition_stride)
        .run(uvec3(tgN, uint32_t(M), 1), uvec3(32, 1, 1));
  }

  // Free the intermediate. The MTLBuffer stays retained by the encoder
  // via setBuffer until commit, so this is safe even though the GPU
  // hasn't run the work yet.
  stream->allocator().free(partial_ptr);
}

//===----------------------------------------------------------------------===//
// NAX split-K dispatch helper.
// MLX-faithful translation of `steel_matmul_splitk_axpby_nax`
//   mlx/backend/metal/matmul.cpp:686-852
// adapted to our simpler tensor_ops::matmul2d-based partial kernel
// (single-tile BM=BN=64, BK=16, 4 simdgroups). Reuses Phase B's accum.
// Caller has already validated NAX eligibility via selectKernel
// (Apple9+, half/bf16, M·N ≥ 4M, K ≥ 10240, K ≥ 3·max(M,N), batch=1).
// Tile / partition params chosen for our BM=BN=64 partial kernel (vs
// MLX's 128×128×512 with 16 simdgroups). We use:
//   BM = BN = 64,  BK = 16
//   split_k_partition_size = 3072  (matches MLX matmul.cpp:711 — gives
//   P=4 partitions for K=12288, P=3 for K=10240)
//===----------------------------------------------------------------------===//

void MatMulOp::dispatchSplitKNAX(
    MetalStream* stream,
    const executorch::aten::Tensor& A,
    const executorch::aten::Tensor& B,
    executorch::aten::Tensor& C,
    int32_t M, int32_t K, int32_t N,
    executorch::aten::ScalarType dtype) {
  // ---- MLX 0.31.2 split-K NAX path via per-shape JIT ----
  // Tile selection per matmul.cpp:678-686.
  int BM, BN, BK, WM, WN;
  int kPartitionK;
  if ((M + N) / 2 < 512 || K <= 4096) {
    BM = BN = 64; BK = 256; WM = WN = 2;
  } else {
    BM = BN = 128; BK = 512; WM = WN = 4;
  }
  if (K <= 1024)        kPartitionK = K / 2;
  else if (K <= 2048)   kPartitionK = 1024;
  else if (K <= 4096)   kPartitionK = 2048;
  else                  kPartitionK = 4096;

  const int P = (K + kPartitionK - 1) / kPartitionK;
  const int split_k_partition_size = kPartitionK;
  const int split_k_partition_stride = M * N;
  const int tn = (N + BN - 1) / BN;
  const int tm = (M + BM - 1) / BM;
  const int swizzle_log = (tm <= 3) ? 0 : 1;
  const int tile = 1 << swizzle_log;
  const int tm_swizzled = (tm + tile - 1) / tile;
  const int tn_swizzled = tn * tile;

  // ---- Allocate fp32 partial buffer ----
  const size_t partial_bytes = size_t(P) * size_t(M) * size_t(N) * sizeof(float);
  void* partial_ptr = stream->allocator().alloc(partial_bytes);
  ET_CHECK_MSG(partial_ptr != nullptr,
               "MatMulOp::dispatchSplitKNAX: alloc(%zu) failed "
               "(P=%d M=%d N=%d)", partial_bytes, P, M, N);

  // ---- NAX partial dispatch (steel_gemm_splitk_nax) ----
  {
    const auto jdtIn = toJitDtype(dtype);
    const auto jdtOut = mlx_jit::JitDtype::Float32;  // partial is fp32
    const char* aTname = mlx_jit::typeToName(jdtIn);
    const char* outTname = mlx_jit::typeToName(jdtOut);
    const std::string baseName = buildBaseName(
        "steel_gemm_splitk_nax", /*ta=*/false, /*tb=*/false,
        aTname, outTname, BM, BN, BK, WM, WN);
    const bool align_M = (M % BM) == 0;
    const bool align_N = (N % BN) == 0;
    const bool align_K = (K % BK) == 0;
    const std::string hashName = baseName + buildSplitKNaxHashSuffix(
        align_M, align_N, align_K);
    const auto fcs = makeMlxSplitKNaxFCs(align_M, align_N);

    auto pso = mlx_jit::shared(stream->compiler())
                   .getSplitKNaxKernel(baseName, hashName, fcs, jdtOut,
                                       /*ta=*/false, /*tb=*/false,
                                       BM, BN, BK, WM, WN);
    ET_CHECK_MSG(pso != nil,
                 "MatMulOp::dispatchSplitKNAX: getSplitKNaxKernel "
                 "returned nil for '%s'", baseName.c_str());

    GEMMSplitKParamsHost params{
        /*M=*/M, /*N=*/N, /*K=*/K,
        /*lda=*/K, /*ldb=*/N, /*ldc=*/N,
        /*tiles_n=*/tn, /*tiles_m=*/tm,
        /*split_k_partitions=*/P,
        /*split_k_partition_stride=*/split_k_partition_stride,
        /*split_k_partition_size=*/split_k_partition_size,
        /*swizzle_log=*/swizzle_log,
        /*gemm_k_iterations_aligned=*/(split_k_partition_size / BK),
    };

    // partial = OUTPUT of split-K phase A.
    auto bo = stream->allocator().bufferForPtr(partial_ptr, partial_bytes);

    // Grid: 1D K-partition-major (matmul.cpp:781-782).
    stream->recorder().beginDispatch(pso)
        .setInput(0, A.const_data_ptr(), A.nbytes())
        .setInput(1, B.const_data_ptr(), B.nbytes())
        .setOutputBuffer(2, bo.mtl, bo.offset, partial_bytes)
        .setBytes(3, &params, sizeof(params))
        .run(uvec3{uint32_t(tn_swizzled * tm_swizzled * P), 1u, 1u},
             uvec3{32u, uint32_t(WN), uint32_t(WM)});
  }

  // ---- Accum dispatch (local matmul_simd_splitk_accum_*; kept off MLX
  // because MLX's gemm_splitk_accum uses dispatch_threads, our stream
  // only exposes dispatch_threadgroups — see helper comment) ----
  {
    char kname[64];
    std::snprintf(kname, sizeof(kname),
                  "matmul_simd_splitk_accum_%s", dtypeSuffix(dtype));
    auto* kernel = getKernel(stream, kname, /*fcs=*/nullptr);
    ET_CHECK_MSG(kernel != nullptr,
                 "MatMulOp::dispatchSplitKNAX: failed to load accum kernel '%s'",
                 kname);

    // partial = INPUT to phase B accum kernel.
    auto bo = stream->allocator().bufferForPtr(partial_ptr, partial_bytes);

    const uint32_t tgN = uint32_t((N + 31) / 32);
    stream->recorder().beginDispatch(kernel)
        .setInputBuffer(0, bo.mtl, bo.offset, partial_bytes)
        .setOutput(1, C.mutable_data_ptr(), C.nbytes())
        .setBytes<int32_t>(2, M)
        .setBytes<int32_t>(3, N)
        .setBytes<int32_t>(4, P)
        .setBytes<int32_t>(5, split_k_partition_stride)
        .run(uvec3{tgN, uint32_t(M), 1u}, uvec3{32u, 1u, 1u});
  }

  stream->allocator().free(partial_ptr);
}

void MatMulOp::dispatchDenseNAX(
    MetalStream* stream,
    const executorch::aten::Tensor& A,
    const executorch::aten::Tensor& B,
    executorch::aten::Tensor& C,
    int32_t M, int32_t K, int32_t N,
    executorch::aten::ScalarType dtype) {
  // ---- Tile selection (kept from pre-JIT empirical tuning) ----
  // Defaults to our (64,64,256,2,2) symmetric small that benched best on
  // M4 Max for prefill shapes. METAL_NAX_TILE overrides for diagnostics
  // (see comment block in the prior implementation).
  int BM = 64, BN = 64, BK = 256, WM = 2, WN = 2;
  static const int forceTile = []() {
    const char* e = getenv("METAL_NAX_TILE");
    if (!e) return 0;
    if (strcmp(e, "asym")  == 0) return 1;  // (64, 128) MLX MacUltra
    if (strcmp(e, "small") == 0) return 2;  // (64, 64)  symmetric — DEFAULT
    if (strcmp(e, "large") == 0) return 3;  // (128, 128) symmetric
    return 0;
  }();
  if (forceTile == 1) { BM = 64;  BN = 128; BK = 256; WM = 2; WN = 4; }
  if (forceTile == 2) { BM = 64;  BN = 64;  BK = 256; WM = 2; WN = 2; }
  if (forceTile == 3) { BM = 128; BN = 128; BK = 512; WM = 4; WN = 4; }

  const int tilesN = (N + BN - 1) / BN;
  const int tilesM = (M + BM - 1) / BM;

  int swizzle_log = 0;
  if (const char* e = getenv("METAL_NAX_SWIZZLE")) {
    swizzle_log = atoi(e);
  }
  const int tile = 1 << swizzle_log;
  const int swizzled_tm = (tilesM + tile - 1) / tile;
  const int swizzled_tn = tilesN * tile;

  // ---- MLX kernel naming (matches matmul.cpp:217-254) ----
  const auto jdt = toJitDtype(dtype);
  const char* tname = mlx_jit::typeToName(jdt);
  const std::string baseName = buildBaseName(
      "steel_gemm_fused_nax", /*ta=*/false, /*tb=*/false,
      tname, tname, BM, BN, BK, WM, WN);

  const bool has_batch = false;          // MatMulOp is non-batched
  const bool use_out_source = false;     // un-fused: no bias / no axpby
  const bool do_axpby = false;
  const bool align_M = (M % BM) == 0;
  const bool align_N = (N % BN) == 0;
  const bool align_K = (K % BK) == 0;

  const std::string hashName = baseName + buildFusedHashSuffix(
      has_batch, use_out_source, do_axpby, align_M, align_N, align_K);

  const auto fcs = makeMlxFusedFCs(
      has_batch, use_out_source, do_axpby, align_M, align_N, align_K);

  ET_LOG(Debug,
         "MatMulOp[mlx_dense_nax/jit]: M=%d K=%d N=%d dtype=%s "
         "tile=(%d,%d,%d,%d,%d) swizzle_log=%d kname=%s",
         M, K, N, dtypeSuffix(dtype), BM, BN, BK, WM, WN, swizzle_log,
         baseName.c_str());

  // ---- Acquire PSO via the free-function singleton ----
  auto pso = mlx_jit::shared(stream->compiler())
                 .getDenseNaxKernel(baseName, hashName, fcs, jdt,
                                    /*ta=*/false, /*tb=*/false,
                                    BM, BN, BK, WM, WN);
  ET_CHECK_MSG(pso != nil,
               "MatMulOp::dispatchDenseNAX: getDenseNaxKernel returned nil "
               "for '%s'", baseName.c_str());

  // ---- Build GEMMParams (struct layout must match MSL exactly) ----
  GEMMParamsHost params{
      /*M=*/M,
      /*N=*/N,
      /*K=*/K,
      /*lda=*/K,    // A row-contig [M,K]
      /*ldb=*/N,    // B row-contig [K,N]
      /*ldd=*/N,    // D row-contig [M,N]
      /*tiles_n=*/tilesN,
      /*tiles_m=*/tilesM,
      /*batch_stride_a=*/0,
      /*batch_stride_b=*/0,
      /*batch_stride_d=*/0,
      /*swizzle_log=*/swizzle_log,
      /*gemm_k_iterations_aligned=*/(K / BK),
      /*batch_ndim=*/0,
  };

  // ---- Bind + dispatch ----
  // Grid: swizzled (tn, tm, batch=1). Block: (32, WN, WM) per MLX's
  // matmul.cpp:305 (MTL::Size(32, wn, wm)).
  stream->recorder().beginDispatch(pso)
      .setInput(0, A.const_data_ptr(), A.nbytes())
      .setInput(1, B.const_data_ptr(), B.nbytes())
      // slot 2 (C / out_source) FC-gated to use_out_source=false → unused.
      .setOutput(3, C.mutable_data_ptr(), C.nbytes())
      .setBytes(4, &params, sizeof(params))
      // slot 5 (addmm_params) FC-gated to use_out_source=false → unused.
      // slot 6/7 (batch_shape/strides) FC-gated to has_batch=false → unused.
      .run(uvec3{uint32_t(swizzled_tn), uint32_t(swizzled_tm), 1u},
           uvec3{32u, uint32_t(WN), uint32_t(WM)});
}

//===----------------------------------------------------------------------===//
// Dispatch
//===----------------------------------------------------------------------===//

void MatMulOp::dispatch(
    MetalStream* stream,
    ::executorch::runtime::Span<::executorch::runtime::EValue*> inputs,
    ::executorch::runtime::Span<::executorch::runtime::EValue*> outputs) {

  // TEMPORARY runtime switch: when METAL_USE_MPSGRAPH=1 (or =true), route ALL
  // matmul cases through MPSGraph instead of our hand-written kernels. Useful
  // for benchmarking and as a sanity-check fallback when the custom kernels
  // misbehave. Selection logic below is left intact (just bypassed).
  // Compiled out entirely when ET_METAL_USE_MPSGRAPH=0 (which is enforced
  // mutually-exclusive with ET_METAL4_ENABLE — see MpsInterop.h).
#if ET_METAL_USE_MPSGRAPH
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
#endif

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

  // GEMV / GEMV_T enabled for M==1. The selector picks GEMV_T whenever
  // M==1 (most common autoregressive-decode shape).
  MatMulKernelType kernelType;
  if (aRC && bRC) {
    kernelType = selectKernel(M, N, K, dtype);
  } else if (aRC && bCC) {
    kernelType = MatMulKernelType::NT;
  } else /* aCC && bRC — A is transposed (TN) */ {
    kernelType = MatMulKernelType::TN;
  }

  // Simd routes through MLX's `gemm` template via the JIT loader rather
  // than tensor_ops::matmul2d. MLX's gemm maps simdgroup_matrix to the
  // Apple9+ NAX hardware path (same HW as tensor_ops::matmul2d), and its
  // BlockLoader / K-loop unroll measure ~45% faster than a hand-rolled
  // tensor_ops wrapper on M4 Max (seq=128 fp32 prefill: 3.36 ms vs
  // 4.91 ms). For NAX-eligible dtypes (fp16/bf16, or fp32+TF32) the
  // routing selects Mlx_Dense_NAX before reaching this point anyway.

  // Phase B: SIMD split-K — two-stage dispatch with an fp32 intermediate.
  // Bypasses the unified single-kernel dispatch flow below.
  if (kernelType == MatMulKernelType::Simd_SplitK) {
    ET_LOG(Debug, "MatMulOp: M=%d, K=%d, N=%d, kernel=splitk", M, K, N);
    dispatchSplitK(stream, A, B, C, M, K, N, dtype);
    return;
  }

  // Phase C: NAX split-K — Apple9+ tensor_ops::matmul2d partial + Phase B accum.
  if (kernelType == MatMulKernelType::SplitK_NAX) {
    ET_LOG(Debug, "MatMulOp: M=%d, K=%d, N=%d, kernel=splitk_nax", M, K, N);
    dispatchSplitKNAX(stream, A, B, C, M, K, N, dtype);
    return;
  }

  // MLX dense NAX (gemm_fused_nax). Tile selection per MLX 0.31.2.
  if (kernelType == MatMulKernelType::Mlx_Dense_NAX) {
    ET_LOG(Debug, "MatMulOp: M=%d, K=%d, N=%d, kernel=mlx_dense_nax", M, K, N);
    dispatchDenseNAX(stream, A, B, C, M, K, N, dtype);
    return;
  }

  // Route the SIMD-MMA family (Simd*/NT/TN) through MLX 0.31.2's `gemm`
  // template (steel_gemm_fused.h) via the per-shape JIT loader. Tile +
  // kernelType selection above owns routing — this just acquires + binds
  // the kernel.
  switch (kernelType) {
    case MatMulKernelType::Simd:
    case MatMulKernelType::Simd_BN32:
    case MatMulKernelType::Simd_M32:
    case MatMulKernelType::Simd_W12:
    case MatMulKernelType::Simd_W12_M32:
    case MatMulKernelType::NT:
    case MatMulKernelType::TN: {
      const auto tile = simdTileFor(kernelType);
      const int32_t tilesM = (M + tile.BM - 1) / tile.BM;
      const int32_t tilesN = (N + tile.BN - 1) / tile.BN;
      const int32_t swizzle_log = (tilesM >= 4 && tilesN >= 4) ? 2 : 0;
      // NT: B is the transposed operand (transpose_b=true).
      // TN: A is the transposed operand (transpose_a=true).
      const bool transpose_a = (kernelType == MatMulKernelType::TN);
      const bool transpose_b = (kernelType == MatMulKernelType::NT);
      ET_LOG(Debug, "MatMulOp: M=%d K=%d N=%d kernel=simd "
                     "(kt=%d tile=(%d,%d,%d,%d,%d) ta=%d tb=%d)",
             M, K, N, int(kernelType),
             tile.BM, tile.BN, tile.BK, tile.WM, tile.WN,
             transpose_a, transpose_b);
      dispatchSimdViaMlxJit(stream, A, B, C, M, K, N,
                            tile.BM, tile.BN, tile.BK, tile.WM, tile.WN,
                            swizzle_log,
                            transpose_a, transpose_b,
                            dtype);
      return;
    }
    default:
      // Other kernelTypes (Naive / Tiled / GEMV / GEMV_T / TensorOps)
      // fall through to the local-kernel dispatch below.
      break;
  }

  // ---- Local-kernel paths (Naive / Tiled / GEMV / GEMV_T) ----
  // GEMV / GEMV_T can also route to MLX's gemv kernel via the JIT
  // loader when METAL_USE_MLX_GEMV=1 (env-gated for A/B testing
  // against our local impl). Naive / Tiled stay local — MLX has no
  // small-tile equivalents.
  static const bool kUseMlxGemv = []() {
    const char* env = getenv("METAL_USE_MLX_GEMV");
    return env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0);
  }();
  if (kUseMlxGemv) {
    if (kernelType == MatMulKernelType::GEMV_T) {
      using mlx_jit_helpers::dispatchGemvTViaMlxJit;
      dispatchGemvTViaMlxJit(stream, A, B, C, K, N, dtype);
      return;
    }
    if (kernelType == MatMulKernelType::GEMV) {
      using mlx_jit_helpers::dispatchGemvViaMlxJit;
      dispatchGemvViaMlxJit(stream, A, B, C, M, K, dtype);
      return;
    }
  }

  // Local fallback: hand-rolled gemv / gemv_t (or Naive / Tiled).
  std::string kname = std::string(kernelTypePrefix(kernelType)) + "_" + dtypeSuffix(dtype);
  auto* kernel = getKernel(stream, kname.c_str(), /*fcs=*/nullptr);

  ET_LOG(Debug, "MatMulOp: M=%d, K=%d, N=%d, kernel=%s", M, K, N, kname.c_str());

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

    case MatMulKernelType::GEMV:
      grid = uvec3(M, 1, 1);
      block = uvec3(32, 1, 1);
      break;

    case MatMulKernelType::GEMV_T:
      grid = uvec3((N + 15) / 16, 1, 1);
      block = uvec3(32, 1, 1);
      break;

    default:
      // Unreachable — Simd*/NT/TN/Mlx_Dense_NAX/Simd_SplitK/SplitK_NAX
      // were all handled above with `return`.
      ET_CHECK_MSG(false, "MatMulOp: unhandled kernelType=%d in local-dispatch "
                          "switch", int(kernelType));
      return;
  }

  // GEMV_T has swapped operand semantics: gemv_t(matrix=B, vector=A, out=C).
  if (kernelType == MatMulKernelType::GEMV_T) {
    stream->recorder().beginDispatch(kernel)
        .setInput(0, B.const_data_ptr(), B.nbytes())   // matrix [K,N]
        .setInput(1, A.const_data_ptr(), A.nbytes())   // vector [K]
        .setOutput(2, C.mutable_data_ptr(), C.nbytes()) // output [N]
        .setBytes<int32_t>(3, M)
        .setBytes<int32_t>(4, K)
        .setBytes<int32_t>(5, N)
        .run(grid, block);
    return;
  }

  // Naive / Tiled / TensorOps / GEMV: simple 6-arg ABI.
  stream->recorder().beginDispatch(kernel)
      .setInput(0, A.const_data_ptr(), A.nbytes())
      .setInput(1, B.const_data_ptr(), B.nbytes())
      .setOutput(2, C.mutable_data_ptr(), C.nbytes())
      .setBytes<int32_t>(3, M)
      .setBytes<int32_t>(4, K)
      .setBytes<int32_t>(5, N)
      .run(grid, block);
}

const char* MatMulOp::kernelSource() const {
  return matmulKernelSource().c_str();
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch
