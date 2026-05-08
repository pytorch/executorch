/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

//===----------------------------------------------------------------------===//
// MatMulMlxJit — shared JIT-routing helpers for the matmul family
// (MatMulOp / AddMMOp / BAddBMMOp / BatchedMatMulOp).
// Each helper acquires an MLX 0.31.2 `gemm` / `gemm_splitk` /
// `gemm_splitk_nax` PSO via ops/mlx_jit/KernelLoader, then binds the
// MLX-style buffer ABI (GEMMParams + optional GEMMAddMMParams +
// optional batch_strides) and dispatches via MetalStream.
// All helpers expect the **caller** to have already chosen kernelType +
// tile (BM, BN, BK, WM, WN). They do not make routing decisions — only
// kernel acquisition + binding.
// This header is the single owner of the host-side mirrors of MLX's
// (GEMMParams / GEMMSpiltKParams / GEMMAddMMParams) structs. The MSL
// layout MUST match exactly — struct layout is verified at static_assert
// below.
//===----------------------------------------------------------------------===//

#include <executorch/backends/metal/core/MetalStream.h>
#include <executorch/backends/metal/ops/registry/OpUtils.h>
#include <executorch/backends/metal/ops/MatMulCommon.h>
#include <executorch/backends/metal/ops/mlx_jit/KernelLoader.h>
#include <executorch/runtime/platform/log.h>

#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>

namespace executorch {
namespace backends {
namespace metal_v2 {
namespace mlx_jit_helpers {

//===----------------------------------------------------------------------===//
// Host mirrors of MLX 0.31.2 structs (mlx/backend/metal/kernels/steel/gemm/
// params.h). Layout verified by static_assert; do NOT change without
// re-verifying against the MSL ABI.
//===----------------------------------------------------------------------===//

// 8 ints (32B) + 3 int64 (24B, 8B-aligned) + 3 ints (12B) + 4B tail pad = 72B.
struct GEMMParamsHost {
  int M;
  int N;
  int K;
  int lda;
  int ldb;
  int ldd;
  int tiles_n;
  int tiles_m;
  int64_t batch_stride_a;
  int64_t batch_stride_b;
  int64_t batch_stride_d;
  int swizzle_log;
  int gemm_k_iterations_aligned;
  int batch_ndim;
};
static_assert(sizeof(GEMMParamsHost) == 72,
              "GEMMParamsHost layout must match MLX upstream GEMMParams");

// 13 ints, all 4B aligned → 52B (struct alignment = 4).
struct GEMMSplitKParamsHost {
  int M;
  int N;
  int K;
  int lda;
  int ldb;
  int ldc;
  int tiles_n;
  int tiles_m;
  int split_k_partitions;
  int split_k_partition_stride;
  int split_k_partition_size;
  int swizzle_log;
  int gemm_k_iterations_aligned;
};
static_assert(sizeof(GEMMSplitKParamsHost) == 52,
              "GEMMSplitKParamsHost layout must match MLX upstream "
              "GEMMSpiltKParams");

// GEMMAddMMParams: 2 ints + 1 int64 + 2 floats. Layout:
//   ldc(4) fdc(4) [pad(0 — int64 aligned to 8 already after 8B of ints)]
//   batch_stride_c(8) alpha(4) beta(4) → total 24B.
struct GEMMAddMMParamsHost {
  int ldc;
  int fdc;
  int64_t batch_stride_c;
  float alpha;
  float beta;
};
static_assert(sizeof(GEMMAddMMParamsHost) == 24,
              "GEMMAddMMParamsHost layout must match MLX upstream "
              "GEMMAddMMParams");

//===----------------------------------------------------------------------===//
// Common naming + FC-tuple helpers (mirror MLX matmul.cpp:217-254).
//===----------------------------------------------------------------------===//

// Map executorch ScalarType → mlx_jit::JitDtype. Only fp32/fp16/bf16
// supported (matches MLX's matmul-kernel coverage).
inline mlx_jit::JitDtype toJitDtype(executorch::aten::ScalarType dt) {
  switch (dt) {
    case executorch::aten::ScalarType::Float:    return mlx_jit::JitDtype::Float32;
    case executorch::aten::ScalarType::Half:     return mlx_jit::JitDtype::Float16;
    case executorch::aten::ScalarType::BFloat16: return mlx_jit::JitDtype::BFloat16;
    default:
      ET_CHECK_MSG(false, "mlx_jit_helpers: unsupported dtype %d", int(dt));
      return mlx_jit::JitDtype::Float32;
  }
}

// MLX kernel name builder (matches matmul.cpp:217-228).
inline std::string buildBaseName(
    const char* prefix,         // "steel_gemm_fused", "steel_gemm_fused_nax", ...
    bool transpose_a, bool transpose_b,
    const char* a_tname,        // type_to_name(a)
    const char* out_tname,      // type_to_name(out)
    int bm, int bn, int bk, int wm, int wn) {
  std::ostringstream s;
  s << prefix << "_"
    << (transpose_a ? 't' : 'n')
    << (transpose_b ? 't' : 'n')
    << "_" << a_tname
    << "_" << out_tname
    << "_bm" << bm << "_bn" << bn << "_bk" << bk
    << "_wm" << wm << "_wn" << wn;
  return s.str();
}

// MLX hash-name suffix for fused (NAX or SIMD) matmul (matmul.cpp:246-252).
inline std::string buildFusedHashSuffix(
    bool has_batch, bool use_out_source, bool do_axpby,
    bool align_M, bool align_N, bool align_K) {
  std::ostringstream s;
  s << "_has_batch_" << (has_batch ? 't' : 'n')
    << "_use_out_source_" << (use_out_source ? 't' : 'n')
    << "_do_axpby_" << (do_axpby ? 't' : 'n')
    << "_align_M_" << (align_M ? 't' : 'n')
    << "_align_N_" << (align_N ? 't' : 'n')
    << "_align_K_" << (align_K ? 't' : 'n');
  return s.str();
}

// MLX hash-name suffix for split-K NAX (matmul.cpp:728-730).
inline std::string buildSplitKNaxHashSuffix(
    bool align_M, bool align_N, bool align_K) {
  std::ostringstream s;
  s << "_align_M_" << (align_M ? 't' : 'n')
    << "_align_N_" << (align_N ? 't' : 'n')
    << "_align_K_" << (align_K ? 't' : 'n');
  return s.str();
}

// FC tuple for MLX dense fused matmul (slots 10/100/110/200/201/202).
inline MetalKernelCompiler::FunctionConstants makeMlxFusedFCs(
    bool has_batch, bool use_out_source, bool do_axpby,
    bool align_M, bool align_N, bool align_K) {
  return MetalKernelCompiler::FunctionConstants{{
      {10,  has_batch},
      {100, use_out_source},
      {110, do_axpby},
      {200, align_M},
      {201, align_N},
      {202, align_K},
  }};
}

// FC tuple for MLX split-K NAX (slots 200/201; align_K is per-tile
// dispatched at runtime in the kernel, not an FC).
inline MetalKernelCompiler::FunctionConstants makeMlxSplitKNaxFCs(
    bool align_M, bool align_N) {
  return MetalKernelCompiler::FunctionConstants{{
      {200, align_M},
      {201, align_N},
  }};
}

//===----------------------------------------------------------------------===//
// dispatchGemmViaMlxJit — general MLX `gemm` template dispatch helper.
// Handles all combinations of:
//   - use_out_source (bias add): pass `bias` non-null
//   - do_axpby (alpha != 1 || beta != 1): inferred from alpha/beta when
//     use_out_source=true; otherwise forced false
//   - batched: pass `batch > 1` (uses GEMMParams.batch_stride_* with
//     has_batch=false; the kernel walks `tid.z * batch_stride` not
//     batch_shape[]/batch_strides[] — simpler for the single-axis batch
//     dim our ops have)
//   - transpose_a / transpose_b: emit MLX kernel with appropriate
//     template args (NN / NT / TN)
// FCs (mirror MLX upstream slot numbers):
//   10  has_batch       (always false here — we use batch_stride_*)
//   100 use_out_source  (== bias != nullptr)
//   110 do_axpby
//   200 align_M
//   201 align_N
//   202 align_K
// Buffer bindings:
//   0  A
//   1  B
//   2  bias / out_source       (skipped if use_out_source=false; FC-gated)
//   3  D (output)
//   4  GEMMParams              (constant struct)
//   5  GEMMAddMMParams         (skipped if use_out_source=false; FC-gated)
// Grid: swizzled (tn, tm, batch). Block: (32, WN, WM).
// Caller is responsible for picking (BM, BN, BK, WM, WN), the swizzle_log,
// the bias stride pattern (bias_stride_m / bias_stride_b), and alpha/beta.
inline void dispatchGemmViaMlxJit(
    MetalStream* stream,
    const executorch::aten::Tensor& A,
    const executorch::aten::Tensor& B,
    const executorch::aten::Tensor* bias,        // nullable
    executorch::aten::Tensor& C,
    int32_t M, int32_t K, int32_t N,
    int32_t batch,                               // 1 for non-batched
    int BM, int BN, int BK, int WM, int WN,
    int32_t swizzle_log,
    bool transpose_a, bool transpose_b,
    int32_t bias_stride_m,                       // ldc; ignored if !bias
    int32_t bias_stride_b,                       // batch_stride_c; ignored if !bias
    float alpha,                                 // ignored if !bias
    float beta,                                  // ignored if !bias
    executorch::aten::ScalarType dtype) {
  const auto jdt = toJitDtype(dtype);
  const char* tname = mlx_jit::typeToName(jdt);

  const std::string baseName = buildBaseName(
      "steel_gemm_fused", transpose_a, transpose_b,
      tname, tname, BM, BN, BK, WM, WN);

  const bool use_out_source = (bias != nullptr);
  const bool do_axpby =
      use_out_source && ((alpha != 1.0f) || (beta != 1.0f));
  const bool has_batch = false;  // we use GEMMParams.batch_stride_* instead
  const bool align_M = (M % BM) == 0;
  const bool align_N = (N % BN) == 0;
  const bool align_K = (K % BK) == 0;

  const std::string hashName = baseName + buildFusedHashSuffix(
      has_batch, use_out_source, do_axpby, align_M, align_N, align_K);
  const auto fcs = makeMlxFusedFCs(
      has_batch, use_out_source, do_axpby, align_M, align_N, align_K);

  ET_LOG(Debug,
         "dispatchGemmViaMlxJit: M=%d K=%d N=%d batch=%d dtype=%s "
         "tile=(%d,%d,%d,%d,%d) ta=%d tb=%d use_out_source=%d "
         "do_axpby=%d swizzle_log=%d kname=%s",
         M, K, N, batch, dtypeSuffix(dtype), BM, BN, BK, WM, WN,
         transpose_a, transpose_b, int(use_out_source), int(do_axpby),
         swizzle_log, baseName.c_str());

  auto pso = mlx_jit::shared(stream->compiler())
                 .getDenseGemmKernel(baseName, hashName, fcs, jdt,
                                     transpose_a, transpose_b,
                                     BM, BN, BK, WM, WN);
  ET_CHECK_MSG(pso != nil,
               "dispatchGemmViaMlxJit: getDenseGemmKernel returned nil "
               "for '%s'", baseName.c_str());

  // ---- GEMMParams ----
  // Leading dims for NN row-major layout:
  //   transpose_a=false → A row-major [M, K]   → lda = K
  //   transpose_a=true  → A.T row-major [K, M] → lda = M
  //   transpose_b=false → B row-major [K, N]   → ldb = N
  //   transpose_b=true  → B.T row-major [N, K] → ldb = K
  //   D row-major [M, N] → ldd = N
  const int lda = transpose_a ? M : K;
  const int ldb = transpose_b ? K : N;
  const int ldd = N;
  const int tilesN = (N + BN - 1) / BN;
  const int tilesM = (M + BM - 1) / BM;

  // Single-axis batched layout: contiguous along the batch dim. When
  // batch == 1, all three batch_stride_* are 0 (so tid.z=0 doesn't move
  // pointers anywhere and the kernel reduces to a non-batched gemm).
  const int64_t bsa = (batch > 1) ? int64_t(M) * int64_t(K) : 0;
  // BatchedMatMulOp's broadcast fast path collapses to batch==1, so we
  // don't need a broadcast bsb here. Caller of the batched form passes
  // bsb = K*N (or 0 for explicit batch-broadcast on B; not currently
  // exercised but supported via this struct field).
  const int64_t bsb = (batch > 1) ? int64_t(K) * int64_t(N) : 0;
  const int64_t bsd = (batch > 1) ? int64_t(M) * int64_t(N) : 0;

  GEMMParamsHost params{
      /*M=*/M, /*N=*/N, /*K=*/K,
      /*lda=*/lda, /*ldb=*/ldb, /*ldd=*/ldd,
      /*tiles_n=*/tilesN, /*tiles_m=*/tilesM,
      /*batch_stride_a=*/bsa,
      /*batch_stride_b=*/bsb,
      /*batch_stride_d=*/bsd,
      /*swizzle_log=*/swizzle_log,
      /*gemm_k_iterations_aligned=*/(K / BK),
      /*batch_ndim=*/0,
  };

  // ---- Bind buffers ----
  auto d = stream->recorder().beginDispatch(pso);
  d.setInput(0, A.const_data_ptr(), A.nbytes());
  d.setInput(1, B.const_data_ptr(), B.nbytes());

  if (use_out_source) {
    // bias bound at slot 2 with addmm params at slot 5.
    d.setInput(2, bias->const_data_ptr(), bias->nbytes());
  }
  // else: slot 2 FC-gated to use_out_source=false → unused; do not bind.

  d.setOutput(3, C.mutable_data_ptr(), C.nbytes());
  d.setBytes(4, &params, sizeof(params));

  if (use_out_source) {
    GEMMAddMMParamsHost addmm_params{
        /*ldc=*/bias_stride_m,
        /*fdc=*/1,
        /*batch_stride_c=*/int64_t(bias_stride_b),
        /*alpha=*/alpha,
        /*beta=*/beta,
    };
    d.setBytes(5, &addmm_params, sizeof(addmm_params));
  }
  // slots 6/7 FC-gated to has_batch=false → unused.

  // ---- Dispatch ----
  // Grid: swizzled (tn, tm, batch).
  const int tile = 1 << swizzle_log;
  const int swizzled_tn = tilesN * tile;
  const int swizzled_tm = (tilesM + tile - 1) / tile;
  uvec3 grid{uint32_t(swizzled_tn), uint32_t(swizzled_tm), uint32_t(batch)};
  uvec3 block{32u, uint32_t(WN), uint32_t(WM)};
  d.run(grid, block);
}

//===----------------------------------------------------------------------===//
// dispatchGemvTViaMlxJit — MLX 0.31.2's gemv_t kernel (MatMulOp's M==1
// autoregressive-decode path: c = A_row @ B where A_row is [K], B is
// [K, N], c is [N]).
// In MLX-speak:  mat = B  (shape [in=K, out=N], transpose_mat=true)
//                in_vec = A_row (shape [K])
//                out_vec = C    (shape [N])
// Tile selection follows MLX's matmul.cpp:1085-1132 (gemv_t branch):
//   K >= 8192 && N >= 2048 → sm=4, sn=8;  else sm=8, sn=4
//   N >= 2048 → bn=16; N >= 512 → bn=4; else bn=2
//   tm = tn = 4 (specialized down to tn=1 for very small N)
//   bm = 1
//   n_out_per_tgp = bn * sn * tn
//   grid = (ceil(N / n_out_per_tgp), 1, 1)
//   block = (32, bn, bm) = (32, bn, 1)  → 32 * bn = 64..512 threads
// Buffer ABI (gemv.metal:438-507):
//   0  mat (B)               9   batch_ndim
//   1  in_vec (A_row)        10  batch_shape*  (skipped, batch_ndim=0)
//   2  bias                  11  vector_batch_stride*  (must bind, kernel reads [0])
//   3  out_vec (C)           12  matrix_batch_stride*  (must bind, kernel reads [0])
//   4  in_vec_size (K)       13  bias_batch_stride*    (skipped, kDoAxpby=false)
//   5  out_vec_size (N)      14  bias_stride           (skipped, kDoAxpby=false)
//   6  matrix_ld (N)
//   7  alpha                 (skipped, kDoAxpby=false)
//   8  beta                  (skipped, kDoAxpby=false)
// kDoNCBatch=false + batch_ndim=0 path: kernel reads vector_batch_stride[0]
// and matrix_batch_stride[0] — we bind a zero int64 at each slot.
inline void dispatchGemvTViaMlxJit(
    MetalStream* stream,
    const executorch::aten::Tensor& A,   // [1, K] = a row vector
    const executorch::aten::Tensor& B,   // [K, N]
    executorch::aten::Tensor& C,         // [1, N]
    int32_t K, int32_t N,
    executorch::aten::ScalarType dtype) {
  const auto jdt = toJitDtype(dtype);
  const char* tname = mlx_jit::typeToName(jdt);

  // Tile-shape pick mirrors MLX matmul.cpp:1091-1110 for transpose_mat=true.
  const int in_vec_len = K;
  const int out_vec_len = N;
  int sm, sn;
  if (in_vec_len >= 8192 && out_vec_len >= 2048) {
    sm = 4; sn = 8;
  } else {
    sm = 8; sn = 4;
  }
  int bn;
  if (out_vec_len >= 2048) {
    bn = 16;
  } else if (out_vec_len >= 512) {
    bn = 4;
  } else {
    bn = 2;
  }
  const int bm = 1;
  int tm = 4;
  int tn = (out_vec_len < 4) ? 1 : 4;

  const int n_out_per_tgp = bn * sn * tn;

  std::ostringstream kn;
  kn << "gemv_t_" << tname
     << "_bm" << bm << "_bn" << bn
     << "_sm" << sm << "_sn" << sn
     << "_tm" << tm << "_tn" << tn
     << "_nc0_axpby0";
  const std::string kernelName = kn.str();

  ET_LOG(Debug,
         "dispatchGemvTViaMlxJit: K=%d N=%d dtype=%s tile=(bm=%d,bn=%d,"
         "sm=%d,sn=%d,tm=%d,tn=%d) n_out_per_tgp=%d kname=%s",
         K, N, dtypeSuffix(dtype),
         bm, bn, sm, sn, tm, tn, n_out_per_tgp, kernelName.c_str());

  auto pso = mlx_jit::shared(stream->compiler())
                 .getGemvTKernel(kernelName, jdt,
                                 bm, bn, sm, sn, tm, tn,
                                 /*nc_batch=*/false, /*axpby=*/false);
  ET_CHECK_MSG(pso != nil,
               "dispatchGemvTViaMlxJit: getGemvTKernel returned nil for '%s'",
               kernelName.c_str());

  // Bindings.
  const int64_t zero64 = 0;
  stream->recorder().beginDispatch(pso)
      .setInput(0, B.const_data_ptr(), B.nbytes())      // mat
      .setInput(1, A.const_data_ptr(), A.nbytes())      // in_vec
      // slot 2 (bias) FC-gated to kDoAxpby=false → unused; do not bind.
      .setOutput(3, C.mutable_data_ptr(), C.nbytes())   // out_vec
      .setBytes<int32_t>(4, K)                          // in_vec_size
      .setBytes<int32_t>(5, N)                          // out_vec_size
      .setBytes<int32_t>(6, N)                          // matrix_ld (B is [K,N] row-major → ld = N)
      // slots 7/8 (alpha/beta) FC-gated to kDoAxpby=false → unused.
      .setBytes<int32_t>(9, 0)                          // batch_ndim
      // slot 10 (batch_shape*) — kernel only reads when kDoNCBatch=true
      //   (false here) → don't bind.
      // slots 11/12: kernel reads vector_batch_stride[0] and
      //   matrix_batch_stride[0] unconditionally in the kDoNCBatch=false
      //   path. Bind a single int64=0 at each.
      .setBytes(11, &zero64, sizeof(zero64))
      .setBytes(12, &zero64, sizeof(zero64))
      // slots 13/14 (bias_batch_stride, bias_stride) — FC-gated to
      //   kDoAxpby=false → unused.
      .run(
          // Grid + block. matmul.cpp:1148-1150:
          //   group_dims = (32, bn, bm)
          //   grid_dims  = (n_tgp, 1, batch_size_out)  — batch=1 for us
          uvec3{uint32_t((N + n_out_per_tgp - 1) / n_out_per_tgp), 1u, 1u},
          uvec3{32u, uint32_t(bn), uint32_t(bm)});
}

//===----------------------------------------------------------------------===//
// dispatchGemvViaMlxJit — MLX's gemv kernel (mat[M,K] @ vec[K] = out[M]).
// Used when N==1 (rare path; selectKernel routes most decode through
// gemv_t). Same ABI as gemv_t.
//===----------------------------------------------------------------------===//

inline void dispatchGemvViaMlxJit(
    MetalStream* stream,
    const executorch::aten::Tensor& A,   // [M, K]
    const executorch::aten::Tensor& B,   // [K, 1] = column vector
    executorch::aten::Tensor& C,         // [M, 1]
    int32_t M, int32_t K,
    executorch::aten::ScalarType dtype) {
  const auto jdt = toJitDtype(dtype);
  const char* tname = mlx_jit::typeToName(jdt);

  // Tile-shape pick mirrors MLX matmul.cpp:1114-1125 (transpose_mat=false).
  const int in_vec_len = K;
  const int out_vec_len = M;
  int bm = (out_vec_len >= 4096) ? 8 : 4;
  int sm = 1, sn = 32;
  int bn = 1;
  if (in_vec_len <= 64) {
    bm = 1; sm = 8; sn = 4;
  } else if (in_vec_len >= 16 * out_vec_len) {
    bm = 1; bn = 8;
  }
  int tm = (out_vec_len < 4) ? 1 : 4;
  int tn = 4;

  const int n_out_per_tgp = bm * sm * tm;

  std::ostringstream kn;
  kn << "gemv_" << tname
     << "_bm" << bm << "_bn" << bn
     << "_sm" << sm << "_sn" << sn
     << "_tm" << tm << "_tn" << tn
     << "_nc0_axpby0";
  const std::string kernelName = kn.str();

  ET_LOG(Debug,
         "dispatchGemvViaMlxJit: M=%d K=%d dtype=%s tile=(bm=%d,bn=%d,"
         "sm=%d,sn=%d,tm=%d,tn=%d) n_out_per_tgp=%d kname=%s",
         M, K, dtypeSuffix(dtype),
         bm, bn, sm, sn, tm, tn, n_out_per_tgp, kernelName.c_str());

  auto pso = mlx_jit::shared(stream->compiler())
                 .getGemvKernel(kernelName, jdt,
                                bm, bn, sm, sn, tm, tn,
                                /*nc_batch=*/false, /*axpby=*/false);
  ET_CHECK_MSG(pso != nil,
               "dispatchGemvViaMlxJit: getGemvKernel returned nil for '%s'",
               kernelName.c_str());

  const int64_t zero64 = 0;
  stream->recorder().beginDispatch(pso)
      .setInput(0, A.const_data_ptr(), A.nbytes())      // mat
      .setInput(1, B.const_data_ptr(), B.nbytes())      // in_vec
      .setOutput(3, C.mutable_data_ptr(), C.nbytes())   // out_vec
      .setBytes<int32_t>(4, K)                          // in_vec_size
      .setBytes<int32_t>(5, M)                          // out_vec_size
      .setBytes<int32_t>(6, K)                          // matrix_ld (A is [M,K] row-major → ld = K)
      .setBytes<int32_t>(9, 0)                          // batch_ndim
      .setBytes(11, &zero64, sizeof(zero64))
      .setBytes(12, &zero64, sizeof(zero64))
      .run(
          uvec3{uint32_t((M + n_out_per_tgp - 1) / n_out_per_tgp), 1u, 1u},
          uvec3{32u, uint32_t(bn), uint32_t(bm)});
}

}  // namespace mlx_jit_helpers
}  // namespace metal_v2
}  // namespace backends
}  // namespace executorch
