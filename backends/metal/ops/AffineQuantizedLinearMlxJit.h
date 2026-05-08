/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

//===----------------------------------------------------------------------===//
// AffineQuantizedLinearMlxJit — JIT-routing helpers for the affine
// quantized linear op family (qmv_fast / qmv / qmv_quad / qmm_t).
// Mirrors MatMulMlxJit.h / SdpaMlxJit.h. Each helper:
//   1. Builds the MLX-style kernel name for the given (dtype, group_size,
//      bits, [D|aligned_N]) combo.
//   2. Acquires a PSO via ops/mlx_jit/KernelLoader (per-shape JIT).
//   3. Binds the MLX qmv/qmm buffer ABI and dispatches via MetalStream.
// IMPORTANT — `biases` semantics: MLX's affine kernels reconstruct
//   w_dequant = w_quant * scale + bias
// PyTorch users typically think in zero-point terms:
//   w_dequant = (w_quant - zero_point) * scale
// These are algebraically equivalent: bias = -scale * zero_point.
// v1's op_linear_4bit.mm passes `Z` (the user-provided zero-point arg)
// DIRECTLY as biases at slot 2 — i.e. torchao's quantize step already
// stores the value in MLX's "biases" form. v2 follows the same convention
// for byte-for-byte compatibility with existing PTEs.
// So in our op surface, the optional `wz` tensor IS the MLX biases
// buffer — not a true PyTorch zero-point. When `wz` is None (symmetric
// quant), we allocate a scratch buffer of zeros and bind it.
//===----------------------------------------------------------------------===//

#include <executorch/backends/metal/core/MetalStream.h>
#include <executorch/backends/metal/ops/registry/OpUtils.h>
#include <executorch/backends/metal/ops/mlx_jit/KernelLoader.h>
#include <executorch/runtime/platform/log.h>

#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <algorithm>  // std::max, std::min (used in splitk dispatch)

namespace executorch {
namespace backends {
namespace metal_v2 {
namespace aql_helpers {

// Map ScalarType → JitDtype for activation (qkv input). Quantized linear
// supports the same input dtypes as MLX's quantized kernels: fp32, fp16,
// and bf16. (uint8 is the wq weight, handled separately.)
inline mlx_jit::JitDtype toJitDtype(executorch::aten::ScalarType dt) {
  switch (dt) {
    case executorch::aten::ScalarType::Float:    return mlx_jit::JitDtype::Float32;
    case executorch::aten::ScalarType::Half:     return mlx_jit::JitDtype::Float16;
    case executorch::aten::ScalarType::BFloat16: return mlx_jit::JitDtype::BFloat16;
    default:
      ET_CHECK_MSG(false, "aql: unsupported activation dtype %d", int(dt));
      return mlx_jit::JitDtype::Float32;
  }
}

// MLX kernel-name builders. Match the host_name format we emit in
// KernelLoader (NOT MLX's AOT naming, since we JIT one per shape).
inline std::string buildQmvFastName(mlx_jit::JitDtype dtype, int group_size, int bits) {
  std::ostringstream s;
  s << "affine_qmv_fast_" << mlx_jit::typeToTemplateArg(dtype)
    << "_gs" << group_size << "_b" << bits << "_batch0";
  return s.str();
}
inline std::string buildQmvName(mlx_jit::JitDtype dtype, int group_size, int bits) {
  std::ostringstream s;
  s << "affine_qmv_" << mlx_jit::typeToTemplateArg(dtype)
    << "_gs" << group_size << "_b" << bits << "_batch0";
  return s.str();
}
inline std::string buildQmvQuadName(mlx_jit::JitDtype dtype, int group_size, int bits, int D) {
  std::ostringstream s;
  s << "affine_qmv_quad_" << mlx_jit::typeToTemplateArg(dtype)
    << "_gs" << group_size << "_b" << bits << "_d" << D << "_batch0";
  return s.str();
}
inline std::string buildQmmTName(mlx_jit::JitDtype dtype, int group_size, int bits, bool aligned_N) {
  std::ostringstream s;
  s << "affine_qmm_t_" << mlx_jit::typeToTemplateArg(dtype)
    << "_gs" << group_size << "_b" << bits
    << "_alN" << (aligned_N ? '1' : '0') << "_batch0";
  return s.str();
}
inline std::string buildQmmTSplitKName(mlx_jit::JitDtype dtype, int group_size, int bits, bool aligned_N) {
  std::ostringstream s;
  s << "affine_qmm_t_splitk_" << mlx_jit::typeToTemplateArg(dtype)
    << "_gs" << group_size << "_b" << bits
    << "_alN" << (aligned_N ? '1' : '0');
  return s.str();
}
inline std::string buildQmmTNaxName(mlx_jit::JitDtype dtype, int group_size, int bits, bool aligned_N) {
  std::ostringstream s;
  s << "affine_qmm_t_nax_" << mlx_jit::typeToTemplateArg(dtype)
    << "_gs" << group_size << "_b" << bits
    << "_alN" << (aligned_N ? '1' : '0')
    << "_bm64_bn64_bk64_wm2_wn2_batch0";
  return s.str();
}
inline std::string buildSplitKAccumName(mlx_jit::JitDtype dtype) {
  std::ostringstream s;
  s << "gemm_splitk_accum_" << mlx_jit::typeToTemplateArg(dtype)
    << "_to_" << mlx_jit::typeToTemplateArg(dtype);
  return s.str();
}

// MLX qmv* / qmm_t kernels expect biases at buffer slot 2. The dispatch
// helpers below take `wz` as a non-null `Tensor&` — translation from
// "no zero-points provided (symmetric quant)" to a zeros-filled biases
// tensor happens at the registered op level (AffineQuantizedLinearOp),
// NOT here. This keeps the JIT layer's contract clean: it always receives
// real biases, never special-cases nullptr.

//===----------------------------------------------------------------------===//
// dispatchAffineQmvViaMlxJit — decode (M=1) routing across the qmv variants.
// Mirrors MLX's host-side qmv / qmv_quad logic
// (mlx/backend/metal/quantized.cpp:177-300).
// Tile selection:
//   - bits==4 && K∈{64,128} → qmv_quad (template adds D parameter)
//   - K % 512 == 0 && N % 8 == 0 → qmv_fast
//   - otherwise → qmv (generic)
// Buffer ABI (qmv* family, non-batched):
//   0  w (uint32_t* — wq's uint8 storage reinterpreted; Metal buffers are
//                    naturally aligned to ≥16 bytes, so safe to cast)
//   1  scales (T*)
//   2  biases (T*)  — wz if given, else allocated zeros
//   3  x (T*)
//   4  y (T*)
//   5  in_vec_size (int) = K
//   6  out_vec_size (int) = N
//   (slots 7-14 are batched-only; we don't bind them since batched=false
//    template-DCEs the access path)
// Grid (qmv_fast / qmv): (M, ceil(N/8), 1).  Block: (32, 2, 1).
// Grid (qmv_quad):       (M, ceil(N/64), 1). Block: (32, 1, 1).
//===----------------------------------------------------------------------===//

inline void dispatchAffineQmvViaMlxJit(
    MetalStream* stream,
    const executorch::aten::Tensor& x,        // [M, K]   activation
    const executorch::aten::Tensor& wq,       // [N, K*nbit/8]  uint8 packed
    const executorch::aten::Tensor& ws,       // [N, K/group_size] scales
    const executorch::aten::Tensor& wz,       // [N, K/group_size] biases
    executorch::aten::Tensor& y,              // [M, N]   output
    int M, int N, int K,
    int group_size, int bits,
    executorch::aten::ScalarType dtype) {
  ET_CHECK_MSG(M >= 1, "dispatchAffineQmvViaMlxJit requires M>=1, got %d", M);

  const auto jdt = toJitDtype(dtype);

  // Tile selection (port from quantized.cpp::qmv_quad and qmv).
  const bool quad_eligible = (bits == 4) && (K == 64 || K == 128);
  const bool fast_eligible = (N % 8 == 0) && (K % 512 == 0);

  enum class Variant { Quad, Fast, Generic };
  Variant variant = quad_eligible
      ? Variant::Quad
      : (fast_eligible ? Variant::Fast : Variant::Generic);

  std::string kname;
  id<MTLComputePipelineState> pso = nil;
  switch (variant) {
    case Variant::Quad: {
      kname = buildQmvQuadName(jdt, group_size, bits, K);
      pso = mlx_jit::shared(stream->compiler())
                .getAffineQmvQuadKernel(kname, jdt, group_size, bits, K);
      break;
    }
    case Variant::Fast: {
      kname = buildQmvFastName(jdt, group_size, bits);
      pso = mlx_jit::shared(stream->compiler())
                .getAffineQmvFastKernel(kname, jdt, group_size, bits);
      break;
    }
    case Variant::Generic: {
      kname = buildQmvName(jdt, group_size, bits);
      pso = mlx_jit::shared(stream->compiler())
                .getAffineQmvKernel(kname, jdt, group_size, bits);
      break;
    }
  }
  ET_CHECK_MSG(pso != nil,
               "dispatchAffineQmvViaMlxJit: PSO acquisition failed for '%s'",
               kname.c_str());

  // Direct biases binding — caller (op layer) is responsible for ensuring
  // `wz` is a real biases tensor (zero-filled scratch in the symmetric
  // case, the user's tensor in the asymmetric case).
  ET_LOG(Debug,
         "dispatchAffineQmvViaMlxJit: M=%d N=%d K=%d gs=%d bits=%d "
         "variant=%s dtype=%s kname=%s",
         M, N, K, group_size, bits,
         variant == Variant::Quad    ? "quad" :
         variant == Variant::Fast    ? "fast" : "generic",
         dtypeSuffix(dtype), kname.c_str());

  // Bind buffers + dispatch.
  uvec3 grid;
  uvec3 block;
  if (variant == Variant::Quad) {
    const int bn = 64;  // quads_per_simd * results_per_quadgroup = 8 * 8
    grid = {uint32_t(M), uint32_t((N + bn - 1) / bn), 1u};
    block = {32u, 1u, 1u};
  } else {
    const int bn = 8;
    grid = {uint32_t(M), uint32_t((N + bn - 1) / bn), 1u};
    block = {32u, 2u, 1u};
  }

  stream->recorder().beginDispatch(pso)
      .setInput(0, wq.const_data_ptr(), wq.nbytes())
      .setInput(1, ws.const_data_ptr(), ws.nbytes())
      .setInput(2, wz.const_data_ptr(), wz.nbytes())
      .setInput(3, x.const_data_ptr(), x.nbytes())
      .setOutput(4, y.mutable_data_ptr(), y.nbytes())
      .setBytes<int32_t>(5, K)
      .setBytes<int32_t>(6, N)
      // Slots 7-14: batched-only, not bound (kernel template-DCEs the
      // access path when batched=false).
      .run(grid, block);

}

//===----------------------------------------------------------------------===//
// dispatchAffineQmmTViaMlxJit — prefill (M>1) qmm_t.
// Mirrors MLX's host-side qmm logic (mlx/backend/metal/quantized.cpp::qmm).
// Buffer ABI (qmm_t, non-batched):
//   0  w   (uint32_t*)         5  K  (int)
//   1  scales (T*)             6  N  (int)
//   2  biases (T*)             7  M  (int)
//   3  x (T*)                  (slots 8-15: batched-only)
//   4  y (T*)
// Tile: BM=BK=BN=32 default; WM=WN=2.
// Grid: (ceil(N/BN), ceil(M/BM), 1).  Block: (32, WN=2, WM=2).
// `aligned_N` template arg: true when N % BN == 0 — skips N-tail safe-load.
//===----------------------------------------------------------------------===//

inline void dispatchAffineQmmTViaMlxJit(
    MetalStream* stream,
    const executorch::aten::Tensor& x,        // [M, K]
    const executorch::aten::Tensor& wq,       // [N, K*nbit/8]
    const executorch::aten::Tensor& ws,       // [N, K/group_size]
    const executorch::aten::Tensor& wz,       // [N, K/group_size] biases
    executorch::aten::Tensor& y,              // [M, N]
    int M, int N, int K,
    int group_size, int bits,
    executorch::aten::ScalarType dtype) {
  ET_CHECK_MSG(M > 1, "dispatchAffineQmmTViaMlxJit requires M>1, got %d", M);

  const auto jdt = toJitDtype(dtype);
  constexpr int BM = 32, BN = 32;
  constexpr int WM = 2, WN = 2;
  const bool aligned_N = (N % BN) == 0;

  const std::string kname = buildQmmTName(jdt, group_size, bits, aligned_N);
  auto pso = mlx_jit::shared(stream->compiler())
                 .getAffineQmmTKernel(kname, jdt, group_size, bits, aligned_N);
  ET_CHECK_MSG(pso != nil,
               "dispatchAffineQmmTViaMlxJit: PSO acquisition failed for '%s'",
               kname.c_str());

  ET_LOG(Debug,
         "dispatchAffineQmmTViaMlxJit: M=%d N=%d K=%d gs=%d bits=%d "
         "alignedN=%d dtype=%s kname=%s",
         M, N, K, group_size, bits, int(aligned_N),
         dtypeSuffix(dtype), kname.c_str());

  stream->recorder().beginDispatch(pso)
      .setInput(0, wq.const_data_ptr(), wq.nbytes())
      .setInput(1, ws.const_data_ptr(), ws.nbytes())
      .setInput(2, wz.const_data_ptr(), wz.nbytes())
      .setInput(3, x.const_data_ptr(), x.nbytes())
      .setOutput(4, y.mutable_data_ptr(), y.nbytes())
      .setBytes<int32_t>(5, K)
      .setBytes<int32_t>(6, N)
      .setBytes<int32_t>(7, M)
      // Slots 8-15: batched-only.
      .run(
          uvec3{
              uint32_t((N + BN - 1) / BN),
              uint32_t((M + BM - 1) / BM),
              1u},
          uvec3{32u, uint32_t(WN), uint32_t(WM)});

}

//===----------------------------------------------------------------------===//
// dispatchAffineQmmTNaxViaMlxJit — prefill (M>1) qmm_t on Apple9+ NAX.
// Mirrors MLX's qmm_nax (mlx/backend/metal/quantized.cpp:473-574).
// Buffer ABI (qmm_t_nax, non-batched): same as qmm_t (slots 0-7),
// with batched stride slots 8-15 template-DCE'd when batched=false.
//   0  w   (uint32_t*)         5  K  (int)
//   1  scales (T*)             6  N  (int)
//   2  biases (T*)             7  M  (int)
//   3  x (T*)                  (slots 8-15: batched-only)
//   4  y (T*)
// Tile: BM=BN=64, BK=64. WM=WN=2.
// Grid: (ceil(N/64), ceil(M/64), 1). Block: (32, 2, 2).
// `aligned_N` template arg: true when N % 64 == 0.
// Caller must have verified MetalDeviceInfo::isNaxAvailable() && K % 64 == 0.
//===----------------------------------------------------------------------===//

inline void dispatchAffineQmmTNaxViaMlxJit(
    MetalStream* stream,
    const executorch::aten::Tensor& x,        // [M, K]
    const executorch::aten::Tensor& wq,       // [N, K*nbit/8]
    const executorch::aten::Tensor& ws,       // [N, K/group_size]
    const executorch::aten::Tensor& wz,       // [N, K/group_size] biases
    executorch::aten::Tensor& y,              // [M, N]
    int M, int N, int K,
    int group_size, int bits,
    executorch::aten::ScalarType dtype) {
  ET_CHECK_MSG(M > 1, "dispatchAffineQmmTNaxViaMlxJit requires M>1, got %d", M);
  ET_CHECK_MSG(K % 64 == 0,
               "dispatchAffineQmmTNaxViaMlxJit requires K %% 64 == 0, got K=%d", K);

  const auto jdt = toJitDtype(dtype);
  constexpr int BM = 64, BN = 64;
  constexpr int WM = 2, WN = 2;
  const bool aligned_N = (N % BN) == 0;

  const std::string kname = buildQmmTNaxName(jdt, group_size, bits, aligned_N);
  auto pso = mlx_jit::shared(stream->compiler())
                 .getAffineQmmTNaxKernel(kname, jdt, group_size, bits, aligned_N);
  ET_CHECK_MSG(pso != nil,
               "dispatchAffineQmmTNaxViaMlxJit: PSO acquisition failed for '%s'",
               kname.c_str());

  ET_LOG(Debug,
         "dispatchAffineQmmTNaxViaMlxJit: M=%d N=%d K=%d gs=%d bits=%d "
         "alignedN=%d dtype=%s kname=%s",
         M, N, K, group_size, bits, int(aligned_N),
         dtypeSuffix(dtype), kname.c_str());

  stream->recorder().beginDispatch(pso)
      .setInput(0, wq.const_data_ptr(), wq.nbytes())
      .setInput(1, ws.const_data_ptr(), ws.nbytes())
      .setInput(2, wz.const_data_ptr(), wz.nbytes())
      .setInput(3, x.const_data_ptr(), x.nbytes())
      .setOutput(4, y.mutable_data_ptr(), y.nbytes())
      .setBytes<int32_t>(5, K)
      .setBytes<int32_t>(6, N)
      .setBytes<int32_t>(7, M)
      // Slots 8-15: batched-only (template-DCE'd when batched=false).
      .run(
          uvec3{
              uint32_t((N + BN - 1) / BN),
              uint32_t((M + BM - 1) / BM),
              1u},
          uvec3{32u, uint32_t(WN), uint32_t(WM)});

}

//===----------------------------------------------------------------------===//
// dispatchAffineQmmTSplitKViaMlxJit — prefill (M>=vector_limit) split-K qmm_t.
// Mirrors MLX's qmm_splitk (mlx/backend/metal/quantized.cpp:774-867).
// Algorithm:
//   1. Choose split_k targeting ~512 threadgroups, clamp to K/group_size,
//      walk down until K % (split_k * group_size) == 0.
//   2. If split_k <= 1 → return false (caller falls back to qmm_t / qmm_t_nax).
//   3. Allocate intermediate [split_k, M, N] in `dtype` via stream->allocator().alloc.
//   4. Dispatch affine_qmm_t_splitk → intermediate.
//   5. Dispatch gemm_splitk_accum (REUSE getSplitKAccumKernel) → y.
//   6. Free intermediate.
// Buffer ABI (affine_qmm_t_splitk):
//   0  w   (uint32_t*)         5  K  (int)
//   1  scales (T*)             6  N  (int)
//   2  biases (T*)             7  M  (int)
//   3  x (T*)                  8  k_partition_size (int)
//   4  y (T*) — intermediate   9  split_k_partition_stride (int)
// Buffer ABI (gemm_splitk_accum):
//   0  C_split (T*)            3  partition_stride (int) = M * N
//   1  D (T*)                  4  ldd (int) = N
//   2  k_partitions (int) = split_k
// Returns true if dispatched, false if split_k degenerated to <= 1.
//===----------------------------------------------------------------------===//

inline bool dispatchAffineQmmTSplitKViaMlxJit(
    MetalStream* stream,
    const executorch::aten::Tensor& x,        // [M, K]
    const executorch::aten::Tensor& wq,       // [N, K*nbit/8]
    const executorch::aten::Tensor& ws,       // [N, K/group_size]
    const executorch::aten::Tensor& wz,       // [N, K/group_size] biases
    executorch::aten::Tensor& y,              // [M, N] output
    int M, int N, int K,
    int group_size, int bits,
    executorch::aten::ScalarType dtype) {
  ET_CHECK_MSG(M > 1, "dispatchAffineQmmTSplitKViaMlxJit requires M>1, got %d", M);

  // Compute split_k (port of qmm_splitk:788-805).
  constexpr int BM = 32, BN = 32;
  const int n_tiles = (N + BN - 1) / BN;
  const int m_tiles = (M + BM - 1) / BM;
  const int current_tgs = n_tiles * m_tiles;
  int split_k = std::max(1, 512 / current_tgs);
  split_k = std::min(split_k, K / group_size);
  while (split_k > 1 && (K % (split_k * group_size) != 0)) {
    split_k--;
  }
  if (split_k <= 1) {
    return false;  // caller falls back to plain qmm_t / qmm_t_nax
  }

  const int k_partition_size = K / split_k;
  const int split_k_partition_stride = M * N;

  // Element size for intermediate buffer.
  size_t element_size = 4;
  switch (dtype) {
    case executorch::aten::ScalarType::Half:
    case executorch::aten::ScalarType::BFloat16: element_size = 2; break;
    default: element_size = 4; break;
  }
  const size_t intermediate_bytes =
      size_t(split_k) * size_t(M) * size_t(N) * element_size;
  void* intermediate = stream->allocator().alloc(intermediate_bytes);
  ET_CHECK_MSG(intermediate != nullptr,
               "dispatchAffineQmmTSplitKViaMlxJit: alloc(%zu) failed",
               intermediate_bytes);

  const auto jdt = toJitDtype(dtype);
  const bool aligned_N = (N % BN) == 0;
  const std::string kname = buildQmmTSplitKName(jdt, group_size, bits, aligned_N);
  auto pso = mlx_jit::shared(stream->compiler())
                 .getAffineQmmTSplitKKernel(kname, jdt, group_size, bits, aligned_N);
  ET_CHECK_MSG(pso != nil,
               "dispatchAffineQmmTSplitKViaMlxJit: PSO acquisition failed for '%s'",
               kname.c_str());

  ET_LOG(Debug,
         "dispatchAffineQmmTSplitKViaMlxJit: M=%d N=%d K=%d gs=%d bits=%d "
         "split_k=%d k_part=%d alignedN=%d dtype=%s kname=%s",
         M, N, K, group_size, bits, split_k, k_partition_size,
         int(aligned_N), dtypeSuffix(dtype), kname.c_str());

  // Phase A: split-K qmm_t → intermediate.
  stream->recorder().beginDispatch(pso)
      .setInput(0, wq.const_data_ptr(), wq.nbytes())
      .setInput(1, ws.const_data_ptr(), ws.nbytes())
      .setInput(2, wz.const_data_ptr(), wz.nbytes())
      .setInput(3, x.const_data_ptr(), x.nbytes())
      .setOutput(4, intermediate, intermediate_bytes)
      .setBytes<int32_t>(5, K)
      .setBytes<int32_t>(6, N)
      .setBytes<int32_t>(7, M)
      .setBytes<int32_t>(8, k_partition_size)
      .setBytes<int32_t>(9, split_k_partition_stride)
      .run(uvec3{uint32_t(n_tiles), uint32_t(m_tiles), uint32_t(split_k)},
           uvec3{32u, 2u, 2u});

  // Phase B: gemm_splitk_accum → y. REUSES the existing getSplitKAccumKernel.
  // gemm_splitk_accum<AccT, OutT> with AccT==OutT==dtype and axpby=false.
  const std::string accum_kname = buildSplitKAccumName(jdt);
  auto accum_pso = mlx_jit::shared(stream->compiler())
                       .getSplitKAccumKernel(accum_kname, jdt, jdt, /*axpby=*/false);
  ET_CHECK_MSG(accum_pso != nil,
               "dispatchAffineQmmTSplitKViaMlxJit: accum PSO acquisition failed");

  // Grid: 2D (ceil(N/32), M). Block: (32, 1, 1) — 32 N-elements per group,
  // one M-row per group. Total threads = (ceil(N/32)*32) × M, where the
  // kernel uses `thread_position_in_grid`.  Safe when N % 32 == 0 (true for
  // all torchao quantized linear shapes — N is always packed to 8 elements
  // and tiled to 32 in qmm_t_splitk's BN).
  stream->recorder().beginDispatch(accum_pso)
      .setInput(0, intermediate, intermediate_bytes)
      .setOutput(1, y.mutable_data_ptr(), y.nbytes())
      .setBytes<int32_t>(2, split_k)
      .setBytes<int32_t>(3, split_k_partition_stride)  // = M * N
      .setBytes<int32_t>(4, N)                          // ldd
      .run(uvec3{uint32_t((N + 31) / 32), uint32_t(M), 1u},
           uvec3{32u, 1u, 1u});

  stream->allocator().free(intermediate);
  return true;
}

}  // namespace aql_helpers
}  // namespace metal_v2
}  // namespace backends
}  // namespace executorch
