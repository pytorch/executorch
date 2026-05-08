/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/metal/ops/registry/MetalOp.h>

namespace executorch {
namespace backends {
namespace metal_v2 {

//===----------------------------------------------------------------------===//
// MatMul Kernel Types (only consumed by MatMulOp::dispatch internally).
//===----------------------------------------------------------------------===//

enum class MatMulKernelType {
  Naive,      // Simple kernel for small matrices
  Tiled,      // Tiled with threadgroup memory (32x32)
  Simd,       // Simdgroup MMA (BM=64x64 tiles, 2x2 simdgroup layout)
  Simd_BN32,  // Simdgroup MMA, BM=64, BN=32, BK=32, 2x2 simd layout
  Simd_M32,   // Simdgroup MMA, BM=32, BN=64, BK=32, 1x4 simd layout
              // (for 2 <= M < 64)
  Simd_W12,   // Simdgroup MMA, BM=64, BN=64, BK=16, 1x2 simd layout (MLX-
              // style: 2 simdgroups/tg → 2x more tgs in flight than Simd)
  Simd_W12_M32, // Simdgroup MMA, BM=32, BN=64, BK=16, 1x2 simd layout —
                // half-M sibling of Simd_W12. MLX picks this for 'd' large
                // half/bf nn deep-K (matmul.cpp:124-129) and 'd' small fp32
                // nt (matmul.cpp:148-153). 64 threads/tg.
  Simd_SplitK,  // SIMD split-K (MLX steel_gemm_splitk). Host allocates a
                // [P, M, N] fp32 intermediate, dispatches the partial
                // kernel with grid.z=P, then reduces via a per-output
                // accum kernel. Picked when batch=1, K≥max(M,N), K%16==0,
                // _tm·_tn ≤ threshold, _tk ≥ 8.
  SplitK_NAX,   // NAX split-K (MLX steel_gemm_splitk_nax). Apple9+ /
                // M3+ only, fp16/bf16 only. Same accum kernel as
                // Simd_SplitK; partial uses tensor_ops::matmul2d.
  Mlx_Dense_NAX, // MLX dense NAX (gemm_fused_nax). Apple9+ M3+ only,
                 // fp16/bf16 only. Selected only when METAL_USE_MLX_DENSE=1
                 // AND split-K-NAX precondition fails. Uses MLX's full
                 // 16-simdgroup cooperative-tensor NAX kernel.
  NT,         // Simd MMA, B is logically transposed (B.T view)
  TN,         // Simd MMA, A is logically transposed (A.T view)
  GEMV,       // y = A @ x   (N == 1)
  GEMV_T,     // C = x @ B   (M == 1, uses gemv_t with swapped operands)
};

//===----------------------------------------------------------------------===//
// MatMulOp - 2D matrix multiply (aten::mm)
// Dispatches to one of several kernel types selected by selectKernel based
// on (M, N, K) and the device tier — see kernelTypePrefix for the names.
//===----------------------------------------------------------------------===//

class MatMulOp : public MetalOp {
 public:
  const char* name() const override { return "aten::mm"; }

  bool supports(ScalarType dtype) const override {
    return isFloatingPoint(dtype);
  }

  std::vector<SizesType> computeOutputShape(
      ::executorch::runtime::Span<::executorch::runtime::EValue*> inputs) const override;

  void dispatch(
      MetalStream* stream,
      ::executorch::runtime::Span<::executorch::runtime::EValue*> inputs,
      ::executorch::runtime::Span<::executorch::runtime::EValue*> outputs) override;

 protected:
  const char* kernelSource() const override;

  MatMulKernelType selectKernel(int64_t M, int64_t N, int64_t K,
                                executorch::aten::ScalarType dtype) const;
  std::string kernelTypePrefix(MatMulKernelType type) const;

  // Phase B: SIMD split-K dispatch helper (mirrors MLX's
  // steel_matmul_splitk_axpby for the no-bias / no-axpby case). Allocates
  // a [P, M, N] fp32 intermediate, dispatches partial then accum, frees.
  // Caller has already validated split-K eligibility.
  void dispatchSplitK(MetalStream* stream,
                      const executorch::aten::Tensor& A,
                      const executorch::aten::Tensor& B,
                      executorch::aten::Tensor& C,
                      int32_t M, int32_t K, int32_t N,
                      executorch::aten::ScalarType dtype);

  // Phase C: NAX split-K dispatch helper (mirrors MLX's
  // steel_matmul_splitk_axpby_nax for the no-axpby case). Apple9+,
  // half/bf16 only. Reuses the Phase B accum kernel.
  void dispatchSplitKNAX(MetalStream* stream,
                         const executorch::aten::Tensor& A,
                         const executorch::aten::Tensor& B,
                         executorch::aten::Tensor& C,
                         int32_t M, int32_t K, int32_t N,
                         executorch::aten::ScalarType dtype);

  // MLX 0.31.2 dense NAX dispatch helper (gemm_fused_nax). Apple9+,
  // half/bf16 only. Single-shot kernel (no split-K, no accum). Tile
  // selection per MLX 0.31.2: small (64,64,256,2,2) when (M+N)/2 < 512
  // || K ≤ 4096, else large (128,128,512,4,4).
  void dispatchDenseNAX(MetalStream* stream,
                        const executorch::aten::Tensor& A,
                        const executorch::aten::Tensor& B,
                        executorch::aten::Tensor& C,
                        int32_t M, int32_t K, int32_t N,
                        executorch::aten::ScalarType dtype);
};

}  // namespace metal_v2
}  // namespace backends
}  // namespace executorch
