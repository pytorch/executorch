/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/portable/runtime/metal_v2/MetalOp.h>

namespace executorch {
namespace backends {
namespace metal_v2 {

//===----------------------------------------------------------------------===//
// MatMul Kernel Types
//===----------------------------------------------------------------------===//

enum class MatMulKernelType {
  Naive,           // Simple kernel for small matrices
  Tiled,           // Tiled with threadgroup memory (32x32)
  Simd,            // Simdgroup MMA (BM=64x64 tiles, 2x2 simdgroup layout)
  Simd_BN32,       // Simdgroup MMA, BM=64, BN=32, BK=32, 2x2 simd layout —
                   // MLX's "small fp32 NN" tile. Smaller BN doubles the
                   // tg count along N (more parallelism for small-N cases),
                   // larger BK halves the K-tile barrier count.
  Simd_M32,        // Simdgroup MMA, BM=32, BN=64, BK=32, 1x4 simd layout
                   // (for 2 <= M < 64, with bounds-check waste for small M)
  NT,              // Simd MMA, B is logically transposed (B.T view)
  TN,              // Simd MMA, A is logically transposed (A.T view)
  GEMV,            // y = A @ x   (N == 1)
  GEMV_T,          // C = x @ B   (M == 1, uses gemv_t with swapped operands)
  TensorOps        // Metal 4 tensor_ops::matmul2d (Apple9+, fastest path when supported)
};

//===----------------------------------------------------------------------===//
// MatMulOp - 2D matrix multiply (aten::mm)
//===----------------------------------------------------------------------===//

class MatMulOp : public MetalOp {
public:
  const char* name() const override { return "aten::mm"; }

  bool supports(ScalarType dtype) const override {
    return isFloatingPoint(dtype);
  }

  std::vector<SizesType> computeOutputShape(
      EValuePtrSpan inputs) const override;

  void dispatch(
      MetalStream* stream,
      EValuePtrSpan inputs,
      EValuePtrSpan outputs) override;

protected:
  const char* kernelSource() const override;

  MatMulKernelType selectKernel(int64_t M, int64_t N, int64_t K) const;
  const char* kernelTypePrefix(MatMulKernelType type) const;
};

//===----------------------------------------------------------------------===//
// BatchedMatMulOp - 3D batched matrix multiply (aten::bmm)
//===----------------------------------------------------------------------===//

class BatchedMatMulOp : public MetalOp {
public:
  const char* name() const override { return "aten::bmm"; }

  bool supports(ScalarType dtype) const override {
    return isFloatingPoint(dtype);
  }

  std::vector<SizesType> computeOutputShape(
      EValuePtrSpan inputs) const override;

  void dispatch(
      MetalStream* stream,
      EValuePtrSpan inputs,
      EValuePtrSpan outputs) override;

protected:
  const char* kernelSource() const override;
};

//===----------------------------------------------------------------------===//
// AddMMOp - fused 2D matrix multiply with bias (aten::addmm)
//
// Computes: out = beta * input + alpha * (mat1 @ mat2)
//   inputs[0] = input (bias) [M, N] — must be 2D contiguous OR 1D-broadcast [N]
//   inputs[1] = mat1 [M, K]
//   inputs[2] = mat2 [K, N]
//   inputs[3] = beta (Scalar, default 1)
//   inputs[4] = alpha (Scalar, default 1)
//
// Currently supports the common LLM/Linear case: alpha=1, beta=1, NN layout,
// bias is [M, N] contiguous OR [N] (1D-broadcast). Falls back to MatMulOp
// when constraints don't hold (caller-side check; for now we just assert).
//
// Saves an entire elementwise add pass over the matmul's output by fusing
// the bias add into the matmul kernel's epilogue (loaded as an 8x8 simdgroup
// fragment, added to the accumulator before simdgroup_store).
//===----------------------------------------------------------------------===//

class AddMMOp : public MetalOp {
public:
  const char* name() const override { return "aten::addmm"; }

  bool supports(ScalarType dtype) const override {
    return isFloatingPoint(dtype);
  }

  std::vector<SizesType> computeOutputShape(
      EValuePtrSpan inputs) const override;

  void dispatch(
      MetalStream* stream,
      EValuePtrSpan inputs,
      EValuePtrSpan outputs) override;

protected:
  const char* kernelSource() const override;
};

} // namespace metal_v2
} // namespace backends
} // namespace executorch
