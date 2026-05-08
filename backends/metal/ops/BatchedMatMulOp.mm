/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "BatchedMatMulOp.h"

#include <executorch/backends/metal/core/MetalStream.h>
#include <executorch/backends/metal/ops/registry/OpUtils.h>
#include <executorch/backends/metal/ops/MatMulCommon.h>
#include <executorch/backends/metal/ops/MatMulKernels.metal.h>
#include <executorch/backends/metal/ops/MatMulMlxJit.h>
#include <executorch/runtime/platform/log.h>

#include <cstdlib>
#include <cstring>
#include <string>

namespace executorch {
namespace backends {
namespace metal_v2 {

using runtime::Error;


//===----------------------------------------------------------------------===//
// BatchedMatMulOp (aten::bmm) - [B, M, K] @ [B, K, N] -> [B, M, N]
//===----------------------------------------------------------------------===//

std::vector<SizesType> BatchedMatMulOp::computeOutputShape(
    ::executorch::runtime::Span<::executorch::runtime::EValue*> inputs) const {

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
    ::executorch::runtime::Span<::executorch::runtime::EValue*> inputs,
    ::executorch::runtime::Span<::executorch::runtime::EValue*> outputs) {

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

    // Pick local fallback for shapes too small to JIT-compile a per-shape
    // MLX kernel for; otherwise route through MLX JIT (always — even for
    // matmul2d-aligned shapes, MLX's `gemm` template empirically beats
    // tensor_ops::matmul2d on M4 Max for fp32 prefill).
    std::string kname;
    uvec3 grid, block;
    if (M2 >= 64 && N >= 64 && K >= 16) {
      // SIMD-MMA collapsed path: route through MLX's `gemm` template via JIT.
      // Bias-less, batch=1 (collapse already flattened batch into M).
      const int32_t tilesM = (M2 + 63) / 64;
      const int32_t tilesN = (N + 63) / 64;
      const int32_t swizzle_log = (tilesM >= 4 && tilesN >= 4) ? 2 : 0;
      ET_LOG(Debug, "BatchedMatMulOp[collapse]: M=%d K=%d N=%d "
                     "tile=(64,64,16,2,2) swizzle_log=%d",
             M2, K, N, swizzle_log);
      using mlx_jit_helpers::dispatchGemmViaMlxJit;
      dispatchGemmViaMlxJit(
          stream, A, B, /*bias=*/nullptr, C,
          M2, K, N, /*batch=*/1,
          /*BM=*/64, /*BN=*/64, /*BK=*/16, /*WM=*/2, /*WN=*/2,
          swizzle_log,
          /*transpose_a=*/false, /*transpose_b=*/false,
          /*bias_stride_m=*/0, /*bias_stride_b=*/0,
          /*alpha=*/0.0f, /*beta=*/0.0f, dtype);
      return;
    } else if (M2 >= 32 && N >= 32) {
      kname = std::string("matmul_tiled_") + dtypeSuffix(dtype);
      grid = uvec3((N + 31) / 32, (M2 + 31) / 32, 1);
      block = uvec3(32, 32, 1);
    } else {
      kname = std::string("matmul_naive_") + dtypeSuffix(dtype);
      grid = uvec3((N + 7) / 8, (M2 + 7) / 8, 1);
      block = uvec3(8, 8, 1);
    }

    ET_LOG(Debug,
           "BatchedMatMulOp: broadcast collapse batch=%d M=%d->%d K=%d N=%d kernel=%s",
           batch, M, M2, K, N, kname.c_str());

    // Local fallbacks (matmul_tiled / matmul_naive) — MLX has no direct
    // equivalent for these small / unaligned paths.
    auto* kernel = getKernel(stream, kname.c_str(), /*fcs=*/nullptr);
    stream->recorder().beginDispatch(kernel)
        .setInput(0, A.const_data_ptr(), A.nbytes())
        .setInput(1, B.const_data_ptr(), B.nbytes())
        .setOutput(2, C.mutable_data_ptr(), C.nbytes())
        .setBytes<int32_t>(3, M2)
        .setBytes<int32_t>(4, K)
        .setBytes<int32_t>(5, N)
        .run(grid, block);
    return;
  }

  // Prefer the SIMD MMA kernel (with tgid.z = batch) for large enough tiles;
  // fall back to the naive batched kernel for small problems where SIMD
  // would have low occupancy. Routed through MLX's `gemm` template via JIT.
  const bool useSimd = (M >= 64) && (N >= 64) && (K >= 16);

  if (useSimd) {
    const int32_t tilesM = (M + 63) / 64;
    const int32_t tilesN = (N + 63) / 64;
    const int32_t swizzle_log = (tilesM >= 4 && tilesN >= 4) ? 2 : 0;
    ET_LOG(Debug, "BatchedMatMulOp: batch=%d M=%d K=%d N=%d "
                   "tile=(64,64,16,2,2) swizzle_log=%d",
           batch, M, K, N, swizzle_log);
    using mlx_jit_helpers::dispatchGemmViaMlxJit;
    dispatchGemmViaMlxJit(
        stream, A, B, /*bias=*/nullptr, C,
        M, K, N, /*batch=*/batch,
        /*BM=*/64, /*BN=*/64, /*BK=*/16, /*WM=*/2, /*WN=*/2,
        swizzle_log,
        /*transpose_a=*/false, /*transpose_b=*/false,
        /*bias_stride_m=*/0, /*bias_stride_b=*/0,
        /*alpha=*/0.0f, /*beta=*/0.0f, dtype);
    return;
  }

  // Naive fallback (small problems).
  int32_t A_batch_stride = M * K;
  int32_t B_batch_stride = K * N;
  int32_t C_batch_stride = M * N;
  std::string kname = std::string("bmm_") + dtypeSuffix(dtype);
  auto* kernel = getKernel(stream, kname.c_str());
  ET_LOG(Debug, "BatchedMatMulOp: naive batch=%d M=%d K=%d N=%d",
         batch, M, K, N);
  uvec3 grid((N + 7) / 8, (M + 7) / 8, batch);
  uvec3 block(8, 8, 1);
  // Naive bmm kernel buffer layout: A,B,C, batch,M,K,N, A_stride,B_stride,C_stride.
  stream->recorder().beginDispatch(kernel)
      .setInput(0, A.const_data_ptr(), A.nbytes())
      .setInput(1, B.const_data_ptr(), B.nbytes())
      .setOutput(2, C.mutable_data_ptr(), C.nbytes())
      .setBytes<int32_t>(3, batch)
      .setBytes<int32_t>(4, M)
      .setBytes<int32_t>(5, K)
      .setBytes<int32_t>(6, N)
      .setBytes<int32_t>(7, A_batch_stride)
      .setBytes<int32_t>(8, B_batch_stride)
      .setBytes<int32_t>(9, C_batch_stride)
      .run(grid, block);
}

const char* BatchedMatMulOp::kernelSource() const {
  // Share matmulKernelSource() — we use matmul_simd_addmm_t (use_out_source
  // =false) for the SIMD fast path and bmm_<dtype> for the naive fallback.
  return matmulKernelSource().c_str();
}

}  // namespace metal_v2
}  // namespace backends
}  // namespace executorch
