/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "BAddBMMOp.h"

#include <executorch/backends/portable/runtime/metal_v2/MetalStream.h>
#include <executorch/backends/portable/runtime/metal_v2/MetalDeviceInfo.h>
#include <executorch/backends/portable/runtime/metal_v2/OpUtils.h>
#include <executorch/backends/portable/runtime/metal_v2/ops/MatMulCommon.h>
#include <executorch/backends/portable/runtime/metal_v2/ops/MatMulKernels.metal.h>
#include <executorch/backends/portable/runtime/metal_v2/ops/MatMulMlxJit.h>
#include <executorch/runtime/platform/log.h>

#include <cstdlib>
#include <cstring>
#include <string>

namespace executorch {
namespace backends {
namespace metal_v2 {

using runtime::Error;


//===----------------------------------------------------------------------===//
// BAddBMMOp (aten::baddbmm) — 3D batched fused matmul + bias.
// Reuses matmul_simd_addmm_t with grid.z = batch. Inputs shapes:
//   bias: [B, M, N]            → bias_stride_b = M*N, bias_stride_m = N
//         [M, N] / [1, M, N]   → bias_stride_b = 0,    bias_stride_m = N
//         [N] / [1, 1, N] / [B, 1, N] → bias_stride_b = ?, bias_stride_m = 0
//   batch1: [B, M, K] (NN — row-contig)
//   batch2: [B, K, N] (NN — row-contig)
// Currently supports NN layout with row-contiguous batch1/batch2. TN/NT
// or non-contiguous batch dims would need additional handling — out of
// scope until a model needs them.
//===----------------------------------------------------------------------===//

std::vector<SizesType> BAddBMMOp::computeOutputShape(
    ::executorch::runtime::Span<::executorch::runtime::EValue*> inputs) const {
  if (inputs.size() < 3 || !inputs[1]->isTensor() || !inputs[2]->isTensor()) {
    return {};
  }
  const auto& batch1 = inputs[1]->toTensor();
  const auto& batch2 = inputs[2]->toTensor();
  if (batch1.dim() != 3 || batch2.dim() != 3) return {};
  return {static_cast<SizesType>(batch1.size(0)),
          static_cast<SizesType>(batch1.size(1)),
          static_cast<SizesType>(batch2.size(2))};
}

void BAddBMMOp::dispatch(
    MetalStream* stream,
    ::executorch::runtime::Span<::executorch::runtime::EValue*> inputs,
    ::executorch::runtime::Span<::executorch::runtime::EValue*> outputs) {
  if (inputs.size() < 3) {
    ET_LOG(Error, "BAddBMMOp: expected at least 3 inputs (input, batch1, batch2)");
    return;
  }
  auto& bias   = inputs[0]->toTensor();
  auto& batch1 = inputs[1]->toTensor();
  auto& batch2 = inputs[2]->toTensor();
  auto& C      = outputs[0]->toTensor();

  auto err = resizeOutput(inputs, outputs[0]);
  if (err != Error::Ok) {
    ET_LOG(Error, "BAddBMMOp: failed to resize output");
    return;
  }

  if (batch1.dim() != 3 || batch2.dim() != 3) {
    ET_LOG(Error, "BAddBMMOp: batch1 and batch2 must be 3D");
    return;
  }
  if (!isRowContiguous(batch1) || !isRowContiguous(batch2)) {
    ET_LOG(Error, "BAddBMMOp: batch1/batch2 must be row-contiguous (NN only)");
    return;
  }

  // Optional Scalar args at positions [3]/[4].
  float beta  = 1.0f;
  float alpha = 1.0f;
  if (inputs.size() >= 4 && inputs[3]) {
    auto& s = *inputs[3];
    if (s.isDouble())   beta = static_cast<float>(s.toDouble());
    else if (s.isInt()) beta = static_cast<float>(s.toInt());
  }
  if (inputs.size() >= 5 && inputs[4]) {
    auto& s = *inputs[4];
    if (s.isDouble())   alpha = static_cast<float>(s.toDouble());
    else if (s.isInt()) alpha = static_cast<float>(s.toInt());
  }

  int32_t B = static_cast<int32_t>(batch1.size(0));
  int32_t M = static_cast<int32_t>(batch1.size(1));
  int32_t K = static_cast<int32_t>(batch1.size(2));
  int32_t N = static_cast<int32_t>(batch2.size(2));
  ScalarType dtype = C.scalar_type();

  // Bias broadcast detection. Three common shapes:
  //   3D [B, M, N] contiguous            → stride_b = M*N, stride_m = N
  //   3D [B, M, N] row-broadcast strides → stride_b = ?,   stride_m = 0/N
  //   2D [M, N] expanded → 3D [B, M, N] with stride[0]==0, stride[1]==N, stride[2]==1
  //   1D [N]   expanded → 3D [B, M, N] with strides all matching broadcast (0,0,1)
  // We support exactly these cases via the (stride_b, stride_m) pair.
  int32_t bias_stride_b = 0;
  int32_t bias_stride_m = 0;

  if (bias.dim() == 1) {
    if (bias.size(0) != N) {
      ET_LOG(Error, "BAddBMMOp: 1D bias size mismatch (got %lld, expected %d)",
             (long long)bias.size(0), N);
      return;
    }
    bias_stride_b = 0;
    bias_stride_m = 0;
  } else if (bias.dim() == 2 && bias.size(0) == M && bias.size(1) == N) {
    bias_stride_b = 0;
    bias_stride_m = N;
  } else if (bias.dim() == 3 && bias.size(1) == M && bias.size(2) == N) {
    if (bias.size(0) == 1 ||
        (bias.strides().size() >= 1 && bias.strides()[0] == 0)) {
      // Broadcast across batch.
      bias_stride_b = 0;
    } else if (bias.size(0) == B) {
      bias_stride_b = M * N;
    } else {
      ET_LOG(Error, "BAddBMMOp: 3D bias batch dim must be 1 or B (got %lld vs B=%d)",
             (long long)bias.size(0), B);
      return;
    }
    // Row dim: stride[1] == 0 → broadcast across M, else stride_m = N.
    if (bias.strides().size() >= 2 && bias.strides()[1] == 0) {
      bias_stride_m = 0;
    } else {
      bias_stride_m = N;
    }
  } else {
    ET_LOG(Error, "BAddBMMOp: unsupported bias shape (dim=%zd)",
           (ptrdiff_t)bias.dim());
    return;
  }

  // Same kernel as AddMMOp's NN path; tg grid extended to z = B so each
  // batch gets its own threadgroup along z. NN-only here (no transposed
  // batched layouts in scope).
  AddmmTilePick tile = pickAddmmTile(M, N, K, /*batch=*/B,
                                     /*transposed_NN_only=*/false,
                                     /*dtype=*/dtype,
                                     /*tier=*/MetalDeviceInfo::tier());

  // Route the batched bias-matmul through MLX 0.31.2's `gemm` template
  // via the JIT loader. has_batch=false: we use GEMMParams.batch_stride_*
  // (single-axis batched layout) instead of the batch_shape[]/
  // batch_strides[] arrays.
  const int32_t swizzle_log = pickSwizzleLog(tile.grid);
  int WM, WN;
  if (tile.block_threads == 64) {
    WM = 1; WN = 2;
  } else if (tile.BM == 32 && tile.BN == 64 && tile.BK == 32) {
    WM = 1; WN = 4;
  } else {
    WM = 2; WN = 2;
  }
  ET_LOG(Debug,
         "BAddBMMOp: B=%d M=%d K=%d N=%d bias_stride=(b=%d, m=%d) "
         "alpha=%g beta=%g tile=(%d,%d,%d,%d,%d) swizzle_log=%d",
         B, M, K, N, bias_stride_b, bias_stride_m, alpha, beta,
         tile.BM, tile.BN, tile.BK, WM, WN, swizzle_log);
  using mlx_jit_helpers::dispatchGemmViaMlxJit;
  dispatchGemmViaMlxJit(
      stream, batch1, batch2, &bias, C,
      M, K, N, /*batch=*/B,
      tile.BM, tile.BN, tile.BK, WM, WN,
      swizzle_log,
      /*transpose_a=*/false, /*transpose_b=*/false,
      bias_stride_m, bias_stride_b,
      alpha, beta, dtype);
}

const char* BAddBMMOp::kernelSource() const {
  return matmulKernelSource().c_str();
}

}  // namespace metal_v2
}  // namespace backends
}  // namespace executorch
