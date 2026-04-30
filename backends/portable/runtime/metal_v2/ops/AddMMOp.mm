/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "AddMMOp.h"

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
// AddMMOp (aten::addmm) — fused bias-matmul.
// Schema: addmm(input, mat1, mat2, *, beta=1, alpha=1) -> Tensor
//   inputs[0] = input (bias) [M, N] OR broadcast (commonly [N])
//   inputs[1] = mat1 [M, K]
//   inputs[2] = mat2 [K, N]
//   inputs[3] = beta (Scalar, default 1) — IGNORED, must be 1 for now
//   inputs[4] = alpha (Scalar, default 1) — IGNORED, must be 1 for now
// Constraints (currently enforced — caller must satisfy or use mm + add):
//   - mat1 row-contiguous, mat2 row-contiguous (NN layout)
//   - bias is [M, N] contiguous OR [N] (1D-broadcast)
//   - beta == alpha == 1
// Falls through to MatMulOp's plain matmul kernel + a separate elementwise
// add IF those constraints don't hold (rare in practice for nn.Linear).
//===----------------------------------------------------------------------===//

std::vector<SizesType> AddMMOp::computeOutputShape(
    ::executorch::runtime::Span<::executorch::runtime::EValue*> inputs) const {
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
    ::executorch::runtime::Span<::executorch::runtime::EValue*> inputs,
    ::executorch::runtime::Span<::executorch::runtime::EValue*> outputs) {

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

  // Optional alpha/beta scalar inputs (PyTorch addmm signature:
  // addmm(input, mat1, mat2, *, beta=1, alpha=1)). When the AOTI shim
  // forwards them they arrive as inputs[3] (beta) and inputs[4] (alpha).
  // Default to 1.0 if absent.
  float beta  = 1.0f;
  float alpha = 1.0f;
  if (inputs.size() >= 4 && inputs[3]) {
    auto& s = *inputs[3];
    if (s.isDouble())      beta = static_cast<float>(s.toDouble());
    else if (s.isInt())    beta = static_cast<float>(s.toInt());
  }
  if (inputs.size() >= 5 && inputs[4]) {
    auto& s = *inputs[4];
    if (s.isDouble())      alpha = static_cast<float>(s.toDouble());
    else if (s.isInt())    alpha = static_cast<float>(s.toInt());
  }

  const bool aRC = isRowContiguous(A);
  const bool bRC = isRowContiguous(B);
  const bool aCC = !aRC && isColContiguous(A);
  const bool bCC = !bRC && isColContiguous(B);

  if (!(aRC || aCC) || !(bRC || bCC)) {
    ET_LOG(Error,
           "AddMMOp: A and B must each be row- or column-contiguous "
           "(got A.RC=%d A.CC=%d B.RC=%d B.CC=%d)", aRC, aCC, bRC, bCC);
    return;
  }
  if (aCC && bCC) {
    ET_LOG(Error, "AddMMOp: TT (both transposed) is not implemented");
    return;
  }

  int32_t M = static_cast<int32_t>(A.size(0));
  int32_t K = static_cast<int32_t>(A.size(1));
  int32_t N = static_cast<int32_t>(B.size(1));
  ScalarType dtype = C.scalar_type();

  // Determine bias stride pattern. Three supported cases:
  //  1) bias is 1D [N] (broadcasts across M rows)              → stride_m = 0
  //  2) bias is 2D [1, N] with strides[0] == 0 (Inductor-style
  //     broadcast — Inductor expands the 1D bias to 2D shape
  //     but with row-stride 0 instead of contiguous)            → stride_m = 0
  //  3) bias is 2D [M, N] contiguous                            → stride_m = N
  // PyTorch's Inductor canonicalizes `mm(x, w) + bias_1d` into
  // `aten.addmm(bias_expanded, x, w)`, where bias_expanded is the 1D bias
  // broadcast (via .expand) to [M, N] — this preserves shape but keeps
  // the underlying storage 1D with strides[0]=0. We must detect this and
  // treat it as the 1D broadcast case, otherwise we'd read past the bias
  // buffer for rows >= 1 and silently get zeros (UB on Metal).
  int32_t bias_stride_m;
  const bool is_2d_row_broadcast =
      (bias.dim() == 2) && bias.size(1) == N &&
      (bias.strides().size() >= 1) && (bias.strides()[0] == 0);
  if (bias.dim() == 1) {
    if (bias.size(0) != N) {
      ET_LOG(Error, "AddMMOp: 1D bias dim mismatch (got %lld, expected %d)",
             (long long)bias.size(0), N);
      return;
    }
    bias_stride_m = 0;  // same row repeated
  } else if (is_2d_row_broadcast) {
    bias_stride_m = 0;  // 2D-broadcast: same row repeated
  } else if (bias.dim() == 2 && bias.size(0) == M && bias.size(1) == N) {
    bias_stride_m = N;
  } else {
    ET_LOG(Error, "AddMMOp: unsupported bias shape (dim=%zd)",
           (ptrdiff_t)bias.dim());
    return;
  }

  // Tile picker mirrors MatMulOp::selectKernel ladder via the shared helper
  // in MatMulCommon.h. NT/TN layouts only have the 64x64x16 tile.
  const bool transposed = !(aRC && bRC);
  AddmmTilePick tile = pickAddmmTile(M, N, K, /*batch=*/1,
                                     /*transposed_NN_only=*/transposed,
                                     /*dtype=*/dtype,
                                     /*tier=*/MetalDeviceInfo::tier());

  // Route addmm through MLX 0.31.2's `gemm` template (steel_gemm_fused.h)
  // with use_out_source=true via the per-shape JIT loader. Selection
  // logic above (tile picker, layout detection, bias-stride computation)
  // owns the routing — this just acquires + binds the kernel.
  // NT: B is .T view (transpose_b=true). TN: A is .T view (transpose_a=true).
  const bool transpose_a = (aCC && bRC);
  const bool transpose_b = (aRC && bCC);
  const int32_t swizzle_log = pickSwizzleLog(tile.grid);
  // WM/WN derived from block_threads + tile shape. The pickAddmmTile
  // catalog uses three (WM, WN) layouts: (1×2)=64 threads, (1×4)=128
  // threads with BM<BN, (2×2)=128 threads otherwise.
  int WM, WN;
  if (tile.block_threads == 64) {
    WM = 1; WN = 2;
  } else if (tile.BM == 32 && tile.BN == 64 && tile.BK == 32) {
    WM = 1; WN = 4;
  } else {
    WM = 2; WN = 2;
  }
  ET_LOG(Debug,
         "AddMMOp: M=%d K=%d N=%d bias_stride_m=%d alpha=%g beta=%g "
         "tile=(%d,%d,%d,%d,%d) ta=%d tb=%d swizzle_log=%d",
         M, K, N, bias_stride_m, alpha, beta,
         tile.BM, tile.BN, tile.BK, WM, WN,
         transpose_a, transpose_b, swizzle_log);
  using mlx_jit_helpers::dispatchGemmViaMlxJit;
  dispatchGemmViaMlxJit(
      stream, A, B, &bias, C,
      M, K, N, /*batch=*/1,
      tile.BM, tile.BN, tile.BK, WM, WN,
      swizzle_log,
      transpose_a, transpose_b,
      bias_stride_m, /*bias_stride_b=*/0,
      alpha, beta, dtype);
}

const char* AddMMOp::kernelSource() const {
  // All four metal_v2 op classes share matmulKernelSource() — the unified
  // matmul_simd_addmm_t handles addmm via use_out_source=true.
  return matmulKernelSource().c_str();
}

}  // namespace metal_v2
}  // namespace backends
}  // namespace executorch
