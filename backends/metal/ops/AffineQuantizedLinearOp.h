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
// AffineQuantizedLinearOp — fused affine-quantized linear via MLX JIT.
// Registered as `executorch_native::affine_quantized_linear.default`
// (custom op declared in apple/metal/affine_quantized_linear_op.py).
// Schema:
//   inputs[0]  x          Tensor [..., K]    activation (fp32/fp16/bf16)
//   inputs[1]  wq         Tensor [N, K*nbit/8]  uint8 packed weight
//   inputs[2]  ws         Tensor [N, K/group_size]  scales (same dtype as x)
//   inputs[3]  wz         Tensor or None      [N, K/group_size] biases
//                                             (None → symmetric, allocates zeros)
//                                             NOTE: this is MLX's "biases"
//                                             (-scale*zero_point precomputed),
//                                             matching v1's op_linear_4bit.mm
//                                             convention.
//   inputs[4]  b          Tensor or None      [N] linear bias (optional)
//   inputs[5]  group_size int                 32, 64, 128, 256, 512, or 1024
//   inputs[6]  nbit       int                 4 or 8 only (initially)
// Output:
//   outputs[0]            Tensor [..., N]
// Routing:
//   M=1 (decode):
//     bits==4 && K∈{64,128}            → qmv_quad
//     N % 8 == 0 && K % 512 == 0       → qmv_fast
//     else                              → qmv (generic)
//   M>1 (prefill):                       qmm_t
// Bias `b` (linear bias) handling: post-add elementwise after the qmv/qmm
// dispatch via the existing AddOp through MetalOpRegistry. The qmm_t
// kernel doesn't natively support output bias; this is the simplest
// correct path for v0.
//===----------------------------------------------------------------------===//

class AffineQuantizedLinearOp : public MetalOp {
 public:
  const char* name() const override {
    return "executorch_native::affine_quantized_linear.default";
  }

  bool supports(ScalarType dtype) const override {
    return dtype == ScalarType::Float ||
           dtype == ScalarType::Half ||
           dtype == ScalarType::BFloat16;
  }

  std::vector<SizesType> computeOutputShape(
      ::executorch::runtime::Span<::executorch::runtime::EValue*> inputs) const override;

  void dispatch(
      MetalStream* stream,
      ::executorch::runtime::Span<::executorch::runtime::EValue*> inputs,
      ::executorch::runtime::Span<::executorch::runtime::EValue*> outputs) override;

 protected:
  // JIT-only — per-shape source assembled at dispatch via
  // ops/mlx_jit/KernelLoader. Base-class big-source path unused.
  const char* kernelSource() const override { return ""; }
};

}  // namespace metal_v2
}  // namespace backends
}  // namespace executorch
