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
// SDPAOp — fused scaled-dot-product-attention.
// Registered as "aten::scaled_dot_product_attention.default" — the actual
// torch.export target name (verified via `torch.export(M, ...)` in the
// et-testing conda env). NOTE: existing matmul ops drop the `.default`
// overload suffix in their registry name (e.g. "aten::mm"); SDPA keeps
// the suffix per explicit request to match the exact aten op name.
// Schema (PyTorch aten::scaled_dot_product_attention.default):
//   inputs[0]  query       Tensor [B, Hq, qL, D]
//   inputs[1]  key         Tensor [B, Hkv, kL, D]
//   inputs[2]  value       Tensor [B, Hkv, kL, V]
//   inputs[3]  attn_mask   optional Tensor (None or [B, Hq, qL, kL] bool/float)
//   inputs[4]  dropout_p   double  (must be 0.0 — eval-only)
//   inputs[5]  is_causal   bool
//   inputs[6]  scale       optional double (None → 1/sqrt(D))
// Output:
//   outputs[0] Tensor [B, Hq, qL, V]
// Routes to one of three MLX-vendored kernel paths:
//   1. Vector single-pass     (qL ≤ 8 and short kL)
//   2. Vector 2-pass          (qL ≤ 8 and long kL or GQA + very long kL)
//   3. Steel attention        (qL > 8; NAX on Apple9+/non-fp32, else SIMD-MMA)
// Routing decision tree mirrors mlx/backend/metal/
// scaled_dot_product_attention.cpp:643-786 verbatim.
//===----------------------------------------------------------------------===//

class SDPAOp : public MetalOp {
 public:
  const char* name() const override {
    return "aten::scaled_dot_product_attention.default";
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
  // SDPA is JIT-only (per-shape kernel source assembled at dispatch via
  // ops/mlx_jit/KernelLoader). The base-class big-source MetalOp::getKernel
  // path is unused — return an empty string.
  const char* kernelSource() const override { return ""; }
};

}  // namespace metal_v2
}  // namespace backends
}  // namespace executorch
