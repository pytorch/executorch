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

// Fused 2D matmul + bias (aten::addmm).
//   out = beta * input + alpha * (mat1 @ mat2)
// Reuses the unified matmul_simd_addmm_t kernel with use_out_source=true.

class AddMMOp : public MetalOp {
 public:
  const char* name() const override { return "aten::addmm"; }

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
};

}  // namespace metal_v2
}  // namespace backends
}  // namespace executorch
