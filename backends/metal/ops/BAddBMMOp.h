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

// 3D batched fused matmul + bias (aten::baddbmm).
//   out[b] = beta * input[b] + alpha * (batch1[b] @ batch2[b])
// Reuses matmul_simd_addmm_t with grid.z = batch and use_out_source=true.

class BAddBMMOp : public MetalOp {
 public:
  const char* name() const override { return "aten::baddbmm"; }

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
