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

// 3D batched matmul (aten::bmm). Uses matmul_simd_addmm_t with
// use_out_source=false for the SIMD fast path; falls back to bmm_<dtype>
// for small problems where SIMD MMA has low occupancy.

class BatchedMatMulOp : public MetalOp {
 public:
  const char* name() const override { return "aten::bmm"; }

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
