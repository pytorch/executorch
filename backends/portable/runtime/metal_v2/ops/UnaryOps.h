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
// ReluOp
//===----------------------------------------------------------------------===//

class ReluOp : public MetalOp {
 public:
  const char* name() const override {
    return "aten::relu";
  }

  bool supports(ScalarType dtype) const override {
    return isFloatingPoint(dtype);
  }

  void dispatch(
      MetalStream* stream,
      ::executorch::runtime::Span<::executorch::runtime::EValue*> inputs,
      ::executorch::runtime::Span<::executorch::runtime::EValue*> outputs) override;

 protected:
  const char* kernelSource() const override;
};

} // namespace metal_v2
} // namespace backends
} // namespace executorch
