/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/portable/runtime/metal_v2/MetalOp.h>
#include <executorch/backends/portable/runtime/metal_v2/OpUtils.h>

namespace executorch {
namespace backends {
namespace metal_v2 {

//===----------------------------------------------------------------------===//
// BinaryOp - Base class for elementwise binary operations
//
// Variant classification, broadcast strides, and contiguous-dim collapsing
// all live in OpUtils.h (shared with other ops).
//===----------------------------------------------------------------------===//

class BinaryOp : public MetalOp {
public:
  virtual const char* opName() const = 0;
  virtual bool hasAlpha() const { return false; }

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

  std::string kernelName(ElementwiseVariant variant, ScalarType dtype) const;
};

//===----------------------------------------------------------------------===//
// Concrete Binary Ops
//===----------------------------------------------------------------------===//

class AddOp : public BinaryOp {
public:
  const char* name() const override { return "aten::add"; }
  const char* opName() const override { return "add"; }
  bool hasAlpha() const override { return true; }
};

class MulOp : public BinaryOp {
public:
  const char* name() const override { return "aten::mul"; }
  const char* opName() const override { return "mul"; }
};

class SubOp : public BinaryOp {
public:
  const char* name() const override { return "aten::sub"; }
  const char* opName() const override { return "sub"; }
};

} // namespace metal_v2
} // namespace backends
} // namespace executorch
