/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/portable/runtime/metal_v2/MetalTypes.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/core/span.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace executorch {
namespace backends {
namespace metal_v2 {

// Forward declarations — defined in MetalStream.h. We use forward decls
// here to avoid pulling Metal headers into pure-C++ op headers.
class MetalStream;
class MetalKernel;

using runtime::etensor::Tensor;
using exec_aten::ArrayRef;
using exec_aten::SizesType;

//===----------------------------------------------------------------------===//
// MetalOp - Base class for GPU operations
//===----------------------------------------------------------------------===//

class MetalOp {
public:
  virtual ~MetalOp() = default;

  /// Op name (e.g., "aten::add", "aten::mm")
  virtual const char* name() const = 0;

  /// Check if this op supports the given dtype
  virtual bool supports(ScalarType dtype) const {
    return dtype == ScalarType::Float;
  }

  /// Convenience alias for the EValue-pointer span used by op interfaces.
  /// Caller provides storage (typically a stack std::array) so dispatch is
  /// alloc-free per call.
  using EValuePtrSpan = runtime::Span<runtime::EValue*>;

  /// Compute output shape from inputs (for resize)
  /// Returns empty vector if output shape matches first input
  virtual std::vector<SizesType> computeOutputShape(
      EValuePtrSpan inputs) const {
    return {};
  }

  /// Dispatch the op using stream
  virtual void dispatch(
      MetalStream* stream,
      EValuePtrSpan inputs,
      EValuePtrSpan outputs) = 0;

protected:
  /// Get or compile kernel by name (caches result)
  MetalKernel* getKernel(MetalStream* stream, const char* kernelName);

  /// Kernel source code (subclass provides)
  virtual const char* kernelSource() const = 0;

  /// Compute grid size from output tensor
  uvec3 computeGrid(const Tensor& output, uint32_t blockSize = 256) const;

  /// Resize output tensor if needed
  runtime::Error resizeOutput(
      EValuePtrSpan inputs,
      runtime::EValue* output) const;

private:
  std::unordered_map<std::string, MetalKernel*> kernelCache_;
};

} // namespace metal_v2
} // namespace backends
} // namespace executorch
