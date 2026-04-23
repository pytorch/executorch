/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalOp.h"
#include <executorch/runtime/platform/assert.h>
#include <executorch/backends/portable/runtime/metal_v2/MetalStream.h>
#include <executorch/runtime/platform/log.h>

namespace executorch {
namespace backends {
namespace metal_v2 {

using runtime::Error;

//===----------------------------------------------------------------------===//
// MetalOp Base Implementation
//===----------------------------------------------------------------------===//

MetalKernel* MetalOp::getKernel(MetalStream* stream, const char* kernelName) {
  auto it = kernelCache_.find(kernelName);
  if (it != kernelCache_.end()) {
    // Treat a previously cached null as a hard failure too — a cached null
    // means a prior compile attempt for this name failed.
    ET_CHECK_MSG(
        it->second != nullptr,
        "MetalOp '%s': previously failed to compile kernel '%s' (cached null). "
        "Most likely the kernel template wasn't instantiated for this dtype "
        "(check kernelSource() for the missing [[host_name(\"...\")]] entry).",
        name(), kernelName);
    return it->second;
  }

  MetalKernel* kernel = stream->compiler()->compile(kernelSource(), kernelName);
  kernelCache_[kernelName] = kernel;
  // Hard-fail on missing kernel rather than letting the dispatch silently
  // no-op. The previous behavior just logged at ERROR and returned, which
  // produced numerically wrong results that *looked* like a successful run
  // (e.g. fake "fast" perf because the kernel wasn't actually executing).
  ET_CHECK_MSG(
      kernel != nullptr,
      "MetalOp '%s': failed to compile/find kernel '%s'. "
      "Most likely the kernel template wasn't instantiated for this dtype "
      "(add a `template [[host_name(\"%s\")]] kernel void ...<T>(...)` line "
      "to kernelSource()'s template instantiation block).",
      name(), kernelName, kernelName);
  return kernel;
}

uvec3 MetalOp::computeGrid(const Tensor& output, uint32_t blockSize) const {
  size_t numel = output.numel();
  return uvec3((uint32_t)((numel + blockSize - 1) / blockSize), 1, 1);
}

Error MetalOp::resizeOutput(
    EValuePtrSpan inputs,
    runtime::EValue* output) const {

  if (!output->isTensor()) {
    return Error::InvalidArgument;
  }

  auto& out_tensor = output->toTensor();
  auto new_shape = computeOutputShape(inputs);

  if (new_shape.empty()) {
    if (!inputs.empty() && inputs[0]->isTensor()) {
      auto& in_tensor = inputs[0]->toTensor();
      new_shape.assign(in_tensor.sizes().begin(), in_tensor.sizes().end());
    }
  }

  if (!new_shape.empty()) {
    auto current = out_tensor.sizes();
    bool needs_resize = (current.size() != new_shape.size());
    if (!needs_resize) {
      for (size_t i = 0; i < current.size(); i++) {
        if (current[i] != new_shape[i]) {
          needs_resize = true;
          break;
        }
      }
    }

    if (needs_resize) {
      return runtime::resize_tensor(out_tensor, ArrayRef<SizesType>(new_shape.data(), new_shape.size()));
    }
  }

  return runtime::Error::Ok;
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch
