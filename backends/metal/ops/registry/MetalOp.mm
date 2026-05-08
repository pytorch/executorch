/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalOp.h"
#include <executorch/runtime/platform/assert.h>
#include <executorch/backends/metal/core/MetalKernelCache.h>
#include <executorch/backends/metal/core/MetalStream.h>
#include <executorch/runtime/platform/log.h>

namespace executorch {
namespace backends {
namespace metal_v2 {

using runtime::Error;

//===----------------------------------------------------------------------===//
// MetalOp Base Implementation
//===----------------------------------------------------------------------===//

MetalKernel* MetalOp::getKernel(
    MetalStream* stream,
    const char* kernelName,
    const MetalKernelCompiler::FunctionConstants* constants) {
  // Cache key includes a hash of kernelSource() content so the same
  // kernelName appearing in different sources (e.g., AOTI's
  // "generated_kernel" emitted by Inductor for different graphs) doesn't
  // collide. Hashing string content (via std::hash<std::string_view>)
  // means two TUs that produce identical source via different pointers
  // still cache-hit.
  const char* source = kernelSource();
  std::string cacheKey = std::to_string(std::hash<std::string_view>{}(
                              std::string_view(source))) +
      "/" + kernelName +
      (constants ? constants->fingerprint() : std::string{});

  // Process-wide MetalKernelCache (the canonical store). On miss the
  // factory compiles and the cache atomically takes ownership; concurrent
  // threads racing the same key see the winner.
  MetalKernel* kernel = MetalKernelCache::shared().findOrInsert(
      cacheKey,
      [&]() {
        return stream->compiler()->compile(source, kernelName, constants);
      });

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
    ::executorch::runtime::Span<::executorch::runtime::EValue*> inputs,
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
