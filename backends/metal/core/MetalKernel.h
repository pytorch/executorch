/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

//===----------------------------------------------------------------------===//
// MetalKernel — Compiled Metal compute pipeline + name. Wraps an
// id<MTLComputePipelineState> and the kernel's MSL host_name. Held by
// MetalKernelCache (process-wide) and looked up by ops via getKernel().
//===----------------------------------------------------------------------===//

#import <Metal/Metal.h>

#include <executorch/backends/metal/core/MetalTypes.h>

#include <cstdint>
#include <string>

namespace executorch {
namespace backends {
namespace metal_v2 {

class MetalKernel {
public:
  MetalKernel(id<MTLComputePipelineState> pipeline, const char* name);
  ~MetalKernel();

  const char* name() const { return name_.c_str(); }
  uvec3 maxThreadsPerThreadgroup() const;

  id<MTLComputePipelineState> pipeline() const { return pipeline_; }

private:
  id<MTLComputePipelineState> pipeline_;
  std::string name_;
};

}  // namespace metal_v2
}  // namespace backends
}  // namespace executorch
