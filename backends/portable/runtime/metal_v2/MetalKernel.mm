/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalStream.h"
#include <executorch/runtime/platform/log.h>

namespace executorch {
namespace backends {
namespace metal_v2 {

//===----------------------------------------------------------------------===//
// MetalKernel
//===----------------------------------------------------------------------===//

MetalKernel::MetalKernel(id<MTLComputePipelineState> pipeline, const char* name)
    : pipeline_(pipeline), name_(name) {
  [pipeline_ retain];
}

MetalKernel::~MetalKernel() {
  [pipeline_ release];
}

uvec3 MetalKernel::maxThreadsPerThreadgroup() const {
  NSUInteger maxThreads = [pipeline_ maxTotalThreadsPerThreadgroup];
  return uvec3(static_cast<uint32_t>(maxThreads), 1, 1);
}


} // namespace metal_v2
} // namespace backends
} // namespace executorch

