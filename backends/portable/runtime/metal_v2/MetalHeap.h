/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

//===----------------------------------------------------------------------===//
// MetalHeap - Pre-allocated memory pool for fast sub-allocation
// Owns one MTLHeap allocated up-front; sub-buffers come from the heap with
// O(100ns) cost vs O(10µs) for fresh device allocations. Reset is wholesale
// (no per-buffer free).
// Extracted from MetalStream.h — definition unchanged.
//===----------------------------------------------------------------------===//

#import <Metal/Metal.h>

#include <cstddef>

namespace executorch {
namespace backends {
namespace metal_v2 {

class MetalHeap {
public:
  MetalHeap(id<MTLDevice> device, size_t size, bool aliasable = false);
  ~MetalHeap();

  /// Allocate buffer from heap (fast: ~100ns vs ~10µs)
  id<MTLBuffer> allocBuffer(size_t size);

  /// Get current used size
  size_t usedSize() const { return usedSize_; }

  /// Get total heap size
  size_t totalSize() const { return totalSize_; }

  /// Reset heap (invalidates all buffers)
  void reset() { usedSize_ = 0; }

private:
  id<MTLHeap> heap_;
  size_t totalSize_;
  size_t usedSize_ = 0;
};

}  // namespace metal_v2
}  // namespace backends
}  // namespace executorch
