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

  /// Allocate buffer from heap (fast: ~100ns vs ~10µs).
  /// ARC-managed return: the buffer is returned autoreleased; callers
  /// that need to retain (e.g. BufferRegistry) should do so explicitly
  /// or via their own ARC __strong storage.
  id<MTLBuffer> allocBuffer(size_t size);

  /// Get current used size
  size_t usedSize() const { return usedSize_; }

  /// Get total heap size
  size_t totalSize() const { return totalSize_; }

  /// Native MTLHeap handle (borrowed; nil when ctor failed). Exposed so
  /// MetalAllocator can pass it to ResidencyManager::pinHeap at
  /// enableHeap time \u2014 the heap arena is the
  /// long-lived residency-set member that covers all heap-vended
  /// sub-buffers. Heap-vended id<MTLBuffer>s inherit residency from
  /// their parent heap, so the single pin at heap creation is enough
  /// for correctness; per-CB pin/unpin on heap-vended buffers becomes
  /// redundant but is left in place (uniformly refcounted) per the
  /// "don't skip in v1" tradeoff in the design doc.
  id<MTLHeap> nativeHeap() const { return heap_; }

private:
  id<MTLHeap> heap_;
  size_t totalSize_;
  size_t usedSize_ = 0;
};

}  // namespace metal_v2
}  // namespace backends
}  // namespace executorch
