/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//===----------------------------------------------------------------------===//
// Compiled with -fobjc-arc (see backends/metal/CMakeLists.txt). The
// libs_/psos_ maps store id<MTLLibrary> / id<MTLComputePipelineState>
// values which are __strong by default under ARC; map insertion retains
// and erase / clear releases automatically. The kernels_ map already
// uses std::unique_ptr<MetalKernel> for ownership.
//
// insertLibrary / insertPso do not consume a caller-owned +1. The
// __strong parameter retains on entry, the map retains on insert, and
// ARC releases the parameter at scope exit. For lost-race duplicates
// the cache does not manually release the parameter; the caller's +1
// is unchanged.
//===----------------------------------------------------------------------===//

#import "MetalKernelCache.h"

namespace executorch {
namespace backends {
namespace metal_v2 {

MetalKernelCache& MetalKernelCache::shared() {
  static MetalKernelCache instance;
  return instance;
}

MetalKernelCache::~MetalKernelCache() {
  // Process-singleton: this only runs at process exit.
}

//===----------------------------------------------------------------------===//
// Kernel store
//===----------------------------------------------------------------------===//

MetalKernel* MetalKernelCache::find(const std::string& key) {
  std::shared_lock<std::shared_mutex> lk(lock_);
  auto it = kernels_.find(key);
  return it == kernels_.end() ? nullptr : it->second.get();
}

MetalKernel* MetalKernelCache::findOrInsert(
    const std::string& key,
    const std::function<std::unique_ptr<MetalKernel>()>& factory) {
  // Fast path: shared-lock read. After warm-up, every call lands here.
  {
    std::shared_lock<std::shared_mutex> lk(lock_);
    auto it = kernels_.find(key);
    if (it != kernels_.end()) {
      return it->second.get();
    }
  }
  // Slow path: factory does ALL work (compile + slot-access parse + any
  // other init) WITHOUT holding any cache lock. This avoids serializing
  // PSO compiles across threads — multiple threads racing the same key
  // each compile independently; the cache picks one winner.
  std::unique_ptr<MetalKernel> built = factory();
  if (!built) {
    return nullptr;  // factory failed; nothing to cache
  }
  std::unique_lock<std::shared_mutex> lk(lock_);
  auto [it, inserted] = kernels_.try_emplace(key, std::move(built));
  // If we lost the race, `built` is destroyed (MetalKernel dtor releases
  // its PSO retain). Returned pointer is the winner's.
  return it->second.get();
}

//===----------------------------------------------------------------------===//
// Library sub-store
//===----------------------------------------------------------------------===//

id<MTLLibrary> MetalKernelCache::findLibrary(const std::string& key) {
  std::shared_lock<std::shared_mutex> lk(lock_);
  auto it = libs_.find(key);
  return it == libs_.end() ? nil : it->second;
}

void MetalKernelCache::insertLibrary(const std::string& key, id<MTLLibrary> lib) {
  if (!lib) return;
  std::unique_lock<std::shared_mutex> lk(lock_);
  (void)libs_.try_emplace(key, lib);
}

id<MTLLibrary> MetalKernelCache::insertOrFindLibrary(
    const std::string& key, id<MTLLibrary> lib) {
  if (!lib) return nil;
  std::unique_lock<std::shared_mutex> lk(lock_);
  // try_emplace returns {iter, inserted}; either way iter->second is
  // the canonical entry (the original on lost race, ours on first-touch).
  auto [it, inserted] = libs_.try_emplace(key, lib);
  return it->second;
}

//===----------------------------------------------------------------------===//
// Raw PSO sub-store
//===----------------------------------------------------------------------===//

id<MTLComputePipelineState> MetalKernelCache::findPso(const std::string& key) {
  std::shared_lock<std::shared_mutex> lk(lock_);
  auto it = psos_.find(key);
  return it == psos_.end() ? nil : it->second;
}

void MetalKernelCache::insertPso(
    const std::string& key, id<MTLComputePipelineState> pso) {
  if (!pso) return;
  std::unique_lock<std::shared_mutex> lk(lock_);
  (void)psos_.try_emplace(key, pso);
}

id<MTLComputePipelineState> MetalKernelCache::insertOrFindPso(
    const std::string& key, id<MTLComputePipelineState> pso) {
  if (!pso) return nil;
  std::unique_lock<std::shared_mutex> lk(lock_);
  auto [it, inserted] = psos_.try_emplace(key, pso);
  return it->second;
}

//===----------------------------------------------------------------------===//
// Testing
//===----------------------------------------------------------------------===//

void MetalKernelCache::resetForTesting() {
  std::unique_lock<std::shared_mutex> lk(lock_);
  kernels_.clear();  // unique_ptrs destroyed; MetalKernels release PSO retain
  libs_.clear();
  psos_.clear();
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch
