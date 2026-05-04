/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalKernelCache.h"

namespace executorch {
namespace backends {
namespace metal_v2 {

MetalKernelCache& MetalKernelCache::shared() {
  static MetalKernelCache instance;
  return instance;
}

MetalKernelCache::~MetalKernelCache() {
  // Process-singleton: this only runs at process exit. Release the +1
  // retain we hold on each cached MTLLibrary / PSO so leak detectors are
  // clean. kernels_ owns its MetalKernels via unique_ptr — they release
  // their PSO retain in MetalKernel::~MetalKernel.
  for (auto& [k, lib] : libs_) {
    if (lib) [lib release];
  }
  for (auto& [k, pso] : psos_) {
    if (pso) [pso release];
  }
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
  // Caller hands us a +1 retained object. We consume that +1: keep it on
  // first insert, or release it if we lost the race to another thread.
  auto [it, inserted] = libs_.try_emplace(key, lib);
  if (!inserted) {
    [lib release];
  }
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
  // Same +1 ownership contract as insertLibrary.
  auto [it, inserted] = psos_.try_emplace(key, pso);
  if (!inserted) {
    [pso release];
  }
}

//===----------------------------------------------------------------------===//
// Testing
//===----------------------------------------------------------------------===//

void MetalKernelCache::resetForTesting() {
  std::unique_lock<std::shared_mutex> lk(lock_);
  kernels_.clear();  // unique_ptrs destroyed; MetalKernels release PSO retain
  for (auto& [k, lib] : libs_) {
    if (lib) [lib release];
  }
  libs_.clear();
  for (auto& [k, pso] : psos_) {
    if (pso) [pso release];
  }
  psos_.clear();
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch
