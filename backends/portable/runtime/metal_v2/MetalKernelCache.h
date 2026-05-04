/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

//===----------------------------------------------------------------------===//
// MetalKernelCache — process-wide canonical store for compiled Metal kernels,
// libraries, and pipeline states.
// Three sub-stores, all keyed by string and protected by a single
// std::shared_mutex (read-mostly: hot path is cache-hit reads):
//   1. KERNEL store  — find / findOrInsert<MetalKernel>
//      The high-level API. Used by MetalOp::getKernel() and AOTI's kernel
//      shim. findOrInsert(key, factory) atomically takes ownership of a
//      caller-built unique_ptr<MetalKernel> on cache miss, or returns the
//      existing entry on hit. The factory is responsible for ALL setup
//      (compile + slot-access parsing + any other init) BEFORE the kernel
//      becomes visible — making post-publish concurrent mutation impossible.
//   2. LIBRARY sub-store — findLibrary / insertLibrary<MTLLibrary>
//      Used internally by MetalKernelCompiler so the big-source path's
//      MTLLibrary (which can host many [[host_name(...)]] kernels) is
//      shared across every (key, funcName) pair that derives from it.
//   3. RAW PSO sub-store — findPso / insertPso<MTLComputePipelineState>
//      Used by the per-shape JIT path (MetalKernelCompiler::
//      getOrCompilePsoFromSource → ops/mlx_jit/KernelLoader). Returns the
//      raw PSO without the MetalKernel wrapper because JIT callers don't
//      need slot-access metadata (they hardcode slot indices in dispatch
//      helpers).
// Why three stores instead of one (key → MetalKernel)?
//   The big-source path needs library-level deduping: ONE MTLLibrary hosts
//   N functions. A single (libKey, funcName) → MetalKernel map would
//   recompile the MSL N times. So the library store is genuinely separate.
//   The raw-PSO store is a perf optimization for the JIT path that doesn't
//   want wrapper-construction overhead per call.
// Thread-safety: all methods take the shared_mutex internally. Hot-path
// reads (find / findLibrary / findPso) take the shared lock; misses (insert
// / findOrInsert factory invocation) take the unique lock. Multiple
// concurrent readers see no contention.
// Lifetime: process-singleton. Cached entries are never evicted. PSOs and
// MTLLibraries are small (a few KiB each) and bounded by the model's
// unique kernel-shape set; if unbounded growth ever becomes a problem, add
// LRU eviction here. Each cached object holds +1 retain; released at
// process exit (dtor) or via resetForTesting().
// (Renamed from SharedPsoCache in 2026 — the old name implied "just PSOs",
// but the cache is now also the canonical kernel-wrapper store.)
//===----------------------------------------------------------------------===//

#include <executorch/backends/portable/runtime/metal_v2/MetalKernel.h>

#import <Metal/Metal.h>

#include <functional>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>

namespace executorch {
namespace backends {
namespace metal_v2 {

class MetalKernelCache {
 public:
  static MetalKernelCache& shared();

  //===--------------------------------------------------------------------===//
  // Kernel store (the high-level API; MetalKernel wrappers).
  //===--------------------------------------------------------------------===//

  // Look up an existing kernel by key. Returns nullptr on miss; never
  // compiles. Cheap shared-lock hash hit.
  MetalKernel* find(const std::string& key);

  // Atomic find-or-insert. On miss, invokes `factory` to build a fully
  // initialized kernel (compile + slot-access parse + any other setup),
  // then inserts. On a concurrent-insert race, the loser's unique_ptr
  // is dropped and the winning entry is returned.
  // The factory must do ALL initialization before returning — once the
  // kernel is in the cache it must be safe for any thread to dispatch
  // it. No setSlotAccess, no other mutator should be called on the
  // returned pointer.
  MetalKernel* findOrInsert(
      const std::string& key,
      const std::function<std::unique_ptr<MetalKernel>()>& factory);

  //===--------------------------------------------------------------------===//
  // Library sub-store (internal MTLLibrary deduping for the big-source +
  // per-shape JIT paths).
  //===--------------------------------------------------------------------===//

  id<MTLLibrary> findLibrary(const std::string& key);
  // Caller hands a +1 retained MTLLibrary. Cache consumes the +1 (keeps
  // on insert, releases on lost-race duplicate).
  void insertLibrary(const std::string& key, id<MTLLibrary> lib);

  //===--------------------------------------------------------------------===//
  // Raw PSO sub-store (per-shape JIT path; no wrapper).
  //===--------------------------------------------------------------------===//

  id<MTLComputePipelineState> findPso(const std::string& key);
  // Caller hands a +1 retained PSO. Cache consumes the +1.
  void insertPso(const std::string& key, id<MTLComputePipelineState> pso);

  //===--------------------------------------------------------------------===//
  // Test-only: drop everything. Production code never calls this — entries
  // outlive the process. Tests use it to keep cache state isolated between
  // cases.
  //===--------------------------------------------------------------------===//

  void resetForTesting();

 private:
  MetalKernelCache() = default;
  ~MetalKernelCache();
  MetalKernelCache(const MetalKernelCache&) = delete;
  MetalKernelCache& operator=(const MetalKernelCache&) = delete;

  // Single rwlock guards all three sub-stores. After warm-up, every call
  // is a shared-lock read; multi-thread inference sees no real contention
  // (atomic-only on the read path).
  mutable std::shared_mutex lock_;
  std::unordered_map<std::string, std::unique_ptr<MetalKernel>> kernels_;
  std::unordered_map<std::string, id<MTLLibrary>> libs_;
  std::unordered_map<std::string, id<MTLComputePipelineState>> psos_;
};

} // namespace metal_v2
} // namespace backends
} // namespace executorch
