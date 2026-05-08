/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

//===----------------------------------------------------------------------===//
// MetalCommandRecorder — owns the encoder + dispatch lifecycle for one
// MetalStream.
// Owns:
//   - IComputeBackend* backend_       : MTL3 dispatch path (always)
//   - MetalMTL4Backend* mtl4Backend_  : MTL4 dispatch path (lazy; ET_METAL4_ENABLE)
//   - psoWrapCache_                   : raw-PSO → MetalKernel wrappers for
//                                       the dispatch(PSO, ...) overload
//   - dispatchCount_, hasPendingWork_, flushInterval_  : auto-flush counters
// Borrows (non-owning, set at construction):
//   - id<MTLDevice>           : for backend construction
//   - id<MTLCommandQueue>     : passed to MTL3 backend
//   - MetalAllocator*         : for setInput/setOutput → bufferForPtr resolution
//   - MetalKernelCompiler*    : for setMTL4Backend wiring at ctor
//   - HazardTracker*          : for trackInput/trackOutput on every bind +
//                               needsBarrierForPending on every dispatch
// Public API surface for the dispatch path:
//   - typed setters (setInput/setOutput/setInOut/setBytes/setBuffer)
//   - Dispatch RAII (beginDispatch + chainable setX + run)
//   - free-form dispatch(kernel/pso, grid, block)
//   - flush / wait / sync
//   - barrier stats accessors
// MetalStream exposes the same methods as one-liner delegating shims so
// callers can use either spelling.
//===----------------------------------------------------------------------===//

#include <executorch/backends/metal/core/HazardTracker.h>
#include <executorch/backends/metal/core/IComputeBackend.h>
#include <executorch/backends/metal/core/MetalKernel.h>
#include <executorch/backends/metal/core/MetalKernelCompiler.h>
#include <executorch/backends/metal/core/MetalTypes.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

#import <Metal/Metal.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <executorch/backends/metal/core/MetalConfig.h>

namespace executorch {
namespace backends {
namespace metal_v2 {

class MetalAllocator;
class MetalMTL4Backend;

class MetalCommandRecorder {
 public:
  // Static cap on per-dispatch buffer slots (matches MTL4 argument-table
  // size). Public so MetalMTL4Backend can size its argument table to match.
  static constexpr size_t kMaxBuffersPerDispatch = 16;

  // Caps inline POD copies fed via setBytes (matches Metal's setBytes:
  // soft limit before performance falls off).
  static constexpr size_t kMaxInlineBytes = 4096;

  MetalCommandRecorder(
      id<MTLDevice> device,
      id<MTLCommandQueue> queue,
      MetalAllocator* allocator,
      MetalKernelCompiler* compiler,
      HazardTracker* hazards);
  ~MetalCommandRecorder();

  MetalCommandRecorder(const MetalCommandRecorder&) = delete;
  MetalCommandRecorder& operator=(const MetalCommandRecorder&) = delete;

  //===--------------------------------------------------------------------===//
  // Side-door encoder contract.
  //
  // Any code path that obtains the underlying MTL{,4}ComputeCommandEncoder
  // outside the typed-setter API MUST declare every MTLBuffer it intends
  // to bind via this method BEFORE invoking the side-door encoder. The
  // declared buffers are recorded in the same per-CB binds_ vector that
  // typed-setter binds populate, and pinBatch covers them at commit so
  // the residency set covers them.
  //
  // Failure mode if violated:
  //   MTL4: page fault on missing residency → undefined GPU behavior.
  //   MTL3: silent first-touch lazy paging → perf cliff, no correctness loss.
  //
  // Current consumer: MpsInterop::encodeWithLegacyCommandBuffer.
  //===--------------------------------------------------------------------===//

  // Public API takes raw (pointer, count) instead of std::span to
  // avoid imposing C++20 on every header consumer.
  void declareSideDoorBinds(id<MTLBuffer> const __unsafe_unretained* bufs, size_t count);

  //===--------------------------------------------------------------------===//
  // Scoped bind+dispatch (RAII).
  //
  // Canonical dispatch API. `beginDispatch(kernel_or_pso)` returns a
  // [[nodiscard]] Dispatch object; chain typed setters and call `.run(grid,
  // block)` to dispatch. `[[nodiscard]]` catches forgotten dispatches at
  // compile time; the captured kernel can't be silently rebound; move-only
  // semantics prevent a single Dispatch from being run twice.
  //===--------------------------------------------------------------------===//

  class [[nodiscard]] Dispatch {
   public:
    Dispatch(Dispatch&& other) noexcept
        : recorder_(other.recorder_),
          kernel_(other.kernel_),
          pso_(other.pso_) {
      other.recorder_ = nullptr;
      other.kernel_ = nullptr;
      other.pso_ = nil;
    }
    Dispatch& operator=(Dispatch&& other) noexcept {
      if (this != &other) {
        // Reassigning over a live Dispatch silently abandons its bindings
        // (already pushed into the encoder) and never calls run() —
        // breaks the "exactly one run() per Dispatch" invariant.
        ET_DCHECK_MSG(
            recorder_ == nullptr,
            "Dispatch::operator=: would discard live dispatch state. "
            "Call .run() on the existing Dispatch before reassigning.");
        recorder_ = other.recorder_;
        kernel_ = other.kernel_;
        pso_ = other.pso_;
        other.recorder_ = nullptr;
      }
      return *this;
    }
    Dispatch(const Dispatch&) = delete;
    Dispatch& operator=(const Dispatch&) = delete;
    ~Dispatch() {
      // Forgetting .run() pushes bindings to the encoder that the next
      // dispatch then overwrites — wasted work, semantic ambiguity. The
      // [[nodiscard]] on beginDispatch only catches the rvalue temp form
      // (`rec.beginDispatch(...);` discarded). The lvalue form
      // (`auto d = rec.beginDispatch(...)`) consumes the return value
      // and silences the warning, so the dtor check is the only line
      // of defense for `auto d = ...; ... ; /* forgot d.run() */ }`.
      ET_DCHECK_MSG(
          recorder_ == nullptr,
          "Dispatch destroyed without calling .run(). The lvalue form "
          "(auto d = recorder.beginDispatch(...)) requires an explicit "
          "d.run(grid, block) before scope exit.");
    }

    Dispatch& setInput(uint32_t slot, const void* ptr, size_t bytes);
    Dispatch& setOutput(uint32_t slot, void* ptr, size_t bytes);
    Dispatch& setInOut(uint32_t slot, void* ptr, size_t bytes);
    Dispatch& setBytes(uint32_t slot, const void* ptr, size_t bytes);
    template <typename T>
    Dispatch& setBytes(uint32_t slot, const T& value) {
      static_assert(std::is_trivially_copyable<T>::value,
                    "setBytes<T>: T must be trivially copyable");
      return setBytes(slot, &value, sizeof(value));
    }
    template <typename T>
    Dispatch& setVectorBytes(uint32_t slot, const std::vector<T>& v) {
      static_assert(std::is_trivially_copyable<T>::value,
                    "setVectorBytes<T>: T must be trivially copyable");
      return setBytes(slot, v.data(), v.size() * sizeof(T));
    }
    Dispatch& setInputBuffer(uint32_t slot, id<MTLBuffer> buf, size_t off, size_t sz);
    Dispatch& setOutputBuffer(uint32_t slot, id<MTLBuffer> buf, size_t off, size_t sz);

    void run(uvec3 grid, uvec3 block);

   private:
    friend class MetalCommandRecorder;
    Dispatch(MetalCommandRecorder* r, MetalKernel* kernel)
        : recorder_(r), kernel_(kernel), pso_(nil) {}
    Dispatch(MetalCommandRecorder* r, id<MTLComputePipelineState> pso)
        : recorder_(r), kernel_(nullptr), pso_(pso) {}

    MetalCommandRecorder* recorder_;
    MetalKernel* kernel_ = nullptr;
    id<MTLComputePipelineState> pso_ = nil;
  };

  Dispatch beginDispatch(MetalKernel* kernel) {
    ET_CHECK_MSG(kernel != nullptr, "MetalCommandRecorder::beginDispatch: kernel is null");
    return Dispatch{this, kernel};
  }
  Dispatch beginDispatch(id<MTLComputePipelineState> pso) {
    ET_CHECK_MSG(pso != nil, "MetalCommandRecorder::beginDispatch: pso is nil");
    return Dispatch{this, pso};
  }

  //===--------------------------------------------------------------------===//
  // Lifecycle (encoder + command buffer)
  //===--------------------------------------------------------------------===//

  void flush();
  void wait();
  void sync() { flush(); wait(); }

  void setFlushInterval(int dispatches);
  int flushInterval() const { return flushInterval_; }

  // Close the active compute encoder (if any). Used by MpsInterop before
  // handing the legacy command buffer off to MPSGraph so pending typed-
  // setter dispatches are sealed first.
  void endEncoder();

  // Mark that something committed work outside the recorder's normal
  // dispatch path (today: MpsInterop committed MPS work on the legacy CB).
  //===--------------------------------------------------------------------===//
  // Diagnostics
  //===--------------------------------------------------------------------===//

  using BarrierStats = HazardTracker::BarrierStats;
  const BarrierStats& barrierStats() const { return hazards_->barrierStats(); }

  //===--------------------------------------------------------------------===//
  // Accessors needed by MpsInterop / cross-component wiring
  //===--------------------------------------------------------------------===//

  IComputeBackend* mtl3Backend() { return backend_.get(); }
#if ET_METAL4_ENABLE
  MetalMTL4Backend* mtl4Backend() { return mtl4Backend_.get(); }
#endif

  // Backend dispatch routes — public so MpsInterop / Stream can grab the
  // active legacy CB. Implemented by walking the MTL4-or-MTL3 toggle.
  IComputeBackend* dispatchBackend();

  //===--------------------------------------------------------------------===//
  // Test-only accessors. NOT for production code paths.
  //===--------------------------------------------------------------------===//

  // Number of distinct MTLBuffers recorded in the current per-CB binds_
  // vector (after dedup). Used by tests to verify the side-door contract.
  size_t boundBufferCountForTesting() const { return binds_.size(); }

  // True iff `buf` is recorded in the current per-CB binds_ vector.
  bool isBoundForTesting(id<MTLBuffer> buf) const {
    return bound_buffers_.find((__bridge void*)buf) != bound_buffers_.end();
  }

 private:
  // Internal typed setters \u2014 callable only via the friended Dispatch class.
  // External callers must use beginDispatch(...).setX(...).run(...).
  void setInput(uint32_t slot, const void* ptr, size_t size);
  void setOutput(uint32_t slot, void* ptr, size_t size);
  void setInOut(uint32_t slot, void* ptr, size_t size);
  void setBytes(uint32_t slot, const void* ptr, size_t size);
  void setInputBuffer(uint32_t slot, id<MTLBuffer> buf, size_t offset, size_t size);
  void setOutputBuffer(uint32_t slot, id<MTLBuffer> buf, size_t offset, size_t size);

  // Internal: record `buf` in the per-CB binds_ vector, deduplicating
  // against bound_buffers_. Identity-keyed by (__bridge void*)id; no
  // retain (the caller / registry holds the strong ref). No-op on nil.
  void recordBind(id<MTLBuffer> buf);

  // Reset per-CB binds state. Called at CB commit boundaries.
  void clearBinds() {
    binds_.clear();
    bound_buffers_.clear();
#if !defined(NDEBUG)
    // side-door debug audit: reset per-CB tracking
    // when the CB commits.
    side_door_invoked_ = false;
    side_door_declared_.clear();
#endif
  }
  // Internal: bind kernel + dispatch + commit hazards + auto-flush. Single
  // source of truth for the dispatch path (used by Dispatch::run).
  void doDispatchKernel(MetalKernel* kernel, uvec3 grid, uvec3 block);
  void ensureCommandBuffer();
  void flushCommitLegacy();

  // Marks pending work so the next flush() doesn't early-return on
  // hasPendingWork_=false. Called by MpsInterop after committing MPS-only
  // work on the legacy queue. Friend so the API doesn't leak publicly.
  void noteMpsWorkPending() { hasPendingWork_ = true; }

  friend class Dispatch;
  friend class MpsInterop;

  // Borrowed (non-owning).
  id<MTLDevice> device_;
  id<MTLCommandQueue> queue_;
  MetalAllocator* allocator_;
  MetalKernelCompiler* compiler_;
  HazardTracker* hazards_;

  // Owned.
  std::unique_ptr<IComputeBackend> backend_;  // MTL3 (always present)
#if ET_METAL4_ENABLE
  std::unique_ptr<MetalMTL4Backend> mtl4Backend_;  // lazy, may be null
  // Cached: non-null iff useMTL4() && mtl4Backend_ && mtl4Backend_->isReady()
  // at construction time. Use this at hot-path call sites to dispatch on
  // the MTL4 backend; falls back to backend_ (MTL3) when null. Saves the
  // triple-AND check on every dispatch.
  MetalMTL4Backend* mtl4BackendIfReady_ = nullptr;
#endif

  // Lazy LRU cache: PSO identity -> MetalKernel wrapper. Used by
  // Dispatch::run on the raw-PSO path so per-shape JIT consumers can
  // dispatch a raw PSO without paying the alloc cost of constructing a
  // MetalKernel on every dispatch.
  // Bounded LRU (kPsoWrapCacheCap entries). Cap chosen high enough for
  // any realistic single-model workload (typical: tens to low-hundreds
  // of unique PSOs); per-shape JIT churn in long-running serving never
  // grows unboundedly.
  static constexpr size_t kPsoWrapCacheCap = 256;
  struct PsoEntry {
    void* key;
    std::unique_ptr<MetalKernel> kernel;
  };
  std::list<PsoEntry> psoWrapLru_;  // newest at front
  std::unordered_map<void*, std::list<PsoEntry>::iterator> psoWrapIndex_;

  // Auto-flush counters.
  bool hasPendingWork_ = false;
  int flushInterval_ = 40;
  int dispatchCount_ = 0;

  // Per-CB residency-bind tracking.
  // Identity-keyed by (__bridge void*)id<MTLBuffer>. Buffer lifetime is
  // NOT owned here — caller / allocator / pool holds the strong ref;
  // the residency set's addAllocation: retains for membership lifetime.
  // bound_buffers_ is the dedup set; binds_ is the ordered vector handed
  // to pinBatch at commit. Cleared on every CB commit boundary via
  // clearBinds(). On commit, pinBatch is followed by capturing binds_
  // into a completion handler that calls unpinBatch (no accumulator).
  std::unordered_set<void*> bound_buffers_;
  std::vector<void*> binds_;

#if !defined(NDEBUG)
  // Side-door encoder contract — Debug enforcement:
  // tracks declarations made via declareSideDoorBinds for the current CB.
  // flush() runs an audit pass before scheduling the unpin handler:
  // after pinBatch, verify each declared buffer is actually present in
  // the residency set's allAllocations. Catches caller bugs (nil/stale
  // buffer) and recorder/ResidencyManager bugs (declared but not
  // pinned).
  //
  // Identity-keyed (__bridge void*) — same scheme as binds_. Reset in
  // clearBinds() at every CB commit boundary. Release builds elide
  // both fields and the audit code entirely.
  bool side_door_invoked_ = false;
  std::vector<void*> side_door_declared_;
#endif

  // Per-CB unpin via completion handler
  // (per-CB unpin path).
  //
  // flush() schedules a completion handler (via the active backend's
  // addCompletionHandler) that captures the just-pinned binds + a
  // shared_ptr<ResidencyManager> + this counter. The handler runs
  // unpinBatch on those binds and decrements the counter.
  //
  // wait() blocks on this counter (in addition to backend->wait()) so
  // that the stream dtor can safely destroy MetalAllocator without
  // racing MTL4's feedback handler (which fires asynchronously on
  // Apple's queue AFTER the GPU signal — backend->wait() returns
  // before the handler runs).
  //
  // shared_ptr keeps the state alive even if the recorder is destroyed
  // mid-flight (defense in depth — wait() in dtor should already drain).
  // Condvar form: handler decrements `count` and notifies; wait() blocks
  // on the cv until count==0. No CPU spin while Apple's queue is delayed.
  struct PendingHandlerState {
    std::mutex mu;
    std::condition_variable cv;
    int count = 0;
  };
  std::shared_ptr<PendingHandlerState> pendingHandlers_ =
      std::make_shared<PendingHandlerState>();
};

} // namespace metal_v2
} // namespace backends
} // namespace executorch
