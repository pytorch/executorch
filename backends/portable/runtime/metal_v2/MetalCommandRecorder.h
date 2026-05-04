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
// MetalStream. Extracted from MetalStream as Step 2 of the god-class
// decomposition.
// Owns:
//   - IComputeBackend* backend_       : MTL3 dispatch path (always)
//   - MetalMTL4Backend* mtl4Backend_  : MTL4 dispatch path (lazy; ET_METAL4_ENABLE)
//   - psoWrapCache_                   : raw-PSO → MetalKernel wrappers for
//                                       the dispatch(PSO, ...) overload
//   - dispatchCount_, hasPendingWork_, flushInterval_  : auto-flush counters
// Borrows (non-owning, set at construction):
//   - id<MTLDevice>           : for backend construction
//   - id<MTLCommandQueue>     : passed to legacy backend
//   - MetalAllocator*         : for setInput/setOutput → bufferForPtr resolution
//   - MetalKernelCompiler*    : for setMTL4Backend wiring at ctor
//   - HazardTracker*          : for trackInput/trackOutput on every bind +
//                               needsBarrierForPending on every dispatch
// Public API surface mirrors what MetalStream used to expose for the
// dispatch path:
//   - typed setters (setInput/setOutput/setInOut/setBytes/setBuffer)
//   - Dispatch RAII (beginDispatch + chainable setX + run)
//   - free-form dispatch(kernel/pso, grid, block) for back-compat
//   - flush / wait / sync
//   - barrier stats accessors
// MetalStream keeps all the corresponding methods as one-liner delegating
// shims so existing op + AOTI shim call sites are unchanged.
//===----------------------------------------------------------------------===//

#include <executorch/backends/portable/runtime/metal_v2/HazardTracker.h>
#include <executorch/backends/portable/runtime/metal_v2/IComputeBackend.h>
#include <executorch/backends/portable/runtime/metal_v2/MetalKernel.h>
#include <executorch/backends/portable/runtime/metal_v2/MetalKernelCompiler.h>
#include <executorch/backends/portable/runtime/metal_v2/MetalTypes.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/runtime.h>

#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <vector>

#ifndef ET_METAL4_ENABLE
#define ET_METAL4_ENABLE 0
#endif

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
  // Slot binding (typed setters)
  //===--------------------------------------------------------------------===//

  void setInput(uint32_t slot, const void* ptr, size_t size);
  void setOutput(uint32_t slot, void* ptr, size_t size);
  void setInOut(uint32_t slot, void* ptr, size_t size);
  void setBytes(uint32_t slot, const void* ptr, size_t size);
  template <typename T>
  void setBytes(uint32_t slot, const T& value) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "setBytes<T>: T must be trivially copyable");
    setBytes(slot, &value, sizeof(value));
  }
  template <typename T>
  void setVectorBytes(uint32_t slot, const std::vector<T>& v) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "setVectorBytes<T>: T must be trivially copyable");
    setBytes(slot, v.data(), v.size() * sizeof(T));
  }

  // Hazard-aware MTLBuffer binders (see MetalStream.h for design rationale).
  void setInputBuffer(uint32_t slot, id<MTLBuffer> buf, size_t offset, size_t size);
  void setOutputBuffer(uint32_t slot, id<MTLBuffer> buf, size_t offset, size_t size);

  //===--------------------------------------------------------------------===//
  // Free-form dispatch (back-compat). New code should prefer Dispatch RAII.
  //===--------------------------------------------------------------------===//

  void dispatch(MetalKernel* kernel, uvec3 grid, uvec3 block);
  void dispatch(id<MTLComputePipelineState> pso, uvec3 grid, uvec3 block);

  //===--------------------------------------------------------------------===//
  // Scoped bind+dispatch (RAII). See MetalStream.h Dispatch doc for full
  // design rationale; this is the same class, just on the recorder now.
  //===--------------------------------------------------------------------===//

  class [[nodiscard]] Dispatch {
   public:
    Dispatch(Dispatch&& other) noexcept
        : recorder_(other.recorder_),
          kernel_(other.kernel_),
          pso_(other.pso_) {
      other.recorder_ = nullptr;
    }
    Dispatch& operator=(Dispatch&& other) noexcept {
      recorder_ = other.recorder_;
      kernel_ = other.kernel_;
      pso_ = other.pso_;
      other.recorder_ = nullptr;
      return *this;
    }
    Dispatch(const Dispatch&) = delete;
    Dispatch& operator=(const Dispatch&) = delete;
    ~Dispatch() = default;

    Dispatch& setInput(uint32_t slot, const void* ptr, size_t bytes);
    Dispatch& setOutput(uint32_t slot, void* ptr, size_t bytes);
    Dispatch& setInOut(uint32_t slot, void* ptr, size_t bytes);
    Dispatch& setBytes(uint32_t slot, const void* ptr, size_t bytes);
    template <typename T>
    Dispatch& setBytes(uint32_t slot, const T& value) {
      recorder_->setBytes(slot, value);
      return *this;
    }
    template <typename T>
    Dispatch& setVectorBytes(uint32_t slot, const std::vector<T>& v) {
      recorder_->setVectorBytes(slot, v);
      return *this;
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

  // Close the active compute encoder (if any). Used by MpsBridge before
  // handing the legacy command buffer off to MPSGraph so pending typed-
  // setter dispatches are sealed first.
  void endEncoder();

  // Mark that something committed work outside the recorder's normal
  // dispatch path (today: MpsBridge committed MPS work on the legacy
  // Diagnostics
  //===--------------------------------------------------------------------===//

  using BarrierStats = HazardTracker::BarrierStats;
  const BarrierStats& barrierStats() const { return hazards_->barrierStats(); }

  //===--------------------------------------------------------------------===//
  // Accessors needed by MpsBridge / cross-component wiring
  //===--------------------------------------------------------------------===//

  IComputeBackend* mtl3Backend() { return backend_.get(); }
#if ET_METAL4_ENABLE
  MetalMTL4Backend* mtl4Backend() { return mtl4Backend_.get(); }
#endif

  // Backend dispatch routes — public so MpsInterop / Stream can grab the
  // active legacy CB. Implemented by walking the MTL4-or-MTL3 toggle.
  IComputeBackend* dispatchBackend();

 private:
  // Internal: bind kernel + dispatch + commit hazards + auto-flush. Single
  // source of truth for the dispatch path (used by both dispatch(...) and
  // Dispatch::run).
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
#endif

  // Lazy cache: PSO identity -> MetalKernel wrapper. Used by the
  // dispatch(PSO, grid, block) overload so per-shape JIT consumers can
  // dispatch a raw PSO without paying the alloc cost of constructing a
  // MetalKernel on every dispatch.
  std::unordered_map<void*, std::unique_ptr<MetalKernel>> psoWrapCache_;

  // Auto-flush counters.
  bool hasPendingWork_ = false;
  int flushInterval_ = 40;
  int dispatchCount_ = 0;
};

} // namespace metal_v2
} // namespace backends
} // namespace executorch
