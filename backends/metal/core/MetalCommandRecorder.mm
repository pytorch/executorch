/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import "MetalCommandRecorder.h"

#import "MetalAllocator.h"
#import "MetalMTL3Backend.h"
#if ET_METAL4_ENABLE
#import "MetalMTL4Backend.h"
#endif
#import "MetalStream.h"  // for useMTL4()

#include <thread>

namespace executorch {
namespace backends {
namespace metal_v2 {

MetalCommandRecorder::MetalCommandRecorder(
    id<MTLDevice> device,
    id<MTLCommandQueue> queue,
    MetalAllocator* allocator,
    MetalKernelCompiler* compiler,
    HazardTracker* hazards)
    : device_(device),
      queue_(queue),
      allocator_(allocator),
      compiler_(compiler),
      hazards_(hazards) {
  // MTL3 dispatch path. Always created — even under MTL4, the MTL3
  // backend is used by MPS interop (which requires a legacy CB).
  backend_ = std::make_unique<MetalMTL3Backend>(device_, queue_);

#if ET_METAL4_ENABLE
  // Lazy MTL4 backend if the runtime opt-in is set.
  if (useMTL4()) {
    if (@available(macOS 26.0, iOS 26.0, *)) {
      mtl4Backend_ = std::make_unique<MetalMTL4Backend>(
          device_, allocator_->residency());
      if (mtl4Backend_->isReady()) {
        mtl4BackendIfReady_ = mtl4Backend_.get();
        if (allocator_->residency()->isEnabled()) {
          mtl4Backend_->addQueueResidency(allocator_->residency()->nativeSet());
        }
      }
      // Wire MTL4 compiler through the backend so the kernel compiler
      // can reach mtl4Compiler_ on demand.
      compiler_->setMTL4Backend(mtl4Backend_.get());
    }
  }
#endif
}

MetalCommandRecorder::~MetalCommandRecorder() {
#if ET_METAL4_ENABLE
  if (@available(macOS 26.0, iOS 26.0, *)) {
    mtl4Backend_.reset();
  }
#endif
  // backend_ is a unique_ptr<IComputeBackend>; its dtor (MetalMTL3Backend's)
  // releases the MTL3 CB / encoder / in-flight CB.
}

//===----------------------------------------------------------------------===//
// Backend routing
//===----------------------------------------------------------------------===//

IComputeBackend* MetalCommandRecorder::dispatchBackend() {
#if ET_METAL4_ENABLE
  if (mtl4BackendIfReady_) {
    return mtl4BackendIfReady_;
  }
#endif
  return backend_.get();
}

void MetalCommandRecorder::ensureCommandBuffer() {
#if ET_METAL4_ENABLE
  if (mtl4BackendIfReady_) {
    if (@available(macOS 26.0, iOS 26.0, *)) {
      mtl4BackendIfReady_->ensureCommandBuffer();
      return;  // MTL4 path active; MTL3 buffer not needed.
    }
  }
#endif
  // MTL3 path: backend_ is a MetalMTL3Backend which provides
  // ILegacyCommandBufferProvider via the virtual hook.
  auto* legacy = backend_->legacyCommandBufferProvider();
  ET_CHECK_MSG(legacy != nullptr,
               "MetalCommandRecorder: MTL3 backend does not provide "
               "ILegacyCommandBufferProvider");
  (void)legacy->ensureLegacyCommandBuffer();
}

void MetalCommandRecorder::endEncoder() {
#if ET_METAL4_ENABLE
  if (mtl4Backend_) {
    if (@available(macOS 26.0, iOS 26.0, *)) {
      mtl4Backend_->endEncoder();
    }
  }
#endif
  backend_->endEncoder();
}


//===----------------------------------------------------------------------===//
// flush / wait / sync
//===----------------------------------------------------------------------===//

void MetalCommandRecorder::flush() {
  if (!hasPendingWork_) {
    return;
  }
  // reset the auto-flush counter so an explicit flush() resets
  // the cycle. Without this, 39 dispatches → user flush() → 1 more
  // dispatch would auto-flush an empty encoder (harmless but wasteful).
  dispatchCount_ = 0;

  endEncoder();

  // Per-CB residency-bind tracking (completion-
  // handler-driven unpin).
  //
  // Pin the just-encoded CB's binds in the residency set NOW (so the
  // CB sees them resident at execution), then schedule a completion
  // handler that unpins them after the GPU is done. The handler
  // captures:
  //   - the binds vector by value (ARC retains each MTLBuffer for the
  //     handler's lifetime, keeping pointers valid for unpinBatch);
  //   - a shared_ptr<ResidencyManager> (keeps the manager alive even
  //     if MetalAllocator is destroyed before the handler fires —
  //     important for MTL4's MTL4CommitFeedback which runs async on
  //     Apple's queue after the GPU signal);
  //   - a shared_ptr<atomic<int>> counter that wait() blocks on, so
  //     the stream dtor can synchronize with pending handlers.
  // allocator_ and allocator_->residency() are both required non-null
  // (allocator_ is required at construction; residency() is unconditionally
  // constructed in MetalAllocator's ctor). Only the empty-binds early-out
  // is meaningful.
  if (!binds_.empty()) {
    // Invariant: binds_ non-empty ⇒ at least one typed setter ran on
    // this CB ⇒ ensureEncoder opened a CB ⇒ the active backend's
    // commit() below cannot early-return on !commandBuffer_. The
    // pending-handler counter increment below is therefore guaranteed
    // to be paired with a completion-handler decrement, so wait()'s
    // condvar predicate will eventually observe count==0.
    auto residency = allocator_->residencyShared();
    std::vector<id<MTLBuffer>> as_id;
    as_id.reserve(binds_.size());
    for (void* p : binds_) as_id.push_back((__bridge id<MTLBuffer>)p);
    residency->pinBatch(as_id.data(), as_id.size());

#if !defined(NDEBUG) && ET_METAL4_AVAILABLE
    // Side-door encoder contract — Debug enforcement:
    // if any declareSideDoorBinds call ran for this CB, verify each
    // declared buffer is now present in the residency set's
    // allAllocations. ET_DCHECK on miss — caller passed a nil/stale
    // buffer or our pinBatch skipped it. Release builds elide.
    if (side_door_invoked_) {
      if (@available(macOS 15.0, iOS 18.0, *)) {
        id<MTLResidencySet> set = residency->nativeSet();
        if (set) {
          NSArray<id<MTLAllocation>>* members = [set allAllocations];
          // Wrap in an NSSet for O(1) containsObject lookups; the loop
          // below would otherwise be O(N×M) since NSArray containsObject
          // is O(N).
          NSSet<id<MTLAllocation>>* memberSet = [NSSet setWithArray:members];
          for (void* p : side_door_declared_) {
            id<MTLAllocation> needle = (__bridge id<MTLAllocation>)p;
            ET_DCHECK_MSG(
                [memberSet containsObject:needle],
                "Side-door bind not pinned: buffer %p declared via "
                "declareSideDoorBinds is not present in MTLResidencySet "
                "after pinBatch. Caller passed a nil/stale id<MTLBuffer>, "
                "or recorder/ResidencyManager skipped it.", p);
          }
        }
      }
    }
#endif

    {
      std::lock_guard<std::mutex> lk(pendingHandlers_->mu);
      pendingHandlers_->count += 1;
    }
    auto pendingHandlers = pendingHandlers_;  // shared_ptr copy
    dispatchBackend()->addCompletionHandler(
        [residency, captured_binds = std::move(as_id), pendingHandlers]() {
          // Note: captured_binds is const inside the lambda by default;
          // unpinBatch doesn't mutate it, just reads pointers.
          residency->unpinBatch(
              const_cast<id<MTLBuffer>*>(captured_binds.data()),
              captured_binds.size());
          {
            std::lock_guard<std::mutex> lk(pendingHandlers->mu);
            pendingHandlers->count -= 1;
          }
          pendingHandlers->cv.notify_all();
        });
    clearBinds();
  }

  // Metal 4: Commit residency set before execution.
  // (allocator_ is required at construction; residency() is unconditionally
  // constructed in MetalAllocator's ctor.)
  allocator_->residency()->commit();

#if ET_METAL4_ENABLE
  if (mtl4BackendIfReady_) {
    if (@available(macOS 26.0, iOS 26.0, *)) {
      mtl4BackendIfReady_->commit();
    }
  }
#endif
  flushCommitLegacy();

  hasPendingWork_ = false;
  // Encoder boundary — work has been committed and any subsequent
  // dispatch on a fresh encoder doesn't have a hazard with prior dispatches.
  hazards_->reset();
}

void MetalCommandRecorder::flushCommitLegacy() {
  // Commits the MTL3 (legacy) command buffer if one is pending. Under
  // MTL4 the MTL3 backend's commandBuffer_ is normally nil (only set when
  // MPSGraph routed work through the legacy path), so backend_->commit()
  // is a no-op in that case. Skip the call entirely under MTL4 to make
  // the intent explicit.
#if ET_METAL4_ENABLE
  if (mtl4BackendIfReady_) return;
#endif
  backend_->commit();
}

void MetalCommandRecorder::wait() {
  flush();
#if ET_METAL4_ENABLE
  if (mtl4Backend_) {
    if (@available(macOS 26.0, iOS 26.0, *)) {
      mtl4Backend_->wait();
    }
  }
#endif
  backend_->wait();
  // Block until any pending completion handlers have finished running.
  // backend_->wait() returns after the GPU signals, but MTL4's
  // MTL4CommitFeedback handler runs asynchronously on Apple's queue
  // AFTER the signal — without this wait, the stream dtor could destroy
  // MetalAllocator while a handler is still executing unpinBatch (the
  // shared_ptr capture would keep the manager alive in that case, but
  // the semantics are cleaner when wait() actually drains).
  // Condvar form: handler does notify_all() after decrementing count;
  // we block on cv.wait until count==0. No CPU spin while Apple's queue
  // is delayed.
  {
    std::unique_lock<std::mutex> lk(pendingHandlers_->mu);
    pendingHandlers_->cv.wait(lk, [&] { return pendingHandlers_->count == 0; });
  }
  // MPS-pending drain stays on MetalStream::wait() — MetalCommandRecorder
  // doesn't know about MpsInterop.
}

void MetalCommandRecorder::setFlushInterval(int dispatches) {
  flushInterval_ = dispatches;
}

//===----------------------------------------------------------------------===//
// Slot binding (typed setters)
//===----------------------------------------------------------------------===//

void MetalCommandRecorder::setInput(uint32_t slot, const void* ptr, size_t size) {
  void* mut_ptr = const_cast<void*>(ptr);
  auto bind = allocator_->bufferForPtr(mut_ptr, size);
  if (bind.mtl) {
    dispatchBackend()->setBuffer(bind.mtl, bind.offset, slot);
    hazards_->trackInput(bind.mtl, bind.offset, bind.offset + size);
    recordBind(bind.mtl);
    ET_LOG(Debug, "setInput slot=%u ptr=%p size=%zu mtl=%p offset=%zu",
           slot, ptr, size, (__bridge void*)bind.mtl, bind.offset);
  } else {
    ET_LOG(Debug, "MetalCommandRecorder::setInput: Using setBytes for ptr %p (%zu bytes)",
           ptr, size);
    dispatchBackend()->setBytes(ptr, size, slot);
  }
}

void MetalCommandRecorder::setOutput(uint32_t slot, void* ptr, size_t size) {
  auto bind = allocator_->bufferForPtr(ptr, size);
  if (bind.mtl) {
    dispatchBackend()->setBuffer(bind.mtl, bind.offset, slot);
    hazards_->trackOutput(bind.mtl, bind.offset, bind.offset + size);
    recordBind(bind.mtl);
    ET_LOG(Debug, "setOutput slot=%u ptr=%p size=%zu mtl=%p offset=%zu",
           slot, ptr, size, (__bridge void*)bind.mtl, bind.offset);
  } else {
    ET_LOG(Debug, "MetalCommandRecorder::setOutput: Using setBytes for ptr %p (%zu bytes)",
           ptr, size);
    dispatchBackend()->setBytes(ptr, size, slot);
  }
}

void MetalCommandRecorder::setInOut(uint32_t slot, void* ptr, size_t size) {
  auto bind = allocator_->bufferForPtr(ptr, size);
  if (bind.mtl) {
    dispatchBackend()->setBuffer(bind.mtl, bind.offset, slot);
    hazards_->trackInput(bind.mtl, bind.offset, bind.offset + size);
    hazards_->trackOutput(bind.mtl, bind.offset, bind.offset + size);
    recordBind(bind.mtl);
    ET_LOG(Debug, "setInOut slot=%u ptr=%p size=%zu mtl=%p offset=%zu",
           slot, ptr, size, (__bridge void*)bind.mtl, bind.offset);
  } else {
    ET_LOG(Debug, "MetalCommandRecorder::setInOut: Using setBytes for ptr %p (%zu bytes)",
           ptr, size);
    dispatchBackend()->setBytes(ptr, size, slot);
  }
}

void MetalCommandRecorder::setBytes(uint32_t slot, const void* ptr, size_t size) {
  ET_CHECK_MSG(size <= kMaxInlineBytes,
               "MetalCommandRecorder::setBytes: size %zu exceeds %zu cap; "
               "use a real buffer for larger payloads.",
               size, kMaxInlineBytes);
  dispatchBackend()->setBytes(ptr, size, slot);
}

void MetalCommandRecorder::setInputBuffer(
    uint32_t slot, id<MTLBuffer> buf, size_t offset, size_t size) {
  dispatchBackend()->setBuffer(buf, offset, slot);
  hazards_->trackInput(buf, offset, offset + size);
  recordBind(buf);
}

void MetalCommandRecorder::setOutputBuffer(
    uint32_t slot, id<MTLBuffer> buf, size_t offset, size_t size) {
  dispatchBackend()->setBuffer(buf, offset, slot);
  hazards_->trackOutput(buf, offset, offset + size);
  recordBind(buf);
}

//===----------------------------------------------------------------------===//
// Per-CB residency-bind tracking.
//
// Identity-key extraction via __bridge cast. We store void* (not
// id<MTLBuffer>) to keep the storage Obj-C-lifetime-neutral; buffer
// lifetime is owned by the registry/pool/external caller. flush()
// bridges back to id<MTLBuffer> at pinBatch time, and the residency
// set's addAllocation: retains for membership lifetime.
//===----------------------------------------------------------------------===//

void MetalCommandRecorder::recordBind(id<MTLBuffer> buf) {
  if (!buf) return;
  void* key = (__bridge void*)buf;
  if (bound_buffers_.insert(key).second) {
    binds_.push_back(key);
  }
}

void MetalCommandRecorder::declareSideDoorBinds(
    id<MTLBuffer> const __unsafe_unretained* bufs, size_t count) {
  for (size_t i = 0; i < count; ++i) recordBind(bufs[i]);
#if !defined(NDEBUG)
  // Side-door encoder contract — Debug enforcement:
  // mark this CB as having a side-door consumer and remember which
  // buffers were declared. flush() will verify each declared buffer
  // ends up in the residency set after pinBatch.
  side_door_invoked_ = true;
  for (size_t i = 0; i < count; ++i) {
    if (bufs[i]) side_door_declared_.push_back((__bridge void*)bufs[i]);
  }
#endif
}

//===----------------------------------------------------------------------===//
// Dispatch path
//===----------------------------------------------------------------------===//

void MetalCommandRecorder::doDispatchKernel(
    MetalKernel* kernel, uvec3 grid, uvec3 block) {
  ET_CHECK_MSG(kernel != nullptr,
               "MetalCommandRecorder::doDispatchKernel: kernel is null");
  ET_CHECK_MSG(kernel->pipeline() != nil,
               "MetalCommandRecorder::doDispatchKernel: kernel has no pipeline");

  dispatchBackend()->setKernel(kernel);

  const bool needsBarrier = hazards_->needsBarrierForPending();

  ET_LOG(Debug, "MetalCommandRecorder: dispatching grid=(%u,%u,%u), block=(%u,%u,%u) "
               "(hazard=%s; pendingIn=%zu pendingOut=%zu)",
         (uint)grid.x, (uint)grid.y, (uint)grid.z,
         (uint)block.x, (uint)block.y, (uint)block.z,
         needsBarrier ? "barrier" : "skip",
         hazards_->pendingInputs().size(),
         hazards_->pendingOutputs().size());

  dispatchBackend()->dispatchHazardAware(grid, block, needsBarrier);
  hazards_->commitPending(needsBarrier);

  hasPendingWork_ = true;

  ++dispatchCount_;
  if (flushInterval_ > 0 && dispatchCount_ >= flushInterval_) {
    flush();
  }
}

//===----------------------------------------------------------------------===//
// MetalCommandRecorder::Dispatch — RAII scope. Forwards to recorder; run()
// invokes doDispatchKernel with the captured kernel/pso.
//===----------------------------------------------------------------------===//

MetalCommandRecorder::Dispatch& MetalCommandRecorder::Dispatch::setInput(
    uint32_t slot, const void* ptr, size_t bytes) {
  recorder_->setInput(slot, ptr, bytes);
  return *this;
}
MetalCommandRecorder::Dispatch& MetalCommandRecorder::Dispatch::setOutput(
    uint32_t slot, void* ptr, size_t bytes) {
  recorder_->setOutput(slot, ptr, bytes);
  return *this;
}
MetalCommandRecorder::Dispatch& MetalCommandRecorder::Dispatch::setInOut(
    uint32_t slot, void* ptr, size_t bytes) {
  recorder_->setInOut(slot, ptr, bytes);
  return *this;
}
MetalCommandRecorder::Dispatch& MetalCommandRecorder::Dispatch::setBytes(
    uint32_t slot, const void* ptr, size_t bytes) {
  recorder_->setBytes(slot, ptr, bytes);
  return *this;
}
MetalCommandRecorder::Dispatch& MetalCommandRecorder::Dispatch::setInputBuffer(
    uint32_t slot, id<MTLBuffer> buf, size_t off, size_t sz) {
  recorder_->setInputBuffer(slot, buf, off, sz);
  return *this;
}
MetalCommandRecorder::Dispatch& MetalCommandRecorder::Dispatch::setOutputBuffer(
    uint32_t slot, id<MTLBuffer> buf, size_t off, size_t sz) {
  recorder_->setOutputBuffer(slot, buf, off, sz);
  return *this;
}

void MetalCommandRecorder::Dispatch::run(uvec3 grid, uvec3 block) {
  ET_CHECK_MSG(recorder_ != nullptr,
               "MetalCommandRecorder::Dispatch::run: dispatch has been moved-from "
               "or already consumed");
  if (kernel_) {
    recorder_->doDispatchKernel(kernel_, grid, block);
  } else {
    void* key = (__bridge void*)pso_;
    auto& lru = recorder_->psoWrapLru_;
    auto& index = recorder_->psoWrapIndex_;
    MetalKernel* k = nullptr;
    auto idx_it = index.find(key);
    if (idx_it == index.end()) {
      // Miss: build wrapper, push to LRU front, evict oldest if over cap.
      // Wrapper name: prefer the PSO's own label (Metal stores it from
      // pipelineDesc.label, set in MetalKernelCompiler) so traces are
      // debuggable. Fall back to "<jit_pso>" if the PSO has no label.
      const char* wrapperName = "<jit_pso>";
      NSString* psoLabel = [pso_ label];
      if (psoLabel && [psoLabel length] > 0) {
        wrapperName = [psoLabel UTF8String];
      }
      lru.push_front({key, std::make_unique<MetalKernel>(pso_, wrapperName)});
      index[key] = lru.begin();
      k = lru.front().kernel.get();
      while (lru.size() > kPsoWrapCacheCap) {
        index.erase(lru.back().key);
        lru.pop_back();
      }
    } else {
      // Hit: move to LRU front. splice on the same list is O(1) and keeps
      // the iterator (and the unique_ptr) valid.
      lru.splice(lru.begin(), lru, idx_it->second);
      k = idx_it->second->kernel.get();
    }
    recorder_->doDispatchKernel(k, grid, block);
  }
  recorder_ = nullptr;
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch
