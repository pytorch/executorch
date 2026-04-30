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
  // R6.2b: legacy MTL3 dispatch path. Always created — even under MTL4,
  // the legacy backend is used by MPS interop (which requires a legacy CB).
  backend_ = std::make_unique<MetalMTL3Backend>(device_, queue_);

#if ET_METAL4_ENABLE
  // R6.3: lazy MTL4 backend if the runtime opt-in is set.
  if (useMTL4()) {
    if (@available(macOS 26.0, iOS 26.0, *)) {
      // Lazy MTL4 backend if the runtime opt-in is set.
      mtl4Backend_ = std::make_unique<MetalMTL4Backend>(
          device_, allocator_->residency());
      if (mtl4Backend_->isReady() && allocator_->residency()->isEnabled()) {
        mtl4Backend_->addQueueResidency(allocator_->residency()->nativeSet());
      }
      // wire MTL4 compiler through the backend so the
      // kernel compiler can reach mtl4Compiler_ on demand.
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
  // releases the legacy CB / encoder / in-flight CB.
}

//===----------------------------------------------------------------------===//
// Backend routing
//===----------------------------------------------------------------------===//

IComputeBackend* MetalCommandRecorder::dispatchBackend() {
#if ET_METAL4_ENABLE
  if (useMTL4() && mtl4Backend_ && mtl4Backend_->isReady()) {
    return mtl4Backend_.get();
  }
#endif
  return backend_.get();
}

void MetalCommandRecorder::ensureCommandBuffer() {
#if ET_METAL4_ENABLE
  if (useMTL4() && mtl4Backend_ && mtl4Backend_->isReady()) {
    if (@available(macOS 26.0, iOS 26.0, *)) {
      mtl4Backend_->ensureCommandBuffer();
      return;  // MTL4 path active; legacy buffer not needed.
    }
  }
#endif
  // MTL3 path: backend_ is a MetalMTL3Backend which implements
  // ILegacyCommandBufferProvider. Cast at the seam to avoid leaking
  // legacy-CB methods through the universal IComputeBackend interface.
  auto* legacy = dynamic_cast<ILegacyCommandBufferProvider*>(backend_.get());
  ET_CHECK_MSG(legacy != nullptr,
               "MetalCommandRecorder: MTL3 backend does not implement "
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

  // Metal 4: Commit residency set before execution.
  if (allocator_) allocator_->residency()->commit();

#if ET_METAL4_ENABLE
  if (useMTL4() && mtl4Backend_) {
    if (@available(macOS 26.0, iOS 26.0, *)) {
      mtl4Backend_->commit();
    }
  }
#endif
  flushCommitLegacy();

  hasPendingWork_ = false;
  // R8.1: encoder boundary — work has been committed and any subsequent
  // dispatch on a fresh encoder doesn't have a hazard with prior dispatches.
  hazards_->reset();
}

void MetalCommandRecorder::flushCommitLegacy() {
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
  // MPS-pending drain stays on MetalStream::wait() — MetalCommandRecorder
  // doesn't know about MpsBridge.
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
    ET_LOG(Debug, "R8.1.dbg setInput slot=%u ptr=%p size=%zu mtl=%p offset=%zu",
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
    ET_LOG(Debug, "R8.1.dbg setOutput slot=%u ptr=%p size=%zu mtl=%p offset=%zu",
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
    ET_LOG(Debug, "R8.1.dbg setInOut slot=%u ptr=%p size=%zu mtl=%p offset=%zu",
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
}

void MetalCommandRecorder::setOutputBuffer(
    uint32_t slot, id<MTLBuffer> buf, size_t offset, size_t size) {
  dispatchBackend()->setBuffer(buf, offset, slot);
  hazards_->trackOutput(buf, offset, offset + size);
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
               "(R8.1 hazard=%s; pendingIn=%zu pendingOut=%zu)",
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

void MetalCommandRecorder::dispatch(
    MetalKernel* kernel, uvec3 grid, uvec3 block) {
  doDispatchKernel(kernel, grid, block);
}

void MetalCommandRecorder::dispatch(
    id<MTLComputePipelineState> pso, uvec3 grid, uvec3 block) {
  ET_CHECK_MSG(pso != nil,
               "MetalCommandRecorder::dispatch(pso, ...): pso is nil");
  void* key = (__bridge void*)pso;
  auto it = psoWrapCache_.find(key);
  MetalKernel* kernel = nullptr;
  if (it == psoWrapCache_.end()) {
    auto wrapper = std::make_unique<MetalKernel>(pso, "<jit_pso>");
    kernel = wrapper.get();
    psoWrapCache_[key] = std::move(wrapper);
  } else {
    kernel = it->second.get();
  }
  doDispatchKernel(kernel, grid, block);
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
    auto it = recorder_->psoWrapCache_.find(key);
    MetalKernel* k = nullptr;
    if (it == recorder_->psoWrapCache_.end()) {
      auto wrapper = std::make_unique<MetalKernel>(pso_, "<jit_pso>");
      k = wrapper.get();
      recorder_->psoWrapCache_[key] = std::move(wrapper);
    } else {
      k = it->second.get();
    }
    recorder_->doDispatchKernel(k, grid, block);
  }
  recorder_ = nullptr;
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch
