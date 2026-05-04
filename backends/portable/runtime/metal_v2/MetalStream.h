/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/portable/runtime/metal_v2/HazardTracker.h>
#include <executorch/backends/portable/runtime/metal_v2/MetalAllocator.h>
#include <executorch/backends/portable/runtime/metal_v2/MetalCommandRecorder.h>
#include <executorch/backends/portable/runtime/metal_v2/MetalKernel.h>
#include <executorch/backends/portable/runtime/metal_v2/MetalKernelCompiler.h>
#include <executorch/backends/portable/runtime/metal_v2/MetalTypes.h>
#include <executorch/backends/portable/runtime/metal_v2/MpsInterop.h>  // for ET_METAL_USE_MPSGRAPH

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

// Metal 4 SDK availability check. Used to gate MTLResidencySet and other
// Metal-15+ APIs that exist as a runtime feature even when ET_METAL4_ENABLE
// (the build flag for the MTL4 dispatch path) is off.
#if (defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000) || \
    (defined(__IPHONE_OS_VERSION_MAX_ALLOWED) && __IPHONE_OS_VERSION_MAX_ALLOWED >= 180000)
#define ET_METAL4_AVAILABLE 1
#else
#define ET_METAL4_AVAILABLE 0
#endif

// Forward decl: lambda signature for encodeWithLegacyCommandBuffer takes
// MPSCommandBuffer*. Actual MPSCommandBuffer.h import lives in MetalStream.mm
// to keep this header lightweight.
@class MPSCommandBuffer;

#include <cstdlib>  // for getenv
#include <cstring>  // for strcmp
#include <functional>
#include <memory>
#include <vector>

//===----------------------------------------------------------------------===//
// ET_METAL4_ENABLE
// Compile-time opt-in for the Metal 4 dispatch path (MTL4Compiler,
// MTL4ComputeCommandEncoder, MTL4ArgumentTable, MTL4CommandQueue, ...).
//===----------------------------------------------------------------------===//
#ifndef ET_METAL4_ENABLE
#define ET_METAL4_ENABLE 0
#endif

namespace executorch {
namespace backends {
namespace metal_v2 {

#if ET_METAL_USE_MPSGRAPH
class MpsInterop;
#endif

/// True iff the Metal 4 dispatch paths are both compiled in AND the runtime
/// OS supports them AND the runtime opt-in is enabled. See full doc in the
/// pre-2026 git history; gated by ET_METAL4_ENABLE + macOS 26+ + the
/// METAL_USE_MTL4 env var (default ON when both prior gates pass).
inline bool useMTL4() {
#if ET_METAL4_ENABLE
  if (@available(macOS 26.0, iOS 26.0, *)) {
    static const bool enabled = []() {
      const char* env = getenv("METAL_USE_MTL4");
      if (env) {
        if (strcmp(env, "0") == 0 || strcmp(env, "false") == 0 ||
            strcmp(env, "FALSE") == 0 || strcmp(env, "off") == 0 ||
            strcmp(env, "OFF") == 0) {
          return false;
        }
      }
      return true;
    }();
    return enabled;
  }
#endif
  return false;
}

//===----------------------------------------------------------------------===//
// MetalStream — thin facade composing the per-thread Metal subsystems.
// Owns (in this construction order):
//   1. device_, queue_                 (immutable Metal handles)
//   2. compiler_  : MetalKernelCompiler
//   3. hazards_   : HazardTracker      (borrowed by allocator_ and recorder_)
//   4. allocator_ : MetalAllocator     (borrows hazards_)
//   5. recorder_  : MetalCommandRecorder (borrows allocator_, compiler_, hazards_)
//   6. mpsBridge_ : MpsBridge
// Stream's own public surface is now mostly delegating shims into the
// peer subsystems for back-compat. New code should prefer:
//   stream->allocator().alloc(N)
//   stream->recorder().beginDispatch(pso)...
//   stream->compiler() / stream->mps() (TODO step 3)
// over the free-form stream-level methods.
// Pre-2026 history: Stream owned encoder + dispatch + alloc + hazards +
// MPS bridge + binary archive in a single ~900-line god class. Three
// extraction passes (Allocator, Recorder, Mps) split it into focused
// peers. The free-form methods on Stream are kept as one-liner shims so
// no op or AOTI shim needs to migrate at once.
//===----------------------------------------------------------------------===//

class MetalStream {
 public:
  MetalStream();
  ~MetalStream();

  MetalStream(const MetalStream&) = delete;
  MetalStream& operator=(const MetalStream&) = delete;

  //===--------------------------------------------------------------------===//
  // Static factories
  //===--------------------------------------------------------------------===//

  /// Get the thread-local MetalStream — the safe default.
  static MetalStream* get();

  /// Create a new independent stream (caller owns lifetime).
  static std::unique_ptr<MetalStream> create();

  //===--------------------------------------------------------------------===//
  // Subsystem accessors (preferred for new code)
  //===--------------------------------------------------------------------===//

  MetalAllocator&        allocator() { return *allocator_; }
  const MetalAllocator&  allocator() const { return *allocator_; }
  MetalCommandRecorder&  recorder()  { return *recorder_; }
  const MetalCommandRecorder& recorder() const { return *recorder_; }
  MetalKernelCompiler*   compiler()  { return compiler_.get(); }
#if ET_METAL_USE_MPSGRAPH
  MpsInterop&            mps()       { return *mpsInterop_; }
  const MpsInterop&      mps() const { return *mpsInterop_; }
#endif

  id<MTLDevice>          device() const { return device_; }

  //===--------------------------------------------------------------------===//
  // Back-compat shims — slot binding & dispatch (delegate to recorder_)
  //===--------------------------------------------------------------------===//

  void setInput(uint32_t slot, const void* ptr, size_t size) {
    recorder_->setInput(slot, ptr, size);
  }
  void setOutput(uint32_t slot, void* ptr, size_t size) {
    recorder_->setOutput(slot, ptr, size);
  }
  void setInOut(uint32_t slot, void* ptr, size_t size) {
    recorder_->setInOut(slot, ptr, size);
  }
  void setBytes(uint32_t slot, const void* ptr, size_t size) {
    recorder_->setBytes(slot, ptr, size);
  }
  template <typename T>
  void setBytes(uint32_t slot, const T& value) {
    recorder_->setBytes<T>(slot, value);
  }
  template <typename T>
  void setVectorBytes(uint32_t slot, const std::vector<T>& v) {
    recorder_->setVectorBytes<T>(slot, v);
  }
  void setInputBuffer(uint32_t slot, id<MTLBuffer> buf, size_t off, size_t sz) {
    recorder_->setInputBuffer(slot, buf, off, sz);
  }
  void setOutputBuffer(uint32_t slot, id<MTLBuffer> buf, size_t off, size_t sz) {
    recorder_->setOutputBuffer(slot, buf, off, sz);
  }

  void dispatch(MetalKernel* kernel, uvec3 grid, uvec3 block) {
    recorder_->dispatch(kernel, grid, block);
  }
  void dispatch(id<MTLComputePipelineState> pso, uvec3 grid, uvec3 block) {
    recorder_->dispatch(pso, grid, block);
  }

  // Dispatch RAII: re-export the Recorder's nested type so existing
  // `stream->beginDispatch(...)` call sites keep working unchanged.
  using Dispatch = MetalCommandRecorder::Dispatch;
  Dispatch beginDispatch(MetalKernel* kernel) { return recorder_->beginDispatch(kernel); }
  Dispatch beginDispatch(id<MTLComputePipelineState> pso) { return recorder_->beginDispatch(pso); }

  void flush() { recorder_->flush(); }
  void wait() { recorder_->wait(); }   // Just delegates; no MPS-event drain (MTL3 only)
  void sync() { flush(); wait(); }

  void setFlushInterval(int dispatches) { recorder_->setFlushInterval(dispatches); }
  int flushInterval() const { return recorder_->flushInterval(); }

  // Re-export for back-compat. Tests + diagnostic dumps still spell this
  // as `MetalStream::BarrierStats`.
  using BarrierStats = HazardTracker::BarrierStats;
  const BarrierStats& barrierStats() const { return recorder_->barrierStats(); }

  //===--------------------------------------------------------------------===//
  // Back-compat shims — memory subsystem (delegate to allocator_)
  //===--------------------------------------------------------------------===//

  void* alloc(size_t bytes) { return allocator_->alloc(bytes); }
  void  free(void* ptr) { allocator_->free(ptr); }
  bool  registerExternalBuffer(void* ptr, size_t bytes,
                               bool strict_zero_copy = false) {
    return allocator_->registerExternalBuffer(ptr, bytes, strict_zero_copy);
  }
  bool  registerSubregion(void* child, void* parent, size_t off, size_t sz) {
    return allocator_->registerSubregion(child, parent, off, sz);
  }
  void  unregisterSubregion(void* child) { allocator_->unregisterSubregion(child); }
  void  notifyExternalWrite(void* ptr, size_t sz) {
    allocator_->notifyExternalWrite(ptr, sz);
  }

  // ⚠️ For Subregion entries returns the *parent* MTLBuffer with no offset
  // info. Callers that need the offset must use bufferAndOffsetForPtr.
  id<MTLBuffer> bufferForPtr(void* ptr, size_t bytes) {
    return allocator_->bufferMtlForPtr(ptr, bytes);
  }
  using BufferBinding = MetalAllocator::BufferBinding;
  BufferBinding bufferAndOffsetForPtr(void* ptr, size_t bytes) {
    return allocator_->bufferForPtr(ptr, bytes);
  }

  void enableHeap(size_t heapSizeBytes, bool aliasable = false) {
    allocator_->enableHeap(heapSizeBytes, aliasable);
  }
  bool heapEnabled() const { return allocator_->heapEnabled(); }
  void setBufferPoolCapacity(size_t bytes) { allocator_->setPoolCapacity(bytes); }
  void prewarmBufferPool(const std::vector<size_t>& sizes) { allocator_->prewarm(sizes); }

  //===--------------------------------------------------------------------===//
  // Back-compat shims — kernel compiler
  //===--------------------------------------------------------------------===//

  bool loadShaderArchive(const char* path) { return compiler_->loadBinaryArchive(path); }
  bool saveShaderArchive(const char* path) { return compiler_->saveBinaryArchive(path); }

  //===--------------------------------------------------------------------===//
  // MPSGraph integration — back-compat shim. Compiled out when
  // ET_METAL_USE_MPSGRAPH=0. Prefer stream->mps().encodeWithLegacyCommandBuffer
  // for new code; this stream-level method just forwards.
  //===--------------------------------------------------------------------===//

#if ET_METAL_USE_MPSGRAPH
  void encodeWithLegacyCommandBuffer(
      std::function<void(MPSCommandBuffer* mpsCB)> encode_fn);
#endif

  //===--------------------------------------------------------------------===//
  // Public constants — re-exported from MetalCommandRecorder for back-compat.
  // MetalMTL4Backend.mm references MetalStream::kMaxBuffersPerDispatch.
  //===--------------------------------------------------------------------===//

  static constexpr size_t kMaxBuffersPerDispatch = MetalCommandRecorder::kMaxBuffersPerDispatch;
  static constexpr size_t kMaxInlineBytes = MetalCommandRecorder::kMaxInlineBytes;

 private:
  // Device flush interval based on architecture. Used by ctor to seed the
  // Recorder's flushInterval_ before any dispatch lands.
  int getDefaultFlushInterval() const;

  // Owned subsystems (in construction order):
  id<MTLDevice> device_;
  id<MTLCommandQueue> queue_;
  std::unique_ptr<MetalKernelCompiler>  compiler_;
  std::unique_ptr<HazardTracker>        hazards_;
  std::unique_ptr<MetalAllocator>       allocator_;
  std::unique_ptr<MetalCommandRecorder> recorder_;
#if ET_METAL_USE_MPSGRAPH
  std::unique_ptr<MpsInterop>           mpsInterop_;
#endif
};

} // namespace metal_v2
} // namespace backends
} // namespace executorch
