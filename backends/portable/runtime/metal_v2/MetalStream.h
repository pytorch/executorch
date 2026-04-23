/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/portable/runtime/metal_v2/MetalTypes.h>

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

#include <algorithm>
#include <functional>
#include <list>
#include <map>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

//===----------------------------------------------------------------------===//
// ET_METAL4_ENABLE
//
// Compile-time opt-in for the Metal 4 dispatch path (MTL4Compiler,
// MTL4ComputeCommandEncoder, MTL4ArgumentTable, MTL4CommandQueue, ...).
//
// Default: 0 (legacy path). Define ET_METAL4_ENABLE=1 in the build to compile
// the MTL4 code paths in. Even when compiled in, MTL4 is only used at runtime
// when the OS supports it (macOS 26+ / iOS 26+) -- see useMTL4() below.
//
// This is independent of ET_METAL4_AVAILABLE (defined in MetalStream.mm),
// which guards macOS-15-era APIs (MTLResidencySet, MTLBinaryArchive, ...).
//===----------------------------------------------------------------------===//
#ifndef ET_METAL4_ENABLE
#define ET_METAL4_ENABLE 0
#endif

namespace executorch {
namespace backends {
namespace metal_v2 {

/// True iff the Metal 4 dispatch paths are both compiled in AND the runtime
/// OS supports them. Use this as the single check at every MTL4 call site:
///   if (useMTL4()) { ...mtl4 path... } else { ...legacy path... }
inline bool useMTL4() {
#if ET_METAL4_ENABLE
  if (@available(macOS 26.0, iOS 26.0, *)) {
    return true;
  }
#endif
  return false;
}

//===----------------------------------------------------------------------===//
// MetalHeap - Pre-allocated memory pool for fast sub-allocation
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// MetalBufferPool - LRU buffer pool with best-fit matching
//===----------------------------------------------------------------------===//

class MetalBufferPool {
public:
  explicit MetalBufferPool(id<MTLDevice> device, size_t maxBytes = 256 * 1024 * 1024);
  ~MetalBufferPool();

  /// Acquire a buffer of at least `size` bytes
  id<MTLBuffer> acquire(size_t size);

  /// Return buffer to pool
  void release(id<MTLBuffer> buffer);

  /// Clear all cached buffers
  void clear();

  /// Current bytes in pool
  size_t cachedBytes() const { return cachedBytes_; }

  /// Maximum bytes the pool will hold before evicting LRU entries.
  size_t maxBytes() const { return maxBytes_; }

  /// Update the cap. If new cap < current cachedBytes_, evicts LRU until
  /// under cap. Useful when caller knows memory budget at init.
  void setMaxBytes(size_t bytes);

  /// Pre-allocate buffers of these sizes and seed them into the cache so
  /// the first round of acquire() calls hit the cache instead of going to
  /// the device. Useful when the caller has a memory plan from AOTI.
  /// If total prewarmed bytes exceeds maxBytes_, oldest entries get evicted.
  void prewarm(const std::vector<size_t>& sizes);

private:
  void evictOldest();

  id<MTLDevice> device_;
  size_t maxBytes_;
  size_t cachedBytes_ = 0;

  struct PoolEntry {
    id<MTLBuffer> buffer;
    size_t size;
  };

  std::list<PoolEntry> lruList_;  // newest at front
  std::multimap<size_t, std::list<PoolEntry>::iterator> sizeMap_;

  static constexpr size_t kMaxHeadroom = 32768;  // 32KB
};

//===----------------------------------------------------------------------===//
// MetalKernel - Compiled Metal compute pipeline
//===----------------------------------------------------------------------===//

class MetalKernel {
public:
  MetalKernel(id<MTLComputePipelineState> pipeline, const char* name);
  ~MetalKernel();

  const char* name() const { return name_.c_str(); }
  uvec3 maxThreadsPerThreadgroup() const;
  void* nativeHandle() { return (__bridge void*)pipeline_; }

  id<MTLComputePipelineState> pipeline() const { return pipeline_; }

private:
  id<MTLComputePipelineState> pipeline_;
  std::string name_;
};

//===----------------------------------------------------------------------===//
// MetalKernelCompiler
//===----------------------------------------------------------------------===//

class MetalKernelCompiler {
public:
  explicit MetalKernelCompiler(id<MTLDevice> device);
  ~MetalKernelCompiler();

  MetalKernel* compile(
      const char* source,
      const char* functionName);

  MetalKernel* loadFromLibrary(
      const void* metallibData,
      size_t metallibSize,
      const char* functionName);

  //=== Binary Archive Support (Metal 4) ===

  /// Load binary archive from file (fast shader loading)
  bool loadBinaryArchive(const char* path);

  /// Save compiled shaders to binary archive
  bool saveBinaryArchive(const char* path);

  /// Check if binary archive is loaded
  bool hasBinaryArchive() const { return binaryArchive_ != nil; }

private:
  id<MTLDevice> device_;
  id<MTLBinaryArchive> binaryArchive_;
  std::unordered_map<std::string, std::unique_ptr<MetalKernel>> cache_;

#if ET_METAL4_ENABLE
  // Lazily-created MTL4 compiler. Used when useMTL4() is true. Reused across
  // pipeline creations so we don't pay the per-compiler setup more than once.
  id<MTL4Compiler> mtl4Compiler_ API_AVAILABLE(macos(26.0), ios(26.0)) = nil;
#endif
};

//===----------------------------------------------------------------------===//
// DispatchSignature - For replay detection
//===----------------------------------------------------------------------===//

struct DispatchSignature {
  MetalKernel* kernel;
  std::vector<size_t> bufferSizes;
  uvec3 grid;
  uvec3 block;

  bool operator==(const DispatchSignature& other) const {
    return kernel == other.kernel &&
           bufferSizes == other.bufferSizes &&
           grid.x == other.grid.x && grid.y == other.grid.y && grid.z == other.grid.z &&
           block.x == other.block.x && block.y == other.block.y && block.z == other.block.z;
  }
};

//===----------------------------------------------------------------------===//
// MetalStream - Main implementation
//
// Uses Metal 4 APIs where available:
// - MTLResidencySet for GPU-resident buffers (macOS 15+, iOS 18+)
// - MTLIndirectCommandBuffer for command replay
//===----------------------------------------------------------------------===//

class MetalStream {
public:
  MetalStream();
  ~MetalStream();

  //=== Core API ===
  void dispatch(
      MetalKernel* kernel,
      std::initializer_list<Arg> args,
      uvec3 grid,
      uvec3 block);

  void flush();
  void wait();
  void sync() { flush(); wait(); }
  void* alloc(size_t bytes);
  void free(void* ptr);
  // Register a host-allocated pointer with the stream so dispatches can
  // resolve it to an MTLBuffer via bufferForPtr(). On Apple Silicon
  // unified memory this is the cheap path for caller-storage tensors.
  //
  // strict_zero_copy:
  //   - false (default): tries newBufferWithBytesNoCopy first; if Metal
  //     refuses (typically because the pointer isn't page-aligned),
  //     falls back to newBufferWithBytes which COPIES the data once.
  //     The copy is fine for one-shot graph inputs/outputs but breaks
  //     true aliasing (subsequent writes to ptr won't be visible to
  //     the GPU buffer).
  //   - true: returns false instead of falling back to a copy. Use this
  //     when the caller needs a guaranteed zero-copy alias (e.g. router
  //     persistent-alias optimization for cross-runtime intermediates).
  bool registerExternalBuffer(
      void* ptr, size_t bytes, bool strict_zero_copy = false);

  //=== Optional Control ===
  void setFlushInterval(int dispatches);
  int flushInterval() const { return flushInterval_; }

  void setUseICB(bool enable) { useICB_ = enable; }
  bool useICB() const { return useICB_; }

  //=== Accessors ===
  MetalKernelCompiler* compiler() { return compiler_.get(); }

  //=== Static factories ===
  /// Get singleton default stream (NOT thread-safe for concurrent dispatch).
  static MetalStream* getDefault();
  /// Get thread-local stream (thread-safe — each thread gets its own).
  static MetalStream* getThreadLocal();
  /// Create a new independent stream (caller owns lifetime).
  static std::unique_ptr<MetalStream> create();

  id<MTLDevice> device() const { return device_; }

  //=== Per-execute lifecycle hook ===
  // Called by synchronize_metal_stream at the end of every AOTI forward
  // call. Resets per-execute bookkeeping that's tracked across iterations
  // for replay correctness:
  //   - currentDispatchIdx_ → 0  position in the replay sequence
  //   - icbRecordedThisIter_ → 0 count of ops submitted (cold or replay)
  //                             this iteration; used by partial-flush as
  //                             upper bound (so iter-2 partial flush at
  //                             matmul_K only drains ops [0..K), not the
  //                             whole ICB which would re-execute later
  //                             ops with stale iter-1 bindings).
  void endExecute() {
    currentDispatchIdx_ = 0;
    icbRecordedThisIter_ = 0;
  }

  //=== Buffer lookup ===
  // Resolve a host pointer to its registered MTLBuffer. Auto-registers if
  // unknown. Used by ops that need to wrap inputs/outputs as MPS-specific
  // buffer types (e.g. MPSGraphTensorData).
  id<MTLBuffer> bufferForPtr(void* ptr, size_t bytes) {
    if (!ptrToBuffer_.count(ptr)) {
      registerExternalBuffer(ptr, bytes);
    }
    auto it = ptrToBuffer_.find(ptr);
    return it == ptrToBuffer_.end() ? nil : it->second;
  }

  //=== MPSGraph integration ===
  // Encode work that requires a legacy MTLCommandBuffer (currently the only
  // such consumer is MPSGraph). MetalStream owns the entire orchestration:
  //
  //   Under MTL3:
  //     - End any open compute encoder, drain pending ICB into the live cb
  //     - Get/create the live legacy commandBuffer_, wrap as MPSCommandBuffer
  //     - Invoke encode_fn(mpsCB) — caller does [graph encodeToCommandBuffer:]
  //     - Adopt back mpsCB.commandBuffer (may be replaced if MPS internally
  //       called commitAndContinue:)
  //
  //   Under MTL4:
  //     - End any open mtl4 encoder, drain pending ICB into mtl4 cb
  //     - Commit current mtl4 cb to mtl4Queue, schedule queue-level signal
  //       on internal MPS-sync event (so MPS work can wait for prior mtl4
  //       work to complete on GPU)
  //     - Create a fresh legacy MTLCommandBuffer, wrap as MPSCommandBuffer
  //     - Encode cb-level wait on the MPS-sync event
  //     - Invoke encode_fn(mpsCB)
  //     - Encode cb-level signal on the MPS-sync event
  //     - Commit (fire-and-forget on legacy queue) and schedule a
  //       queue-level wait on mtl4Queue so the next mtl4 cb commit gates on
  //       MPS completion. wait() at execute end blocks on the MPS event.
  //
  // No CPU stall under MTL4 — both queues run concurrently, gated by the
  // cross-queue event. Caller's encode_fn just does its MPS encode and
  // returns; all sync is handled internally.
  void encodeWithLegacyCommandBuffer(
      std::function<void(MPSCommandBuffer* mpsCB)> encode_fn);

  //=== Binary Archive ===
  bool loadShaderArchive(const char* path) {
    return compiler_->loadBinaryArchive(path);
  }
  bool saveShaderArchive(const char* path) {
    return compiler_->saveBinaryArchive(path);
  }

  //=== Heap Allocation ===
  /// Enable heap-based allocation (faster, but fixed size)
  /// Call before any alloc() calls
  void enableHeap(size_t heapSizeBytes, bool aliasable = false);

  /// Check if heap is enabled
  bool heapEnabled() const { return useHeap_; }

  //=== Buffer pool tuning ===
  /// Set the LRU buffer-pool capacity (bytes). Default 256 MiB.
  /// If new cap < currently-cached bytes, evicts LRU until under cap.
  /// Useful when caller knows the model's working-set size.
  void setBufferPoolCapacity(size_t bytes) {
    if (bufferPool_) bufferPool_->setMaxBytes(bytes);
  }

  /// Pre-allocate buffers of these sizes and seed them into the pool.
  /// Useful when caller has a memory plan from AOTI / model compiler so
  /// the first iteration's alloc() calls hit the cache instead of cold-
  /// allocating from the device.
  void prewarmBufferPool(const std::vector<size_t>& sizes) {
    if (bufferPool_) bufferPool_->prewarm(sizes);
  }

  //=== Thread Safety ===
  /// Enable mutex protection for shared stream (default: false)
  void setThreadSafe(bool enabled) { threadSafe_ = enabled; }

private:
  // Internal MPS-sync helpers — used only by encodeWithLegacyCommandBuffer.
  // (Were previously public methods named publicEndEncoder, publicQueue,
  // publicFlushPendingICB, publicCommandBuffer, publicAdoptCommandBuffer,
  // publicCommandBufferDone, publicMTL4CommitAndSignal, publicMTL4QueueWait,
  // publicNoteMpsEventValue. Made private as MPSGraphOp now goes through the
  // single high-level encodeWithLegacyCommandBuffer entry point.)

  // Drain any ICB ops recorded since the last partial/full ICB flush into
  // the live cb (without commit). Bounded by icbRecordedThisIter_ for
  // replay correctness — see field comment.
  void flushPendingICB();

  // Get/create the live legacy MTLCommandBuffer. Always creates from queue_
  // (under both MTL3 and MTL4) so callers needing a legacy cb get one.
  id<MTLCommandBuffer> getOrCreateLegacyCommandBuffer();

  // Adopt a (possibly replaced) live legacy cb after an external encoder
  // (e.g. MPSGraph encodeToCommandBuffer:) may have called commitAndContinue.
  void adoptLegacyCommandBuffer(id<MTLCommandBuffer> newCB);

  // Mark the live legacy cb as committed externally; next dispatch opens fresh.
  void releaseLegacyCommandBuffer();

#if ET_METAL4_ENABLE
  // Commit current mtl4 cb (if any work) to mtl4Queue and schedule
  // queue-level signal=value on `event` after committed work completes.
  void mtl4CommitAndSignal(id<MTLEvent> event, uint64_t value);

  // Schedule queue-level wait on mtl4Queue for event=value before the next
  // committed mtl4 cb runs.
  void mtl4QueueWait(id<MTLEvent> event, uint64_t value);

  // Lazy-create the per-stream MPS-sync event used by
  // encodeWithLegacyCommandBuffer to bracket MPS encode with cross-queue
  // wait/signal. One event per MetalStream is sufficient (each stream has
  // its own queues; no cross-stream MPS sync needed).
  id<MTLSharedEvent> getOrCreateMpsSyncEvent();
#endif

  void endEncoderInternal() { endEncoder(); }

  //=========================================================================
  // Internal — implementation details (continued)
  //=========================================================================
private:
  void ensureCommandBuffer();
  void ensureEncoder();
  void endEncoder();

  // Reset the ICB record/replay cache. Called internally on signature
  // mismatch (next dispatch differs from what's recorded). Not exposed
  // publicly because external callers can't know when to invoke it —
  // signature tracking is internal. Future: if/when we expose graph
  // boundaries (e.g. "AOTI just compiled a new model variant, drop old
  // recording"), promote back to public.
  void invalidate();

  void encodeDispatch(MetalKernel* kernel, const std::vector<Arg>& args, uvec3 grid, uvec3 block);
  void updateArgumentBuffer(size_t dispatchIdx, const std::vector<Arg>& args);
  DispatchSignature buildSignature(MetalKernel* kernel, const std::vector<Arg>& args, uvec3 grid, uvec3 block);

  // ICB helpers
  void setupICB();
  void setupArgumentBuffer(size_t numDispatches);
  void encodeIntoICB(MetalKernel* kernel, const std::vector<Arg>& args, uvec3 grid, uvec3 block);
  void executeICB();
  bool supportsGPUAddress() const;

  // Metal 4: Residency helpers
  void setupResidencySet();
  void addToResidencySet(id<MTLBuffer> buffer);
  void commitResidencySet();

#if ET_METAL4_ENABLE
  // Initialize MTL4 queue/allocator/argument-table/scratch/event when
  // useMTL4(). Called from the constructor; nil-safe so legacy code path
  // works when MTL4 init fails.
  void setupMTL4() API_AVAILABLE(macos(26.0), ios(26.0));
  // Release all MTL4 objects. Called from destructor.
  void teardownMTL4() API_AVAILABLE(macos(26.0), ios(26.0));
#endif

  // Device flush interval based on architecture
  int getDefaultFlushInterval() const;

private:
  id<MTLDevice> device_;
  id<MTLCommandQueue> queue_;
  id<MTLCommandBuffer> commandBuffer_;

  // In-flight (committed-but-not-drained) command buffer.
  id<MTLCommandBuffer> inFlightCommandBuffer_ = nil;
  id<MTLComputeCommandEncoder> encoder_;

#if ET_METAL4_ENABLE
  // ----- Metal 4 dispatch members -----
  // All gated by ET_METAL4_ENABLE at compile time and useMTL4() at runtime.
  // Created by setupMTL4() during construction. nil if useMTL4() is false.
  id<MTL4CommandQueue> mtl4Queue_ API_AVAILABLE(macos(26.0), ios(26.0)) = nil;
  id<MTL4CommandAllocator> mtl4Allocator_ API_AVAILABLE(macos(26.0), ios(26.0)) = nil;
  id<MTL4CommandBuffer> mtl4CommandBuffer_ API_AVAILABLE(macos(26.0), ios(26.0)) = nil;
  id<MTL4CommandBuffer> mtl4InFlightCommandBuffer_ API_AVAILABLE(macos(26.0), ios(26.0)) = nil;
  id<MTL4ComputeCommandEncoder> mtl4Encoder_ API_AVAILABLE(macos(26.0), ios(26.0)) = nil;
  // Single argument table reused across dispatches in the direct path.
  // Bindings are overwritten per dispatch, then setArgumentTable: + dispatch
  // captures the snapshot.
  id<MTL4ArgumentTable> mtl4ArgTable_ API_AVAILABLE(macos(26.0), ios(26.0)) = nil;
  // Bump scratch buffer for inline scalars (replaces setBytes:atIndex:).
  // 1 MB ring; reset on flush().
  id<MTLBuffer> mtl4ScalarScratch_ = nil;
  size_t mtl4ScalarScratchOffset_ = 0;
  // Completion signal for wait(). Incremented on each commit; wait() blocks
  // until this value is reached.
  id<MTLSharedEvent> mtl4CompletionEvent_ API_AVAILABLE(macos(26.0), ios(26.0)) = nil;
  uint64_t mtl4CompletionValue_ = 0;
  // Per-stream MPS-sync event (lazy-created via getOrCreateMpsSyncEvent()
  // when first MPS op runs). Used by encodeWithLegacyCommandBuffer to
  // bracket MPS encode with cross-queue wait/signal between mtl4Queue and
  // legacy queue_. Single event suffices: same stream, monotonic counter.
  id<MTLSharedEvent> mpsSyncEvent_ API_AVAILABLE(macos(26.0), ios(26.0)) = nil;
  uint64_t mpsSyncCounter_ = 1;
  // Tracks outstanding MPS work committed to the legacy queue_ during this
  // execute. Set by publicNoteMpsEventValue() (called from MPSGraphOp under
  // MTL4 after committing each mps cb); wait() blocks on this value to
  // ensure MPS work completes before the CPU reads outputs.
  id<MTLSharedEvent> pendingMpsEvent_ = nil;
  uint64_t pendingMpsEventValue_ = 0;
  // Fence pool for cross-encoder dependencies (ICB segment chain).
  // Created lazily; small ring of MTLFence objects.
  std::vector<id<MTLFence>> mtl4Fences_;
#endif


  // Buffer management
  std::unique_ptr<MetalBufferPool> bufferPool_;
  std::unique_ptr<MetalHeap> heap_;           // Fast allocation for model buffers
  bool useHeap_ = false;
  std::unordered_map<void*, id<MTLBuffer>> ptrToBuffer_;
  std::unordered_set<void*> externalBuffers_;  // Track which buffers are external (not allocated by us)

  // Kernel compiler
  std::unique_ptr<MetalKernelCompiler> compiler_;

  // ICB for replay
  id<MTLIndirectCommandBuffer> icb_;
  id<MTLBuffer> argumentBuffer_;          // Holds GPU addresses for all dispatches
  size_t argumentBufferSize_ = 0;
  size_t argumentBufferOffset_ = 0;
  bool icbValid_ = false;
  size_t icbDispatchCount_ = 0;
  // For ICB+MPS coexistence: tracks how many ICB commands have already been
  // executeCommandsInBuffer:'d into the live cmd buffer via partial flushes
  // (publicFlushPendingICB called from MPSGraphOp before its encode). flush()
  // only runs commands at indices [icbExecutedCount_, icbDispatchCount_) so
  // we don't double-execute. Reset to 0 at end of flush() and on invalidate.
  size_t icbExecutedCount_ = 0;
  size_t maxICBCommands_ = 1024;          // Max dispatches per ICB

  // Per-dispatch argument layout
  static constexpr size_t kMaxBuffersPerDispatch = 8;
  static constexpr size_t kMaxScalarsPerDispatch = 8;
  struct DispatchArgLayout {
    size_t offset;        // Offset in argumentBuffer_
    size_t numBuffers;
    size_t numScalars;
  };
  std::vector<DispatchArgLayout> argLayouts_;
  // Per-ICB-command pipeline state, parallel to icbDispatchCount_. Needed
  // for MTL4 ICB execute path because the encoder must have a pipeline
  // state set BEFORE setArgumentTable: takes effect for the upcoming
  // dispatch (MTL4 binds-via-table model). The pipeline is also recorded
  // inside the ICB command itself (setComputePipelineState during encode),
  // but the encoder needs a copy too.
  std::vector<id<MTLComputePipelineState>> dispatchPipelines_;

  // Dependency tracking for ICB barriers
  std::unordered_set<void*> writtenBuffers_;  // Buffers written by previous ops
  std::vector<size_t> barrierIndices_;        // Dispatch indices where barriers needed

  // Signature tracking
  std::vector<DispatchSignature> signatures_;
  size_t currentDispatchIdx_ = 0;
  bool isReplaying_ = false;
  // Number of dispatches submitted (cold or replay) in the CURRENT iteration.
  // Reset to 0 by publicEndExecute(). Used by publicFlushPendingICB() as the
  // upper bound when partial-draining ICB at MPS dispatch boundaries — so on
  // replay iterations, we don't re-execute ICB ops that haven't been replayed
  // yet (which would re-run them with stale prior-iter bindings).
  size_t icbRecordedThisIter_ = 0;

  // Set by dispatch() (both encode and replay paths), cleared in flush()
  // after the command buffer has been committed. Gates flush()'s submission
  // step so a flush() called twice in a row with no intervening dispatch is
  // a no-op. The direct (non-ICB) path was implicitly idempotent because
  // commit set commandBuffer_ = nil; ICB needed an explicit flag because
  // icbDispatchCount_ and icb_ stay set across flush() for replay.
  bool hasPendingWork_ = false;

  // Auto-batching
  int flushInterval_ = 40;
  bool useICB_ = false;  // ICB disabled by default - enable with METAL_USE_ICB=1
  int dispatchCount_ = 0;

  // Thread safety (optional - use setThreadSafe(true) to enable)
  bool threadSafe_ = false;
  std::mutex mutex_;

  // Metal 4: ResidencySet for GPU-resident memory
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
  id<MTLResidencySet> residencySet_ API_AVAILABLE(macos(15.0), ios(18.0));
#endif
  bool useResidencySet_ = false;
};

} // namespace metal_v2
} // namespace backends
} // namespace executorch
