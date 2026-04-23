/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// MetalStream — the runtime stream itself. Helper classes (MetalHeap,
// MetalBufferPool, MetalKernel, MetalKernelCompiler) live in their own .mm
// files alongside their declarations in MetalStream.h.

#import "MetalStream.h"
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <executorch/runtime/platform/log.h>
#include <algorithm>
#include <cstring>

namespace executorch {
namespace backends {
namespace metal_v2 {

// Singleton
static MetalStream* defaultStream_ = nullptr;
static thread_local MetalStream* threadLocalStream_ = nullptr;

MetalStream* MetalStream::getDefault() {
  if (!defaultStream_) {
    defaultStream_ = new MetalStream();
  }
  return defaultStream_;
}

MetalStream* MetalStream::getThreadLocal() {
  if (!threadLocalStream_) {
    threadLocalStream_ = new MetalStream();
  }
  return threadLocalStream_;
}

std::unique_ptr<MetalStream> MetalStream::create() {
  return std::make_unique<MetalStream>();
}

MetalStream::MetalStream() {
  @autoreleasepool {
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) {
      ET_LOG(Error, "MetalStream: failed to create Metal device");
      return;
    }
    [device_ retain];

    queue_ = [device_ newCommandQueue];
    [queue_ retain];

    bufferPool_ = std::make_unique<MetalBufferPool>(device_);
    compiler_ = std::make_unique<MetalKernelCompiler>(device_);

    flushInterval_ = getDefaultFlushInterval();

    // Check env var for ICB mode (disabled by default)
    // ICB mode encodes commands for replay but doesn't handle data dependencies
    const char* icbEnv = getenv("METAL_USE_ICB");
    useICB_ = icbEnv && (strcmp(icbEnv, "1") == 0 || strcmp(icbEnv, "true") == 0);

    // Metal 4: Setup ResidencySet for GPU-resident memory
    setupResidencySet();

#if ET_METAL4_ENABLE
    // Metal 4 dispatch path: queue/allocator/arg-table/scratch/event
    if (useMTL4()) {
      if (@available(macOS 26.0, iOS 26.0, *)) {
        setupMTL4();
      }
    }
#endif

    ET_LOG(Info, "MetalStream: initialized with device '%s', flush interval=%d, ICB=%s",
           [[device_ name] UTF8String], flushInterval_, useICB_ ? "enabled" : "disabled");
  }
}

MetalStream::~MetalStream() {
  @autoreleasepool {
    sync();  // flush() + wait() — drains any pending and in-flight work.

    if (icb_) [icb_ release];
    if (argumentBuffer_) [argumentBuffer_ release];
    if (encoder_) [encoder_ release];
    if (commandBuffer_) [commandBuffer_ release];
    if (inFlightCommandBuffer_) [inFlightCommandBuffer_ release];

#if ET_METAL4_ENABLE
    if (@available(macOS 26.0, iOS 26.0, *)) {
      teardownMTL4();
    }
#endif

#if ET_METAL4_AVAILABLE
    if (@available(macOS 15.0, iOS 18.0, *)) {
      if (residencySet_) [residencySet_ release];
    }
#endif

    // Release all tracked buffers
    for (auto& [ptr, buffer] : ptrToBuffer_) {
      [buffer release];
    }

    [queue_ release];
    [device_ release];

    ET_LOG(Debug, "MetalStream: Destroyed");
  }
}

int MetalStream::getDefaultFlushInterval() const {
  // Determine flush interval based on GPU architecture
  // Architecture string ends with: 'p' = iPhone, 'g' = base, 's' = Max, 'd' = Ultra
  char suffix = 'g';

  if (@available(macOS 13.0, iOS 16.0, *)) {
    id architecture = [device_ architecture];
    if (architecture) {
      NSString* name = [architecture name];
      if (name && [name length] > 0) {
        suffix = [name characterAtIndex:[name length] - 1];
      }
    }
  }

  switch (suffix) {
    case 'p': return 20;  // iPhone - more conservative
    case 'g': return 40;  // Base/Pro
    case 's': return 50;  // Max
    case 'd': return 50;  // Ultra
    default:  return 40;
  }
}

void MetalStream::setupResidencySet() {
#if ET_METAL4_AVAILABLE
  if (@available(macOS 15.0, iOS 18.0, *)) {
    MTLResidencySetDescriptor* desc = [[MTLResidencySetDescriptor alloc] init];
    desc.label = @"GpuStream ResidencySet";
    // Initial capacity - will grow as needed
    desc.initialCapacity = 64;

    NSError* error = nil;
    residencySet_ = [device_ newResidencySetWithDescriptor:desc error:&error];
    [desc release];

    if (residencySet_) {
      useResidencySet_ = true;
      ET_LOG(Info, "MetalStream: Metal 4 ResidencySet enabled");
    } else {
      ET_LOG(Info, "MetalStream: ResidencySet not available: %s",
             error ? [[error localizedDescription] UTF8String] : "unknown");
    }
  }
#endif
}

void MetalStream::addToResidencySet(id<MTLBuffer> buffer) {
#if ET_METAL4_AVAILABLE
  if (@available(macOS 15.0, iOS 18.0, *)) {
    if (useResidencySet_ && residencySet_) {
      [residencySet_ addAllocation:buffer];
    }
  }
#endif
}

void MetalStream::commitResidencySet() {
#if ET_METAL4_AVAILABLE
  if (@available(macOS 15.0, iOS 18.0, *)) {
    if (useResidencySet_ && residencySet_) {
      // BOTH calls are required:
      //   commit  — applies pending addAllocation:/removeAllocation: changes
      //             to the set itself. Without this, allocations stay in a
      //             "pending" state and the set acts as if empty. For MTL4
      //             this manifests as kernels reading from never-resident
      //             memory → silent zeros / input-passthrough output.
      //   requestResidency — asks the OS to physically page-in the now-
      //             committed allocations. Best-effort; safe to call again.
      [residencySet_ commit];
      [residencySet_ requestResidency];
      ET_LOG(Debug, "MetalStream: Committed ResidencySet (size=%llu bytes)",
             (unsigned long long)[residencySet_ allocatedSize]);
    }
  }
#endif
}

void MetalStream::enableHeap(size_t heapSizeBytes, bool aliasable) {
  if (heap_) {
    ET_LOG(Info, "MetalStream: Heap already enabled");
    return;
  }

  heap_ = std::make_unique<MetalHeap>(device_, heapSizeBytes, aliasable);
  if (heap_ && heap_->totalSize() > 0) {
    useHeap_ = true;
    ET_LOG(Info, "MetalStream: Heap enabled (%zu MB)", heapSizeBytes / (1024*1024));
  }
}

bool MetalStream::supportsGPUAddress() const {
#if ET_METAL4_AVAILABLE
  if (@available(macOS 15.0, iOS 18.0, *)) {
    return [device_ supportsFamily:MTLGPUFamilyMetal3];
  }
#endif
  return false;
}

#if ET_METAL4_ENABLE
void MetalStream::setupMTL4() {
  if (@available(macOS 26.0, iOS 26.0, *)) {
    @autoreleasepool {
      NSError* err = nil;

      // Command queue
      MTL4CommandQueueDescriptor* qDesc = [[MTL4CommandQueueDescriptor alloc] init];
      mtl4Queue_ = [device_ newMTL4CommandQueueWithDescriptor:qDesc error:&err];
      [qDesc release];
      if (!mtl4Queue_ || err) {
        ET_LOG(Error, "MetalStream: MTL4CommandQueue creation failed: %s",
               err ? [[err localizedDescription] UTF8String] : "unknown");
        mtl4Queue_ = nil;
        return;
      }
      [mtl4Queue_ retain];

      // Add residency set to MTL4 queue too (so MTL4 cmd buffers see resident memory)
      if (residencySet_) {
        [mtl4Queue_ addResidencySet:residencySet_];
      }

      // Command allocator
      err = nil;
      MTL4CommandAllocatorDescriptor* aDesc = [[MTL4CommandAllocatorDescriptor alloc] init];
      mtl4Allocator_ = [device_ newCommandAllocatorWithDescriptor:aDesc error:&err];
      [aDesc release];
      if (!mtl4Allocator_ || err) {
        ET_LOG(Error, "MetalStream: MTL4CommandAllocator creation failed: %s",
               err ? [[err localizedDescription] UTF8String] : "unknown");
        [mtl4Queue_ release]; mtl4Queue_ = nil;
        mtl4Allocator_ = nil;
        return;
      }
      [mtl4Allocator_ retain];

      // Argument table (single, reused per dispatch)
      MTL4ArgumentTableDescriptor* atDesc = [[MTL4ArgumentTableDescriptor alloc] init];
      atDesc.maxBufferBindCount = kMaxBuffersPerDispatch;
      err = nil;
      mtl4ArgTable_ = [device_ newArgumentTableWithDescriptor:atDesc error:&err];
      [atDesc release];
      if (!mtl4ArgTable_ || err) {
        ET_LOG(Error, "MetalStream: MTL4ArgumentTable creation failed: %s",
               err ? [[err localizedDescription] UTF8String] : "unknown");
        [mtl4Allocator_ release]; mtl4Allocator_ = nil;
        [mtl4Queue_ release]; mtl4Queue_ = nil;
        mtl4ArgTable_ = nil;
        return;
      }
      [mtl4ArgTable_ retain];

      // Inline-scalar bump scratch (1 MB, shared storage)
      constexpr size_t kScratchBytes = 1u << 20;
      mtl4ScalarScratch_ = [device_ newBufferWithLength:kScratchBytes
                                                options:MTLResourceStorageModeShared];
      [mtl4ScalarScratch_ retain];
      addToResidencySet(mtl4ScalarScratch_);
      mtl4ScalarScratchOffset_ = 0;

      // Completion event for wait()
      mtl4CompletionEvent_ = [device_ newSharedEvent];
      [mtl4CompletionEvent_ retain];
      mtl4CompletionValue_ = 0;

      ET_LOG(Info, "MetalStream: MTL4 dispatch path initialized "
                   "(queue+allocator+arg-table+scratch+event)");
    }
  }
}

void MetalStream::teardownMTL4() {
  if (@available(macOS 26.0, iOS 26.0, *)) {
    if (mtl4Encoder_) { [mtl4Encoder_ release]; mtl4Encoder_ = nil; }
    if (mtl4CommandBuffer_) { [mtl4CommandBuffer_ release]; mtl4CommandBuffer_ = nil; }
    if (mtl4InFlightCommandBuffer_) { [mtl4InFlightCommandBuffer_ release]; mtl4InFlightCommandBuffer_ = nil; }
    if (mtl4ArgTable_) { [mtl4ArgTable_ release]; mtl4ArgTable_ = nil; }
    if (mtl4Allocator_) { [mtl4Allocator_ release]; mtl4Allocator_ = nil; }
    if (mtl4Queue_) { [mtl4Queue_ release]; mtl4Queue_ = nil; }
    if (mtl4ScalarScratch_) { [mtl4ScalarScratch_ release]; mtl4ScalarScratch_ = nil; }
    if (mtl4CompletionEvent_) { [mtl4CompletionEvent_ release]; mtl4CompletionEvent_ = nil; }
    if (mpsSyncEvent_) { [mpsSyncEvent_ release]; mpsSyncEvent_ = nil; }
    for (id<MTLFence> f : mtl4Fences_) [f release];
    mtl4Fences_.clear();
  }
}
#endif

void MetalStream::setupICB() {
  if (icb_) return;

  @autoreleasepool {
    MTLIndirectCommandBufferDescriptor* desc = [[MTLIndirectCommandBufferDescriptor alloc] init];
    desc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
    // inheritBuffers depends on dispatch model:
    //  - MTL3 (legacy): NO. ICB commands carry their own setKernelBuffer:
    //    bindings, executed by MTLComputeCommandEncoder which has no
    //    argument-table model.
    //  - MTL4: YES. ICB commands inherit from the executing
    //    MTL4ComputeCommandEncoder's argument table. Metal validation
    //    rejects setKernelBuffer: on a inherit-mode ICB, so the encode
    //    path also conditionally skips setKernelBuffer: under MTL4.
    desc.inheritBuffers = useMTL4() ? YES : NO;
    desc.inheritPipelineState = NO;
    desc.maxKernelBufferBindCount = kMaxBuffersPerDispatch;

    icb_ = [device_ newIndirectCommandBufferWithDescriptor:desc
                                           maxCommandCount:maxICBCommands_
                                                   options:MTLResourceStorageModeShared];
    [desc release];

    if (icb_) {
      [icb_ retain];
      ET_LOG(Info, "MetalStream: Created ICB with max %zu commands", maxICBCommands_);
      // Add the ICB itself to the residency set so it's GPU-resident under
      // MTL4 (which has no automatic residency tracking for resources
      // referenced via executeCommandsInBuffer:). MTLIndirectCommandBuffer
      // conforms to MTLAllocation so addAllocation: accepts it directly.
#if ET_METAL4_AVAILABLE
      if (@available(macOS 15.0, iOS 18.0, *)) {
        if (useResidencySet_ && residencySet_) {
          [residencySet_ addAllocation:icb_];
          ET_LOG(Info, "MetalStream: Added ICB to residency set");
        }
      }
#endif
    }
  }
}

void MetalStream::setupArgumentBuffer(size_t numDispatches) {
  // Calculate required size
  // Each dispatch needs: 8 GPU addresses (64 bytes) + 8 scalars (64 bytes) = 128 bytes
  size_t bytesPerDispatch = (kMaxBuffersPerDispatch * sizeof(uint64_t)) +
                            (kMaxScalarsPerDispatch * sizeof(int64_t));
  size_t requiredSize = numDispatches * bytesPerDispatch;

  if (!argumentBuffer_ || argumentBufferSize_ < requiredSize) {
    if (argumentBuffer_) {
      [argumentBuffer_ release];
    }

    argumentBufferSize_ = std::max(requiredSize, (size_t)(1024 * bytesPerDispatch));  // Pre-allocate for 1024
    argumentBuffer_ = [device_ newBufferWithLength:argumentBufferSize_
                                           options:MTLResourceStorageModeShared];
    [argumentBuffer_ retain];

    addToResidencySet(argumentBuffer_);
    ET_LOG(Info, "MetalStream: Created argument buffer (%zu bytes)", argumentBufferSize_);
  }
}

void* MetalStream::alloc(size_t bytes) {
  id<MTLBuffer> buffer = nil;

  // Try heap first (faster: ~100ns vs ~10µs)
  if (useHeap_ && heap_) {
    buffer = heap_->allocBuffer(bytes);
  }

  // Fallback to buffer pool
  if (!buffer) {
    buffer = bufferPool_->acquire(bytes);
  }

  if (!buffer) {
    ET_LOG(Error, "MetalStream::alloc: failed to allocate %zu bytes", bytes);
    return nullptr;
  }

  void* ptr = [buffer contents];
  ptrToBuffer_[ptr] = buffer;
  [buffer retain];  // Keep alive while in ptrToBuffer_

  // Metal 4: Add to residency set for GPU-resident memory
  // This ensures the buffer stays resident on GPU, avoiding page faults
  addToResidencySet(buffer);

  ET_LOG(Debug, "MetalStream::alloc: allocated %zu bytes at %p (heap=%d)",
         bytes, ptr, useHeap_ && heap_);
  return ptr;
}

void MetalStream::free(void* ptr) {
  if (!ptr) return;

  auto it = ptrToBuffer_.find(ptr);
  if (it != ptrToBuffer_.end()) {
    id<MTLBuffer> buffer = it->second;
    ptrToBuffer_.erase(it);
    // Only return to pool if we allocated it (not external)
    if (!externalBuffers_.count(ptr)) {
      bufferPool_->release(buffer);
    }
    [buffer release];
  }
}

bool MetalStream::registerExternalBuffer(
    void* ptr, size_t bytes, bool strict_zero_copy) {
  if (!ptr || bytes == 0) return false;

  // Check if already registered
  if (ptrToBuffer_.count(ptr)) {
    return true;
  }

  // Check alignment - page size is 16KB on ARM64
  bool pageAligned = ((uintptr_t)ptr % 16384) == 0;
  ET_LOG(Info, "MetalStream: Registering external buffer %p (%zu bytes, page_aligned=%d, strict_zero_copy=%d)",
         ptr, bytes, pageAligned, strict_zero_copy);

  // For unified memory (Apple Silicon), wrap existing memory with MTLBuffer
  // This allows GPU to access CPU-allocated memory directly
  // Note: Memory must be page-aligned for newBufferWithBytesNoCopy
  id<MTLBuffer> buffer = [device_ newBufferWithBytesNoCopy:ptr
                                                    length:bytes
                                                   options:MTLResourceStorageModeShared
                                               deallocator:nil];

  if (!buffer) {
    if (strict_zero_copy) {
      // Caller requires a true alias; refuse rather than silently copying.
      ET_LOG(Info,
             "MetalStream: zero-copy wrap failed for %p (%zu bytes); strict mode -> refusing fallback",
             ptr, bytes);
      return false;
    }
    // Fallback: copy to a new GPU buffer
    // WARNING: For output buffers, results won't be visible to CPU!
    ET_LOG(Info, "MetalStream: newBufferWithBytesNoCopy failed (not page-aligned?), copying to new buffer");
    buffer = [device_ newBufferWithBytes:ptr
                                  length:bytes
                                 options:MTLResourceStorageModeShared];
    if (!buffer) {
      ET_LOG(Error, "MetalStream: Failed to create buffer for external memory %p", ptr);
      return false;
    }
  } else {
    ET_LOG(Info, "MetalStream: Zero-copy buffer wrap succeeded");
  }

  [buffer retain];
  ptrToBuffer_[ptr] = buffer;
  externalBuffers_.insert(ptr);

  // Add to residency set for GPU access
  addToResidencySet(buffer);

  ET_LOG(Info, "MetalStream: Registered external buffer %p -> MTLBuffer %p", ptr, (__bridge void*)buffer);
  return true;
}

void MetalStream::ensureCommandBuffer() {
#if ET_METAL4_ENABLE
  if (useMTL4()) {
    if (@available(macOS 26.0, iOS 26.0, *)) {
      if (mtl4Queue_ && !mtl4CommandBuffer_) {
        mtl4CommandBuffer_ = [device_ newCommandBuffer];
        [mtl4CommandBuffer_ retain];
        [mtl4CommandBuffer_ beginCommandBufferWithAllocator:mtl4Allocator_];
      }
      if (mtl4Queue_) return;  // MTL4 path active; legacy buffer not needed
    }
  }
#endif
  if (!commandBuffer_) {
    commandBuffer_ = [queue_ commandBuffer];
    [commandBuffer_ retain];
  }
}

void MetalStream::ensureEncoder() {
#if ET_METAL4_ENABLE
  if (useMTL4()) {
    if (@available(macOS 26.0, iOS 26.0, *)) {
      if (mtl4Queue_) {
        if (!mtl4Encoder_) {
          ensureCommandBuffer();
          mtl4Encoder_ = [mtl4CommandBuffer_ computeCommandEncoder];
          [mtl4Encoder_ retain];
          [mtl4Encoder_ setArgumentTable:mtl4ArgTable_];
        }
        return;
      }
    }
  }
#endif
  if (!encoder_) {
    ensureCommandBuffer();
    encoder_ = [commandBuffer_ computeCommandEncoder];
    [encoder_ retain];
  }
}

void MetalStream::endEncoder() {
#if ET_METAL4_ENABLE
  if (@available(macOS 26.0, iOS 26.0, *)) {
    if (mtl4Encoder_) {
      [mtl4Encoder_ endEncoding];
      [mtl4Encoder_ release];
      mtl4Encoder_ = nil;
    }
  }
#endif
  if (encoder_) {
    [encoder_ endEncoding];
    [encoder_ release];
    encoder_ = nil;
  }
}

DispatchSignature MetalStream::buildSignature(
    MetalKernel* kernel,
    const std::vector<Arg>& args,
    uvec3 grid,
    uvec3 block) {
  DispatchSignature sig;
  sig.kernel = kernel;
  sig.grid = grid;
  sig.block = block;

  for (const auto& arg : args) {
    if (arg.type == Arg::BUFFER) {
      sig.bufferSizes.push_back(arg.buffer.size);
    }
  }

  return sig;
}

void MetalStream::dispatch(
    MetalKernel* kernel,
    std::initializer_list<Arg> argsList,
    uvec3 grid,
    uvec3 block) {

  // Optional thread safety
  std::unique_lock<std::mutex> lock(mutex_, std::defer_lock);
  if (threadSafe_) {
    lock.lock();
  }

  std::vector<Arg> args(argsList);
  DispatchSignature sig = buildSignature(kernel, args, grid, block);

  // Check if we can replay with argument buffer update only
  if (icbValid_ && currentDispatchIdx_ < signatures_.size() &&
      sig == signatures_[currentDispatchIdx_]) {
    // Fast path: just update GPU addresses in argument buffer
    isReplaying_ = true;
    updateArgumentBuffer(currentDispatchIdx_, args);
    currentDispatchIdx_++;
    icbRecordedThisIter_++;
    hasPendingWork_ = true;
    return;
  }

  // Slow path: need to encode
  if (icbValid_) {
    // Signature mismatch - invalidate
    invalidate();
  }

  isReplaying_ = false;

  // Setup ICB and argument buffer on first dispatch
  if (!icb_) {
    setupICB();
  }
  setupArgumentBuffer(maxICBCommands_);

  // Encode into ICB with argument buffer binding
  encodeIntoICB(kernel, args, grid, block);
  signatures_.push_back(sig);
  // Whether encodeIntoICB took the real ICB branch or fell through to the
  // direct encodeDispatch, sync() now has work to do — flag it.
  hasPendingWork_ = true;
}

void MetalStream::encodeIntoICB(
    MetalKernel* kernel,
    const std::vector<Arg>& args,
    uvec3 grid,
    uvec3 block) {

  // Use direct encoding unless ICB is explicitly enabled
  // ICB executes commands concurrently which breaks data dependencies
  if (!useICB_) {
    encodeDispatch(kernel, args, grid, block);
    return;
  }

  // ICB path - encode commands for potential replay
  if (!icb_ || icbDispatchCount_ >= maxICBCommands_) {
    ET_LOG(Info, "MetalStream: ICB full or missing, using direct encoding");
    encodeDispatch(kernel, args, grid, block);
    return;
  }

  auto* metalKernel = static_cast<MetalKernel*>(kernel);
  if (!metalKernel || !metalKernel->pipeline()) {
    ET_LOG(Error, "MetalStream: Invalid kernel or pipeline for ICB");
    encodeDispatch(kernel, args, grid, block);
    return;
  }

  // Check if pipeline supports ICB
  if (![metalKernel->pipeline() supportIndirectCommandBuffers]) {
    ET_LOG(Info, "MetalStream: Pipeline doesn't support ICB, using direct encoding");
    encodeDispatch(kernel, args, grid, block);
    return;
  }

  // Dependency tracking: check if any input was written by a previous op
  // If so, we need a barrier before this dispatch
  bool needsBarrier = false;
  void* outputBuffer = nullptr;

  // Find output buffer (last buffer arg is typically output)
  for (const auto& arg : args) {
    if (arg.type == Arg::BUFFER) {
      // Check if this input was a previous output
      if (writtenBuffers_.count(arg.buffer.ptr)) {
        needsBarrier = true;
      }
    }
  }
  // Last buffer is output - track it
  for (auto it = args.rbegin(); it != args.rend(); ++it) {
    if (it->type == Arg::BUFFER) {
      outputBuffer = it->buffer.ptr;
      break;
    }
  }

  if (needsBarrier && icbDispatchCount_ > 0) {
    // Record barrier point - we'll insert memory barrier before this dispatch
    barrierIndices_.push_back(icbDispatchCount_);
    ET_LOG(Info, "MetalStream: Barrier needed before ICB[%zu] (RAW dependency)", icbDispatchCount_);
  }

  // Track this op's output for future dependency checks
  if (outputBuffer) {
    writtenBuffers_.insert(outputBuffer);
  }

  // Get indirect compute command at current index
  id<MTLIndirectComputeCommand> icbCmd = [icb_ indirectComputeCommandAtIndex:icbDispatchCount_];

  // Set pipeline state
  [icbCmd setComputePipelineState:metalKernel->pipeline()];
  // Track for MTL4 flush path (encoder needs to know pipeline before dispatch).
  dispatchPipelines_.push_back(metalKernel->pipeline());

  // Calculate argument buffer offset for this dispatch
  size_t bytesPerDispatch = (kMaxBuffersPerDispatch * sizeof(uint64_t)) +
                            (kMaxScalarsPerDispatch * sizeof(int64_t));
  size_t argOffset = icbDispatchCount_ * bytesPerDispatch;

  // Record layout for replay
  DispatchArgLayout layout;
  layout.offset = argOffset;
  layout.numBuffers = 0;
  layout.numScalars = 0;

  // Fill argument buffer with GPU addresses and scalars
  char* argBase = (char*)[argumentBuffer_ contents] + argOffset;
  uint64_t* gpuAddrs = (uint64_t*)argBase;
  int64_t* scalars = (int64_t*)(argBase + kMaxBuffersPerDispatch * sizeof(uint64_t));

  uint32_t bufferIndex = 0;
  uint32_t scalarIndex = 0;

  for (const auto& arg : args) {
    switch (arg.type) {
      case Arg::BUFFER: {
        // Auto-register external buffers
        if (!ptrToBuffer_.count(arg.buffer.ptr)) {
          registerExternalBuffer(arg.buffer.ptr, arg.buffer.size);
        }

        auto it = ptrToBuffer_.find(arg.buffer.ptr);
        if (it != ptrToBuffer_.end()) {
          // Store GPU address in argument buffer
#if ET_METAL4_AVAILABLE
          if (@available(macOS 15.0, iOS 18.0, *)) {
            gpuAddrs[bufferIndex] = [it->second gpuAddress];
          } else {
            gpuAddrs[bufferIndex] = (uint64_t)(__bridge void*)it->second;
          }
#else
          gpuAddrs[bufferIndex] = (uint64_t)(__bridge void*)it->second;
#endif
          // Bind actual buffer to ICB command (legacy MTL3 path).
          // Skip under MTL4: ICB was created with inheritBuffers=YES, and
          // Metal validation rejects setKernelBuffer: on inherit-mode ICBs.
          // Bindings come from the encoder's arg table at execute time.
          if (!useMTL4()) {
            [icbCmd setKernelBuffer:it->second offset:0 atIndex:bufferIndex];
          }
          bufferIndex++;
          layout.numBuffers++;
        }
        break;
      }
      case Arg::SCALAR_INT: {
        // For ICB, scalars need to be stored in a buffer
        // Create a small buffer for the scalar value
        scalars[scalarIndex] = arg.scalar_int;

        // Calculate offset into argument buffer for this scalar
        size_t scalarOffset = argOffset + kMaxBuffersPerDispatch * sizeof(uint64_t) + scalarIndex * sizeof(int64_t);
        if (!useMTL4()) {
          [icbCmd setKernelBuffer:argumentBuffer_ offset:scalarOffset atIndex:bufferIndex];
        }

        bufferIndex++;
        scalarIndex++;
        layout.numScalars++;
        break;
      }
      case Arg::SCALAR_FLOAT: {
        scalars[scalarIndex] = (int64_t)arg.scalar_float;
        size_t scalarOffset = argOffset + kMaxBuffersPerDispatch * sizeof(uint64_t) + scalarIndex * sizeof(int64_t);
        if (!useMTL4()) {
          [icbCmd setKernelBuffer:argumentBuffer_ offset:scalarOffset atIndex:bufferIndex];
        }
        bufferIndex++;
        scalarIndex++;
        layout.numScalars++;
        break;
      }
      case Arg::TENSOR: {
        // For ICB, treat tensor like a buffer
        if (!ptrToBuffer_.count(arg.tensor.ptr)) {
          registerExternalBuffer(arg.tensor.ptr, arg.tensor.size);
        }
        auto it = ptrToBuffer_.find(arg.tensor.ptr);
        if (it != ptrToBuffer_.end()) {
#if ET_METAL4_AVAILABLE
          if (@available(macOS 15.0, iOS 18.0, *)) {
            gpuAddrs[bufferIndex] = [it->second gpuAddress];
          } else {
            gpuAddrs[bufferIndex] = (uint64_t)(__bridge void*)it->second;
          }
#else
          gpuAddrs[bufferIndex] = (uint64_t)(__bridge void*)it->second;
#endif
          if (!useMTL4()) {
            [icbCmd setKernelBuffer:it->second offset:0 atIndex:bufferIndex];
          }
          bufferIndex++;
          layout.numBuffers++;
        }
        break;
      }
    }
  }

  argLayouts_.push_back(layout);

  // Set threadgroup size and dispatch
  MTLSize mtlBlock = MTLSizeMake(block.x, block.y, block.z);
  MTLSize mtlGrid = MTLSizeMake(grid.x, grid.y, grid.z);

  [icbCmd concurrentDispatchThreadgroups:mtlGrid threadsPerThreadgroup:mtlBlock];
  icbDispatchCount_++;
  icbRecordedThisIter_++;
  ET_LOG(Info, "MetalStream: Encoded into ICB[%zu]: kernel=%s, grid=(%u,%u,%u)",
         icbDispatchCount_-1, kernel->name(), grid.x, grid.y, grid.z);
}

void MetalStream::encodeDispatch(
    MetalKernel* kernel,
    const std::vector<Arg>& args,
    uvec3 grid,
    uvec3 block) {

  ET_LOG(Info, "MetalStream::encodeDispatch: kernel=%s, args=%zu",
         kernel ? kernel->name() : "null", args.size());

  auto* metalKernel = static_cast<MetalKernel*>(kernel);
  if (!metalKernel || !metalKernel->pipeline()) {
    ET_LOG(Error, "MetalStream::encodeDispatch: invalid kernel/pipeline");
    return;
  }

  ensureEncoder();

#if ET_METAL4_ENABLE
  if (useMTL4()) {
    if (@available(macOS 26.0, iOS 26.0, *)) {
      if (mtl4Queue_ && mtl4Encoder_) {
        [mtl4Encoder_ setComputePipelineState:metalKernel->pipeline()];

        // Per-dispatch arg-table updates. Inline scalars go through the
        // bump scratch buffer; their GPU address is set on the table slot.
        uint32_t bufferIndex = 0;
        char* scratchPtr = (char*)[mtl4ScalarScratch_ contents];
        MTLGPUAddress scratchBase = [mtl4ScalarScratch_ gpuAddress];
        const size_t kScratchCap = (size_t)[mtl4ScalarScratch_ length];
        const size_t kAlign = 16;

        for (const auto& arg : args) {
          switch (arg.type) {
            case Arg::BUFFER: {
              if (!ptrToBuffer_.count(arg.buffer.ptr)) {
                registerExternalBuffer(arg.buffer.ptr, arg.buffer.size);
              }
              auto it = ptrToBuffer_.find(arg.buffer.ptr);
              if (it != ptrToBuffer_.end()) {
                MTLGPUAddress addr = [it->second gpuAddress];
                [mtl4ArgTable_ setAddress:addr atIndex:bufferIndex++];
              } else {
                ET_LOG(Error, "MetalStream(MTL4): no buffer for ptr %p", arg.buffer.ptr);
                bufferIndex++;
              }
              break;
            }
            case Arg::SCALAR_INT: {
              // Align + bump scratch
              size_t off = (mtl4ScalarScratchOffset_ + kAlign - 1) & ~(kAlign - 1);
              if (off + sizeof(int64_t) > kScratchCap) {
                ET_LOG(Error, "MetalStream(MTL4): scalar scratch exhausted (call flush more often)");
                bufferIndex++;
                break;
              }
              memcpy(scratchPtr + off, &arg.scalar_int, sizeof(int64_t));
              mtl4ScalarScratchOffset_ = off + sizeof(int64_t);
              [mtl4ArgTable_ setAddress:(scratchBase + off) atIndex:bufferIndex++];
              break;
            }
            case Arg::SCALAR_FLOAT: {
              float f = static_cast<float>(arg.scalar_float);
              size_t off = (mtl4ScalarScratchOffset_ + kAlign - 1) & ~(kAlign - 1);
              if (off + sizeof(float) > kScratchCap) {
                ET_LOG(Error, "MetalStream(MTL4): scalar scratch exhausted");
                bufferIndex++;
                break;
              }
              memcpy(scratchPtr + off, &f, sizeof(float));
              mtl4ScalarScratchOffset_ = off + sizeof(float);
              [mtl4ArgTable_ setAddress:(scratchBase + off) atIndex:bufferIndex++];
              break;
            }
            case Arg::TENSOR: {
              // For now: bind underlying buffer (same as legacy path)
              if (!ptrToBuffer_.count(arg.tensor.ptr)) {
                registerExternalBuffer(arg.tensor.ptr, arg.tensor.size);
              }
              auto it = ptrToBuffer_.find(arg.tensor.ptr);
              if (it != ptrToBuffer_.end()) {
                MTLGPUAddress addr = [it->second gpuAddress];
                [mtl4ArgTable_ setAddress:addr atIndex:bufferIndex++];
              } else {
                bufferIndex++;
              }
              break;
            }
          }
        }

        MTLSize mtlGrid = MTLSizeMake(grid.x, grid.y, grid.z);
        MTLSize mtlBlock = MTLSizeMake(block.x, block.y, block.z);
        ET_LOG(Info, "MetalStream(MTL4): dispatching grid=(%u,%u,%u), block=(%u,%u,%u)",
               (uint)mtlGrid.width, (uint)mtlGrid.height, (uint)mtlGrid.depth,
               (uint)mtlBlock.width, (uint)mtlBlock.height, (uint)mtlBlock.depth);
        // Re-bind the (now-mutated) argument table just before dispatch.
        // setArgumentTable: snapshots the table state — without re-binding
        // after our setAddress: mutations, the encoder would dispatch with
        // stale (encoder-creation-time) table contents.
        [mtl4Encoder_ setArgumentTable:mtl4ArgTable_];
        [mtl4Encoder_ dispatchThreadgroups:mtlGrid threadsPerThreadgroup:mtlBlock];
        // Insert a memory barrier so the *next* dispatch in this encoder
        // sees this dispatch's writes. MTL4 has no automatic hazard
        // tracking — without an explicit intra-encoder barrier, multiple
        // dispatches in the same encoder may run concurrently, violating
        // RAW dependencies for any model where one op reads another op's
        // output (matmul→matmul, conv→relu, etc.). This is conservative
        // (every dispatch gets a barrier even when independent); a future
        // optimization would track per-arg producer/consumer like the
        // ICB path's barrierIndices_ logic does.
        [mtl4Encoder_ barrierAfterEncoderStages:MTLStageDispatch
                            beforeEncoderStages:MTLStageDispatch
                              visibilityOptions:MTL4VisibilityOptionDevice];
        icbDispatchCount_++;
        icbRecordedThisIter_++;
        return;
      }
    }
  }
#endif

  // ----- Legacy path -----
  [encoder_ setComputePipelineState:metalKernel->pipeline()];

  uint32_t bufferIndex = 0;
  for (const auto& arg : args) {
    switch (arg.type) {
      case Arg::BUFFER: {
        // Auto-register external buffers
        if (!ptrToBuffer_.count(arg.buffer.ptr)) {
          registerExternalBuffer(arg.buffer.ptr, arg.buffer.size);
        }

        auto it = ptrToBuffer_.find(arg.buffer.ptr);
        if (it != ptrToBuffer_.end()) {
          [encoder_ setBuffer:it->second offset:0 atIndex:bufferIndex++];
        } else {
          // Registration failed, use setBytes as last resort (copies data)
          ET_LOG(Info, "MetalStream: Using setBytes for ptr %p (%zu bytes)",
                 arg.buffer.ptr, arg.buffer.size);
          [encoder_ setBytes:arg.buffer.ptr length:arg.buffer.size atIndex:bufferIndex++];
        }
        break;
      }
      case Arg::SCALAR_INT:
        [encoder_ setBytes:&arg.scalar_int length:sizeof(int64_t) atIndex:bufferIndex++];
        break;
      case Arg::SCALAR_FLOAT: {
        float f = static_cast<float>(arg.scalar_float);
        [encoder_ setBytes:&f length:sizeof(float) atIndex:bufferIndex++];
        break;
      }
      case Arg::TENSOR: {
#if __has_include(<Metal/MTLTensor.h>) && defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
        if (@available(macOS 26.0, iOS 26.0, *)) {
          // Metal 4.1: Create MTLTensor from buffer
          // First ensure buffer is registered
          if (!ptrToBuffer_.count(arg.tensor.ptr)) {
            registerExternalBuffer(arg.tensor.ptr, arg.tensor.size);
          }

          auto it = ptrToBuffer_.find(arg.tensor.ptr);
          if (it != ptrToBuffer_.end()) {
            // Create tensor descriptor
            MTLTensorDescriptor* desc = [[MTLTensorDescriptor alloc] init];

            // Set dimensions
            NSInteger dims[8];
            NSInteger strides[8];
            for (int i = 0; i < arg.tensor.rank; i++) {
              dims[i] = arg.tensor.dims[i];
              strides[i] = arg.tensor.strides[i];
            }
            desc.dimensions = [[MTLTensorExtents alloc] initWithRank:arg.tensor.rank values:dims];
            desc.strides = [[MTLTensorExtents alloc] initWithRank:arg.tensor.rank values:strides];
            desc.dataType = (MTLTensorDataType)arg.tensor.dtype;
            desc.usage = MTLTensorUsageCompute;

            NSError* error = nil;
            id<MTLTensor> tensor = [device_ newTensorWithDescriptor:desc error:&error];
            if (tensor) {
              // Tensor wraps the buffer - set it at the buffer index
              // Note: MTLTensor conforms to MTLResource, bind via argument buffer or
              // use the underlying buffer
              [encoder_ setBuffer:it->second offset:0 atIndex:bufferIndex++];
              ET_LOG(Info, "MetalStream: Created tensor [%lld x %lld] at index %u",
                     arg.tensor.dims[0], arg.tensor.dims[1], bufferIndex - 1);
            } else {
              ET_LOG(Error, "MetalStream: Failed to create tensor: %s",
                     error ? [[error localizedDescription] UTF8String] : "unknown");
              // Fallback to buffer
              [encoder_ setBuffer:it->second offset:0 atIndex:bufferIndex++];
            }
          }
        } else
#endif
        {
          // Fallback: treat as buffer for older Metal versions
          if (!ptrToBuffer_.count(arg.tensor.ptr)) {
            registerExternalBuffer(arg.tensor.ptr, arg.tensor.size);
          }
          auto it = ptrToBuffer_.find(arg.tensor.ptr);
          if (it != ptrToBuffer_.end()) {
            [encoder_ setBuffer:it->second offset:0 atIndex:bufferIndex++];
          }
        }
        break;
      }
    }
  }

  MTLSize mtlGrid = MTLSizeMake(grid.x, grid.y, grid.z);
  MTLSize mtlBlock = MTLSizeMake(block.x, block.y, block.z);

  ET_LOG(Info, "MetalStream: dispatching grid=(%u,%u,%u), block=(%u,%u,%u)",
         (uint)mtlGrid.width, (uint)mtlGrid.height, (uint)mtlGrid.depth,
         (uint)mtlBlock.width, (uint)mtlBlock.height, (uint)mtlBlock.depth);

  [encoder_ dispatchThreadgroups:mtlGrid threadsPerThreadgroup:mtlBlock];
  icbDispatchCount_++;
  icbRecordedThisIter_++;
}

void MetalStream::updateArgumentBuffer(size_t dispatchIdx, const std::vector<Arg>& args) {
  if (dispatchIdx >= argLayouts_.size()) return;

  const auto& layout = argLayouts_[dispatchIdx];
  char* argBase = (char*)[argumentBuffer_ contents] + layout.offset;
  uint64_t* gpuAddrs = (uint64_t*)argBase;
  int64_t* scalarSlots =
      (int64_t*)(argBase + kMaxBuffersPerDispatch * sizeof(uint64_t));

  // Fast-path replay. We skip re-encoding the ICB command, but the kernel ABI
  // reads its inputs via [[buffer(N)]] bindings (set with setKernelBuffer
  // during encoding), NOT via dereferencing the GPU addresses we stash in
  // gpuAddrs[]. So when the caller hands us new MTLBuffer objects for the
  // same logical args (typical AOTI pattern: fresh per-execute allocations),
  // we must re-bind them on the existing ICB command — otherwise the kernel
  // reads from the encode-time MTLBuffers, which by now have been recycled
  // by the buffer pool and contain stale or unrelated data.
  //
  // Cost: a handful of setKernelBuffer calls per dispatch (~hundreds of ns
  // each on Apple Silicon) — still well under a single re-encode.
  id<MTLIndirectComputeCommand> icbCmd =
      icb_ ? [icb_ indirectComputeCommandAtIndex:dispatchIdx] : nil;

  size_t bufIdx = 0;     // index into gpuAddrs[] (counts buffer / tensor args)
  size_t scalarIdx = 0;  // index into scalarSlots[] (counts scalar args)
  uint32_t slot = 0;     // slot on the ICB command (counts ALL kinds of args)
  for (const auto& arg : args) {
    switch (arg.type) {
      case Arg::BUFFER: {
        if (bufIdx < layout.numBuffers) {
          auto it = ptrToBuffer_.find(arg.buffer.ptr);
          if (it != ptrToBuffer_.end()) {
#if ET_METAL4_AVAILABLE
            if (@available(macOS 15.0, iOS 18.0, *)) {
              gpuAddrs[bufIdx] = [it->second gpuAddress];
            } else {
              gpuAddrs[bufIdx] = (uint64_t)(__bridge void*)it->second;
            }
#else
            gpuAddrs[bufIdx] = (uint64_t)(__bridge void*)it->second;
#endif
            // Re-bind the actual MTLBuffer on the ICB command. Without this,
            // the kernel reads from the encode-time buffer (which may have
            // been freed and reused by the pool).
            // Skip under MTL4: ICB created with inheritBuffers=YES; bindings
            // come from the encoder's argument table (populated by the MTL4
            // partial-flush / final-flush logic from argLayouts_).
            if (icbCmd && !useMTL4()) {
              [icbCmd setKernelBuffer:it->second offset:0 atIndex:slot];
            }
            bufIdx++;
          }
        }
        slot++;
        break;
      }
      case Arg::TENSOR: {
        if (bufIdx < layout.numBuffers) {
          auto it = ptrToBuffer_.find(arg.tensor.ptr);
          if (it != ptrToBuffer_.end()) {
#if ET_METAL4_AVAILABLE
            if (@available(macOS 15.0, iOS 18.0, *)) {
              gpuAddrs[bufIdx] = [it->second gpuAddress];
            } else {
              gpuAddrs[bufIdx] = (uint64_t)(__bridge void*)it->second;
            }
#else
            gpuAddrs[bufIdx] = (uint64_t)(__bridge void*)it->second;
#endif
            if (icbCmd && !useMTL4()) {
              [icbCmd setKernelBuffer:it->second offset:0 atIndex:slot];
            }
            bufIdx++;
          }
        }
        slot++;
        break;
      }
      case Arg::SCALAR_INT: {
        // Scalar values are written into argumentBuffer_; the binding to
        // argumentBuffer_ itself doesn't change between iterations, but the
        // VALUE the kernel reads might (e.g. dynamic-shape models passing
        // changing M/K/N). Update the value in-place.
        if (scalarIdx < layout.numScalars) {
          scalarSlots[scalarIdx] = arg.scalar_int;
          scalarIdx++;
        }
        slot++;
        break;
      }
      case Arg::SCALAR_FLOAT: {
        if (scalarIdx < layout.numScalars) {
          // encodeIntoICB stored scalar_float as int64_t (cast from double).
          // Match the same cast here so kernels see consistent bit patterns.
          scalarSlots[scalarIdx] = (int64_t)arg.scalar_float;
          scalarIdx++;
        }
        slot++;
        break;
      }
    }
  }
}

// flush() submits any pending work to the GPU command queue. Non-blocking:
// returns as soon as the command buffer is committed; the GPU may still be
// processing when this returns. Idempotent: a no-op if no dispatch() has
// happened since the last flush.
// =============================================================================
// MPS-bridge internal helpers — moved out of MetalStream.h to keep header
// lightweight and reduce inline-induced header-pull-in of Metal types.
// =============================================================================

void MetalStream::flushPendingICB() {
  if (!useICB_ || !icb_) return;
  size_t upper = std::min(icbRecordedThisIter_, icbDispatchCount_);
  if (upper <= icbExecutedCount_) return;
  ensureCommandBuffer();

  std::vector<size_t> ends;
  for (size_t b : barrierIndices_) {
    if (b > icbExecutedCount_ && b < upper) {
      ends.push_back(b);
    }
  }
  ends.push_back(upper);

#if ET_METAL4_ENABLE
  if (useMTL4()) {
    if (@available(macOS 26.0, iOS 26.0, *)) {
      size_t start = icbExecutedCount_;
      for (size_t i = 0; i < ends.size(); ++i) {
        size_t end = ends[i];
        if (end > start) {
          id<MTL4ComputeCommandEncoder> enc =
              [mtl4CommandBuffer_ computeCommandEncoder];
          if (start < dispatchPipelines_.size() && start < argLayouts_.size()) {
            [enc setComputePipelineState:dispatchPipelines_[start]];
            const auto& layout = argLayouts_[start];
            char* argBase = (char*)[argumentBuffer_ contents] + layout.offset;
            uint64_t* gpuAddrs = (uint64_t*)argBase;
            MTLGPUAddress argBufBase = [argumentBuffer_ gpuAddress];
            size_t scalarBase = layout.offset + kMaxBuffersPerDispatch * sizeof(uint64_t);
            for (size_t j = 0; j < layout.numBuffers; ++j) {
              [mtl4ArgTable_ setAddress:gpuAddrs[j] atIndex:j];
            }
            for (size_t j = 0; j < layout.numScalars; ++j) {
              MTLGPUAddress sAddr = argBufBase + scalarBase + j * sizeof(int64_t);
              [mtl4ArgTable_ setAddress:sAddr atIndex:layout.numBuffers + j];
            }
          }
          [enc setArgumentTable:mtl4ArgTable_];
          NSRange range = NSMakeRange(start, end - start);
          [enc executeCommandsInBuffer:icb_ withRange:range];
          [enc endEncoding];
        }
        start = end;
      }
      icbExecutedCount_ = upper;
      hasPendingWork_ = true;
      return;
    }
  }
#endif

  size_t start = icbExecutedCount_;
  for (size_t i = 0; i < ends.size(); ++i) {
    size_t end = ends[i];
    if (end > start) {
      id<MTLComputeCommandEncoder> enc = [commandBuffer_ computeCommandEncoder];
      if (argumentBuffer_) [enc useResource:argumentBuffer_ usage:MTLResourceUsageRead];
      for (auto& [ptr, buffer] : ptrToBuffer_) {
        [enc useResource:buffer usage:MTLResourceUsageRead | MTLResourceUsageWrite];
      }
      NSRange range = NSMakeRange(start, end - start);
      [enc executeCommandsInBuffer:icb_ withRange:range];
      if (i < ends.size() - 1) {
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }
      [enc endEncoding];
    }
    start = end;
  }
  icbExecutedCount_ = upper;
  hasPendingWork_ = true;
}

id<MTLCommandBuffer> MetalStream::getOrCreateLegacyCommandBuffer() {
  if (!commandBuffer_) {
    commandBuffer_ = [queue_ commandBuffer];
    [commandBuffer_ retain];
  }
  hasPendingWork_ = true;
  return commandBuffer_;
}

void MetalStream::adoptLegacyCommandBuffer(id<MTLCommandBuffer> newCB) {
  if (newCB == commandBuffer_) {
    hasPendingWork_ = true;
    return;
  }
  if (commandBuffer_) {
    [commandBuffer_ release];
    commandBuffer_ = nil;
  }
  if (newCB) {
    commandBuffer_ = newCB;
    [commandBuffer_ retain];
  }
  hasPendingWork_ = true;
}

void MetalStream::releaseLegacyCommandBuffer() {
  if (commandBuffer_) {
    [commandBuffer_ release];
    commandBuffer_ = nil;
  }
}

#if ET_METAL4_ENABLE
void MetalStream::mtl4CommitAndSignal(id<MTLEvent> event, uint64_t value) {
  if (!useMTL4()) return;
  if (@available(macOS 26.0, iOS 26.0, *)) {
    flushPendingICB();
    commitResidencySet();
    if (mtl4CommandBuffer_) {
      ++mtl4CompletionValue_;
      [mtl4CommandBuffer_ endCommandBuffer];
      const id<MTL4CommandBuffer> bufs[1] = { mtl4CommandBuffer_ };
      [mtl4Queue_ commit:bufs count:1];
      if (mtl4InFlightCommandBuffer_) [mtl4InFlightCommandBuffer_ release];
      mtl4InFlightCommandBuffer_ = mtl4CommandBuffer_;
      mtl4CommandBuffer_ = nil;
      mtl4ScalarScratchOffset_ = 0;
      [mtl4Queue_ signalEvent:mtl4CompletionEvent_ value:mtl4CompletionValue_];
      [mtl4Queue_ signalEvent:event value:value];
    } else {
      [mtl4Queue_ signalEvent:event value:value];
    }
  }
}

void MetalStream::mtl4QueueWait(id<MTLEvent> event, uint64_t value) {
  if (!useMTL4()) return;
  if (@available(macOS 26.0, iOS 26.0, *)) {
    [mtl4Queue_ waitForEvent:event value:value];
  }
}

id<MTLSharedEvent> MetalStream::getOrCreateMpsSyncEvent() {
  if (!mpsSyncEvent_) {
    mpsSyncEvent_ = [device_ newSharedEvent];
    [mpsSyncEvent_ retain];
  }
  return mpsSyncEvent_;
}
#endif

// =============================================================================
// encodeWithLegacyCommandBuffer — single high-level API for ops that need a
// legacy MTLCommandBuffer (currently only MPSGraph).
//
// Replaces what used to be a 9-method dance MPSGraphOp had to perform.
// All the wait/signal/commit/adopt orchestration lives here now.
// =============================================================================
void MetalStream::encodeWithLegacyCommandBuffer(
    std::function<void(MPSCommandBuffer* mpsCB)> encode_fn) {
  // Both paths first close any open compute encoder and drain pending ICB
  // ops into the live cb (so they execute BEFORE the MPS work).
  endEncoderInternal();
  flushPendingICB();

#if ET_METAL4_ENABLE
  if (useMTL4()) {
    if (@available(macOS 26.0, iOS 26.0, *)) {
      // ── MTL4 path: async cross-queue sync ──
      //
      // mtl4Queue commits prior MTL4 work + signals event=preValue.
      // Fresh legacy cb encodes wait(preValue) → MPS body → signal(postValue),
      // commits fire-and-forget. mtl4Queue then waits postValue before any
      // subsequent mtl4 cb commit. wait() at execute end blocks on postValue.
      id<MTLSharedEvent> event = getOrCreateMpsSyncEvent();
      uint64_t preValue  = mpsSyncCounter_++;
      uint64_t postValue = mpsSyncCounter_++;

      mtl4CommitAndSignal(event, preValue);

      id<MTLCommandBuffer> cb = [queue_ commandBuffer];
      MPSCommandBuffer* mpsCB =
          [MPSCommandBuffer commandBufferWithCommandBuffer:cb];

      [mpsCB.commandBuffer encodeWaitForEvent:event value:preValue];
      encode_fn(mpsCB);
      [mpsCB.commandBuffer encodeSignalEvent:event value:postValue];

      [mpsCB commit];

      mtl4QueueWait(event, postValue);
      // Track outstanding MPS event so wait() at end-of-execute blocks
      // until MPS work signals.
      pendingMpsEvent_ = event;
      pendingMpsEventValue_ = postValue;
      return;
    }
  }
#endif

  // ── MTL3 path: adopt-and-share single legacy cb ──
  //
  // MPSGraph encodes into MetalStream's live legacy cb. Subsequent kernel
  // dispatches encode into the same cb. Single end-of-execute commit.
  // MPSGraph may internally call commitAndContinue:, in which case the
  // original cb gets sealed and mpsCB.commandBuffer points to a fresh one;
  // we adopt that fresh one so subsequent dispatches go to the right place.
  id<MTLCommandBuffer> cb = getOrCreateLegacyCommandBuffer();
  MPSCommandBuffer* mpsCB =
      [MPSCommandBuffer commandBufferWithCommandBuffer:cb];
  encode_fn(mpsCB);
  adoptLegacyCommandBuffer(mpsCB.commandBuffer);
}

void MetalStream::flush() {
  // Optional thread safety
  std::unique_lock<std::mutex> lock(mutex_, std::defer_lock);
  if (threadSafe_) {
    lock.lock();
  }

  if (!hasPendingWork_) {
    // Nothing dispatched since last flush — keep flush() cheap & re-entrant.
    return;
  }

  // End any active direct encoder first.
  endEncoder();

  // Metal 4: Commit residency set before execution.
  commitResidencySet();

  // ICB path: encode the (already-recorded) ICB commands into segmented
  // encoders inside our command buffer. Direct path skips this — its
  // dispatchThreadgroups calls already populated commandBuffer_'s encoder.
  if (icbDispatchCount_ > 0 && icb_ && useICB_) {
    ensureCommandBuffer();

    // Build barrier points list (add end as final point)
    std::vector<size_t> barriers = barrierIndices_;
    barriers.push_back(icbDispatchCount_);  // Final segment ends at dispatch count

    bool icbExecuted = false;

#if ET_METAL4_ENABLE
    if (useMTL4()) {
      if (@available(macOS 26.0, iOS 26.0, *)) {
        if (mtl4CommandBuffer_) {
          // Lazily grow the fence pool. Need (numSegments - 1) fences to
          // chain segment[i] -> segment[i+1].
          size_t neededFences = barriers.size() > 1 ? barriers.size() - 1 : 0;
          while (mtl4Fences_.size() < neededFences) {
            id<MTLFence> f = [device_ newFence];
            mtl4Fences_.push_back(f);
          }

          size_t segmentStart = icbExecutedCount_;  // skip already-drained range
          for (size_t barrierIdx = 0; barrierIdx < barriers.size(); barrierIdx++) {
            size_t segmentEnd = barriers[barrierIdx];
            // Skip barriers that fall inside the already-drained range
            // (set by a prior publicFlushPendingICB() call from MPSGraphOp).
            if (segmentEnd <= segmentStart) continue;
            if (segmentEnd > segmentStart) {
              id<MTL4ComputeCommandEncoder> enc =
                  [mtl4CommandBuffer_ computeCommandEncoder];

              // Hypothesis: under MTL4, kernel [[buffer(N)]] reads come
              // from the ENCODER'S argument table, not from per-command
              // setKernelBuffer: bindings recorded in the legacy ICB.
              // To verify, populate the arg table with the same bindings
              // the ICB command at segmentStart was encoded with. This
              // only works if all dispatches in the segment share the
              // same bindings — which is true here because we put a
              // barrier between every dependent dispatch (1 cmd/segment).
              // Per-segment table state means the ICB command's per-cmd
              // setKernelBuffer is functionally redundant on MTL4 — but
              // necessary on MTL3.
              if (segmentStart < argLayouts_.size()) {
                const auto& layout = argLayouts_[segmentStart];
                char* argBase = (char*)[argumentBuffer_ contents] + layout.offset;
                uint64_t* gpuAddrs = (uint64_t*)argBase;
                MTLGPUAddress argBufBase = [argumentBuffer_ gpuAddress];
                size_t scalarBase = layout.offset + kMaxBuffersPerDispatch * sizeof(uint64_t);
                // Set the encoder's pipeline state to the first dispatch in
                // this segment. MTL4 requires setComputePipelineState BEFORE
                // setArgumentTable: takes effect for the upcoming dispatch.
                if (segmentStart < dispatchPipelines_.size()) {
                  [enc setComputePipelineState:dispatchPipelines_[segmentStart]];
                }
                // Buffers: slots [0, numBuffers) — addresses live in gpuAddrs[].
                for (size_t i = 0; i < layout.numBuffers; ++i) {
                  [mtl4ArgTable_ setAddress:gpuAddrs[i] atIndex:i];
                }
                // Scalars: slots [numBuffers, numBuffers+numScalars) — addresses
                // are inside argumentBuffer_ at scalarBase + i*8.
                for (size_t i = 0; i < layout.numScalars; ++i) {
                  MTLGPUAddress sAddr = argBufBase + scalarBase + i * sizeof(int64_t);
                  [mtl4ArgTable_ setAddress:sAddr atIndex:layout.numBuffers + i];
                }
              }
              [enc setArgumentTable:mtl4ArgTable_];

              // Wait on the fence signaled by the previous segment.
              if (barrierIdx > 0) {
                [enc waitForFence:mtl4Fences_[barrierIdx - 1]
                  beforeEncoderStages:MTLStageDispatch];
              }

              // No useResource: needed -- residency set added to mtl4Queue_
              // covers all our buffers.
              NSRange range = NSMakeRange(segmentStart, segmentEnd - segmentStart);
              [enc executeCommandsInBuffer:icb_ withRange:range];

              // Signal a fence for the next segment to wait on.
              if (barrierIdx < barriers.size() - 1) {
                [enc updateFence:mtl4Fences_[barrierIdx]
                  afterEncoderStages:MTLStageDispatch];
              }

              [enc endEncoding];

              ET_LOG(Info, "MetalStream(MTL4)::flush: Executed ICB segment [%zu-%zu) (%zu commands)",
                     segmentStart, segmentEnd, segmentEnd - segmentStart);
            }
            segmentStart = segmentEnd;
          }
          ET_LOG(Info, "MetalStream(MTL4)::flush: Executed ICB with %zu commands, %zu fence-barriers",
                 icbDispatchCount_, barriers.size() > 1 ? barriers.size() - 1 : 0);
          icbExecutedCount_ = icbDispatchCount_;  // mark drained
          icbValid_ = true;
          icbExecuted = true;
        }
      }
    }
#endif

    if (!icbExecuted) {
      // Legacy ICB execution path.
      // ICB+MPS coexistence: skip commands at indices < icbExecutedCount_;
      // those were already encoded into the cmd buffer by an earlier
      // publicFlushPendingICB() call (triggered by MPSGraphOp before its
      // own encode). Only the tail [icbExecutedCount_, icbDispatchCount_)
      // remains.
      size_t segmentStart = icbExecutedCount_;
      for (size_t barrierIdx = 0; barrierIdx < barriers.size(); barrierIdx++) {
        size_t segmentEnd = barriers[barrierIdx];
        if (segmentEnd <= segmentStart) continue;  // already executed by partial flush

        if (segmentEnd > segmentStart) {
          id<MTLComputeCommandEncoder> enc = [commandBuffer_ computeCommandEncoder];

          if (argumentBuffer_) {
            [enc useResource:argumentBuffer_ usage:MTLResourceUsageRead];
          }
          for (auto& [ptr, buffer] : ptrToBuffer_) {
            [enc useResource:buffer usage:MTLResourceUsageRead | MTLResourceUsageWrite];
          }

          NSRange range = NSMakeRange(segmentStart, segmentEnd - segmentStart);
          [enc executeCommandsInBuffer:icb_ withRange:range];

          if (barrierIdx < barriers.size() - 1) {
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
          }

          [enc endEncoding];

          ET_LOG(Info, "MetalStream::flush: Executed ICB segment [%zu-%zu) (%zu commands)",
                 segmentStart, segmentEnd, segmentEnd - segmentStart);
        }
        segmentStart = segmentEnd;
      }

      ET_LOG(Info, "MetalStream::flush: Executed ICB with %zu commands, %zu barriers (icbExecutedCount=%zu before flush)",
             icbDispatchCount_, barrierIndices_.size(), icbExecutedCount_);

      icbExecutedCount_ = icbDispatchCount_;  // tail now drained
      // Mark ICB as valid for replay across future execute()s.
      icbValid_ = true;
    }
  }

  // For replay across executes: reset icbExecutedCount_ so the next execute
  // (which re-runs partial flushes from MPSGraphOp at the same op points)
  // sees the right "nothing executed yet" baseline. Note icbDispatchCount_
  // also resets between executes via the existing replay-vs-encode logic.
  icbExecutedCount_ = 0;

  // Commit (non-blocking). Stash the buffer so wait() can block on it later.
#if ET_METAL4_ENABLE
  if (useMTL4()) {
    if (@available(macOS 26.0, iOS 26.0, *)) {
      if (mtl4CommandBuffer_) {
        // Drain prior MTL4 in-flight buffer if caller flushed twice without wait().
        if (mtl4InFlightCommandBuffer_) {
          if (mtl4CompletionValue_ > 0) {
            [mtl4CompletionEvent_ waitUntilSignaledValue:mtl4CompletionValue_
                                                 timeoutMS:UINT64_MAX];
          }
          [mtl4InFlightCommandBuffer_ release];
          mtl4InFlightCommandBuffer_ = nil;
        }
        // Order matters: signalEvent enqueues a signal that fires after all
        // work *previously* committed completes. To signal "after our cmd
        // buffer", we must commit FIRST, then signalEvent.
        // (Earlier comment claimed signal-then-commit was verified — it was
        // not; the standalone test used waitUntilCompleted on the cmd buffer
        // directly, which worked for a different reason.)
        ++mtl4CompletionValue_;
        [mtl4CommandBuffer_ endCommandBuffer];
        const id<MTL4CommandBuffer> bufs[1] = { mtl4CommandBuffer_ };
        [mtl4Queue_ commit:bufs count:1];
        [mtl4Queue_ signalEvent:mtl4CompletionEvent_ value:mtl4CompletionValue_];
        mtl4InFlightCommandBuffer_ = mtl4CommandBuffer_;
        mtl4CommandBuffer_ = nil;
        // Reset per-flush state for MTL4 path.
        mtl4ScalarScratchOffset_ = 0;
      }
    }
  }
#endif
  if (commandBuffer_) {
    if (inFlightCommandBuffer_) {
      // Caller flushed twice without an intervening wait(). Drain the older
      // submission first so we don't leak completion ownership of two cbufs.
      [inFlightCommandBuffer_ waitUntilCompleted];
      if ([inFlightCommandBuffer_ status] == MTLCommandBufferStatusError) {
        ET_LOG(Error, "MetalStream: prior in-flight command buffer error: %s",
               [[inFlightCommandBuffer_ error] localizedDescription].UTF8String);
      }
      [inFlightCommandBuffer_ release];
      inFlightCommandBuffer_ = nil;
    }
    [commandBuffer_ commit];
    inFlightCommandBuffer_ = commandBuffer_;  // ownership transfer; will be released in wait()
    commandBuffer_ = nil;
  }

  // Reset per-batch state. Keep icbDispatchCount_, icbValid_, barrierIndices_
  // — they're needed on replay. On signature change, invalidate() clears them.
  // Note: currentDispatchIdx_ and icbRecordedThisIter_ are reset in
  // publicEndExecute() (called at execute boundary), not here. flush() can
  // be called mid-execute (by metal_copy_memory or future paths) and must
  // not clobber per-iteration state.
  isReplaying_ = false;
  hasPendingWork_ = false;
  writtenBuffers_.clear();
}

// wait() blocks until all previously-flushed work has completed. Calls
// flush() implicitly so callers don't have to remember the pair.
// Idempotent: a no-op if nothing in flight and nothing pending.
void MetalStream::wait() {
  // Push out anything still pending — keeps wait() the "drain" primitive
  // most callers expect. flush() takes the mutex; we don't here, since the
  // wait is on a property of the in-flight command buffer (not our mutable
  // state).
  flush();

#if ET_METAL4_ENABLE
  if (useMTL4()) {
    if (@available(macOS 26.0, iOS 26.0, *)) {
      if (mtl4InFlightCommandBuffer_) {
        if (mtl4CompletionValue_ > 0) {
          [mtl4CompletionEvent_ waitUntilSignaledValue:mtl4CompletionValue_
                                              timeoutMS:UINT64_MAX];
        }
        [mtl4InFlightCommandBuffer_ release];
        mtl4InFlightCommandBuffer_ = nil;
      }
    }
  }
#endif

  if (inFlightCommandBuffer_) {
    [inFlightCommandBuffer_ waitUntilCompleted];

    if ([inFlightCommandBuffer_ status] == MTLCommandBufferStatusError) {
      ET_LOG(Error, "MetalStream: command buffer error: %s",
             [[inFlightCommandBuffer_ error] localizedDescription].UTF8String);
    }

    [inFlightCommandBuffer_ release];
    inFlightCommandBuffer_ = nil;
  }

#if ET_METAL4_ENABLE
  // Wait for any outstanding MPS work committed during this execute on the
  // legacy queue_ via MPSGraphOp's async path. We don't hold MTLCommandBuffer
  // handles for those (fire-and-forget commits), so we sync via the shared
  // event MPSGraphOp signals at end-of-MPS-encode.
  if (pendingMpsEvent_ && pendingMpsEventValue_ > 0) {
    [pendingMpsEvent_ waitUntilSignaledValue:pendingMpsEventValue_
                                    timeoutMS:UINT64_MAX];
    pendingMpsEvent_ = nil;
    pendingMpsEventValue_ = 0;
  }
#endif
}

void MetalStream::invalidate() {
  icbValid_ = false;
  signatures_.clear();
  argLayouts_.clear();
  dispatchPipelines_.clear();
  currentDispatchIdx_ = 0;
  icbDispatchCount_ = 0;
  argumentBufferOffset_ = 0;
  isReplaying_ = false;
  hasPendingWork_ = false;
  writtenBuffers_.clear();
  barrierIndices_.clear();
}

void MetalStream::setFlushInterval(int dispatches) {
  flushInterval_ = dispatches;
}

} // namespace metal_v2
} // namespace backends
} // namespace executorch
