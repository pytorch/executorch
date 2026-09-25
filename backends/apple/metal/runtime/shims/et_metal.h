/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>
// Forward declarations for MetalPerformanceShadersGraph types
@class MPSGraph;
@class MPSCommandBuffer;
// Metal type definitions for Objective-C compilation
typedef id<MTLDevice> MTLDevice_t;
typedef id<MTLCommandQueue> MTLCommandQueue_t;
typedef id<MTLCommandBuffer> MTLCommandBuffer_t;
typedef id<MTLComputeCommandEncoder> MTLComputeCommandEncoder_t;
typedef id<MTLComputePipelineState> MTLComputePipelineState_t;
typedef id<MTLFunction> MTLFunction_t;
typedef id<MTLLibrary> MTLLibrary_t;
typedef id<MTLBuffer> MTLBuffer_t;
typedef dispatch_queue_t dispatch_queue_t;
typedef MPSGraph* MPSGraph_t;
typedef MPSCommandBuffer* MPSCommandBuffer_t;
typedef NSDictionary* NSDictionary_t;
#else
// Forward declarations for C++ compilation
typedef void* MTLDevice_t;
typedef void* MTLCommandQueue_t;
typedef void* MTLCommandBuffer_t;
typedef void* MTLComputeCommandEncoder_t;
typedef void* MTLComputePipelineState_t;
typedef void* MTLFunction_t;
typedef void* MTLLibrary_t;
typedef void* MTLBuffer_t;
typedef void* dispatch_queue_t;
typedef void* MPSGraph_t;
typedef void* MPSCommandBuffer_t;
typedef void* NSDictionary_t;
#endif

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace executorch::runtime::etensor {
class Tensor;
}

namespace executorch {
namespace backends {
namespace metal {

// Forward declarations
class ETMetalKernelFunction;
class ETMetalStream;

// =======================
// SyncType - Metal synchronization options
// =======================
enum class SyncType {
  NONE, // no commit to command buffer
  COMMIT, // commit and flush the command buffer
  COMMIT_AND_WAIT, // flush and wait for command buffer execution to finish
  COMMIT_AND_CONTINUE, // commit and continue with a new underlying command
                       // buffer
  COMMIT_ADAPTIVE, // commit adaptively based on available memory
};

// =======================
// ETMetalShaderLibrary - ExecuTorch Metal shader library management
// =======================

/**
 * @class ETMetalShaderLibrary
 * @brief Manages Metal shader library compilation and kernel function
 * retrieval.
 *
 * This class provides a high-level interface for compiling Metal shading
 * language source code into a Metal library and creating compute pipeline
 * states for kernel functions. It handles the creation and caching of Metal
 * compute pipeline states and functions, which should be reused across multiple
 * kernel dispatches.
 *
 * The class automatically compiles the provided shader source code upon
 * construction and maintains an internal cache of compute pipeline states for
 * different kernel functions to avoid redundant compilation.
 *
 * Example usage:
 * @code
 * std::string shaderSource = R"(
 *   #include <metal_stdlib>
 *   using namespace metal;
 *   kernel void my_kernel(device float* data [[buffer(0)]],
 *                        uint tid [[thread_position_in_grid]]) {
 *     data[tid] = data[tid] * 2.0;
 *   }
 * )";
 *
 * ETMetalShaderLibrary library(shaderSource);
 * auto kernelFunction = library.getKernelFunction("my_kernel");
 * @endcode
 */
class ETMetalShaderLibrary {
 public:
  ETMetalShaderLibrary(const std::string& source);
  ~ETMetalShaderLibrary();

  std::shared_ptr<ETMetalKernelFunction> getKernelFunction(
      const std::string& name);

 private:
  void compileLibrary();
  std::pair<MTLComputePipelineState_t, MTLFunction_t> getLibraryPipelineState(
      const std::string& functionName);

  friend class ETMetalKernelFunction;

  std::string shaderSource_;
  MTLLibrary_t library_;
  std::unordered_map<
      std::string,
      std::pair<MTLComputePipelineState_t, MTLFunction_t>>
      pipelineStates_;
};

// =======================
// ETMetalKernelFunction - ExecuTorch Metal kernel function execution
// =======================

/**
 * @class ETMetalKernelFunction
 * @brief Represents a Metal compute kernel function ready for execution.
 *
 * This class encapsulates a Metal compute pipeline state and function,
 * providing a high-level interface for setting kernel arguments and dispatching
 * compute work to the GPU. It handles the encoding of compute commands and
 * manages the interaction with Metal's compute command encoder.
 *
 * The class supports different dispatch patterns:
 * - Single-dimension dispatch for linear workloads
 * - Multi-dimensional dispatch for grid-based workloads
 * - Custom thread group sizes for performance optimization
 *
 * Kernel arguments can be set using tensors (which will be mapped to Metal
 * buffers) or scalar values. The class handles the encoding of these arguments
 * into the compute command encoder.
 *
 * Example usage:
 * @code
 * // Get kernel function from library
 * auto kernelFunction = library.getKernelFunction("vector_add");
 *
 * // Start encoding commands
 * kernelFunction->startEncoding();
 *
 * // Set tensor arguments
 * kernelFunction->setArg(0, inputTensorA);
 * kernelFunction->setArg(1, inputTensorB);
 * kernelFunction->setArg(2, outputTensor);
 *
 * // Set scalar argument
 * kernelFunction->setArg(3, static_cast<int64_t>(numElements));
 *
 * // Dispatch for linear workload
 * kernelFunction->dispatchSingle(numElements);
 * @endcode
 */
class ETMetalKernelFunction {
 public:
  ETMetalKernelFunction(MTLComputePipelineState_t cps, MTLFunction_t func);
  ~ETMetalKernelFunction();

  void startEncoding();
  void setArg(unsigned idx, const executorch::runtime::etensor::Tensor& tensor);
  void setArg(unsigned idx, int64_t val);
  void setArg(unsigned idx, uint32_t val);
  void setArg(unsigned idx, float val);
  void setArg(unsigned idx, bool val);
  void setArg(unsigned idx, const void* data, size_t size);

  // Helper for Metal uint3 struct
  void setArgUint3(unsigned idx, uint32_t x, uint32_t y, uint32_t z);

  void dispatchSingle(uint64_t length);
  void dispatchSingleWithGroupSize(uint64_t length, uint64_t group_size);
  void dispatchArray(const uint64_t* length, size_t length_size);
  void dispatchArrayWithGroupSize(
      const uint64_t* length,
      size_t length_size,
      const uint64_t* group_size,
      size_t group_size_size);

  // Dispatch with explicit threadgroup count (not thread count)
  void dispatchThreadgroups(
      uint64_t gridX,
      uint64_t gridY,
      uint64_t gridZ,
      uint64_t threadsX,
      uint64_t threadsY,
      uint64_t threadsZ);

  void runCommandBlock(std::function<void(void)> f);

 private:
  MTLComputePipelineState_t cps_;
  MTLFunction_t func_;
  MTLComputeCommandEncoder_t encoder_;
};

// =======================
// ETMetalStream - Metal command buffer and synchronization management
// =======================

/**
 * @class ETMetalStream
 * @brief Manages Metal compute command streams and provides GPU
 * synchronization.
 *
 * This class serves as the central management hub for Metal GPU operations,
 * providing a stream-based abstraction similar to CUDA streams. It handles
 * command buffer lifecycle, compute command encoder management, and various
 * synchronization patterns required for efficient GPU computation.
 *
 * Key features:
 * - Lazy command buffer and encoder creation for optimal resource usage
 * - Thread-safe operations using serial dispatch queues
 * - Multiple synchronization modes (COMMIT, COMMIT_AND_WAIT,
 * COMMIT_AND_CONTINUE, etc.)
 * - Kernel coalescing to batch multiple operations efficiently
 * - MPSGraph integration for executing fall back operations (mm, conv, sdpa)
 * - Memory operations (copy, fill) with GPU acceleration via blit encoders
 *
 * The stream follows PyTorch's MPS stream design patterns, providing similar
 * semantics for command buffer management and synchronization.
 *
 * Example usage:
 * @code
 * // Get current stream (typically the default stream)
 * ETMetalStream* stream = getCurrentMetalStream();
 *
 * // Execute kernel operations (handled automatically)
 * auto kernelFunction = library.getKernelFunction("my_kernel");
 * kernelFunction->startEncoding();
 * kernelFunction->setArg(0, inputTensor);
 * kernelFunction->dispatchSingle(numElements);
 *
 * // Synchronize to ensure completion
 * stream->synchronize(SyncType::COMMIT_AND_WAIT);
 *
 * // Copy between GPU buffers using blit encoder
 * stream->copy(srcBuffer, dstBuffer, numBytes, 0, 0, SyncType::COMMIT);
 * @endcode
 */
class ETMetalStream {
 public:
  ETMetalStream();
  ~ETMetalStream();

  // Get the default stream (singleton)
  static ETMetalStream* getDefaultStream();

  // Device and queue access
  MTLDevice_t device() const {
    return device_;
  }
  MTLCommandQueue_t commandQueue() const {
    return commandQueue_;
  }
  dispatch_queue_t queue() const {
    return serialQueue_;
  }

  // Synchronization methods
  void synchronize(SyncType syncType = SyncType::COMMIT_AND_WAIT);
  void synchronize(); // Overload for backward compatibility
  bool isEmpty() const;
  // How many COMMIT_AND_WAIT syncs have completed. Work queued before one is
  // done once it completes.
  uint64_t completedWaits() const {
    return completedWaits_;
  }

  // Command buffer management with lazy creation
  MPSCommandBuffer_t commandBuffer();
  MTLComputeCommandEncoder_t commandEncoder();

  void endKernelCoalescing();

  // MPSGraph execution. `settle_aliases` is for a graph fed an aliasing buffer
  // (see get_mtl_buffer): the stream then waits for the GPU both before and
  // after encoding the graph, as one step.
  void executeMPSGraph(
      MPSGraph_t mpsGraph,
      NSDictionary_t feeds,
      NSDictionary_t results,
      SyncType syncType = SyncType::COMMIT_ADAPTIVE,
      bool settle_aliases = false);

  // Command buffer lifecycle management
  void commitCommandBuffer(MTLCommandBuffer_t commandBuffer);
  void flush();

  // Dispatch pipelining: periodically commitAndContinue so the driver
  // can prepare batch N+1 while the GPU executes batch N.
  void notifyDispatch();
  void setFlushInterval(int interval);

  // Memory operations
  void fill(
      MTLBuffer_t buffer,
      uint8_t value,
      size_t length,
      size_t offset,
      SyncType syncType = SyncType::NONE);
  void copy(
      MTLBuffer_t srcBuffer,
      MTLBuffer_t dstBuffer,
      size_t length,
      size_t srcOffset,
      size_t dstOffset,
      SyncType syncType = SyncType::NONE);

 private:
  // Private synchronization methods
  void commit();
  void commitAndWait();
  void commitAndContinue();

 private:
  // Private members
  MTLDevice_t device_;
  MTLCommandQueue_t commandQueue_;
  MPSCommandBuffer_t commandBuffer_;
  MPSCommandBuffer_t prevCommandBuffer_; // For commit-and-continue pattern
  MTLComputeCommandEncoder_t commandEncoder_;
  dispatch_queue_t serialQueue_; // For thread safety

  // Configuration
  bool enableCommitAndContinue_;
  int flushInterval_; // 0 = disabled, >0 = flush every N dispatches
  std::atomic<int> dispatchCount_; // dispatches since last flush
  uint64_t completedWaits_ = 0;

  // Singleton instance
  static ETMetalStream* defaultStream_;
};

// =======================
// Global storage management functions
// =======================
void storeFunctionHandle(
    ETMetalKernelFunction* raw_function,
    std::shared_ptr<ETMetalKernelFunction> function_shared_ptr);
void storeLibraryHandle(
    ETMetalShaderLibrary* raw_library,
    std::unique_ptr<ETMetalShaderLibrary> library);
bool removeFunctionHandle(ETMetalKernelFunction* raw_function);
bool removeLibraryHandle(ETMetalShaderLibrary* raw_library);

// =======================
// Global stream access functions
// =======================
ETMetalStream* getCurrentMetalStream();
void setCurrentMetalStream(ETMetalStream* stream);

// =======================
// Metal stream synchronization functions (C++ interface with exceptions)
// =======================
void synchronize_metal_stream();
void synchronize_metal_stream_with_type(int sync_type);

// =======================
// Metal helper functions (C interface)
// =======================
#ifdef __cplusplus
extern "C" {
#endif

// Memory management functions for Metal
void* metal_allocate_buffer(long bytes);
// Like metal_allocate_buffer, and sets `*may_be_in_use` when the buffer comes
// from the pool and the stream has not waited since it was freed. Buffers are
// recycled without a wait, so work queued before such a buffer was freed may
// still use it: the CPU must wait for that work before writing it.
void* metal_allocate_buffer_tracking_use(long bytes, bool* may_be_in_use);
void metal_deallocate_buffer(void* ptr);
bool metal_is_device_pointer(void* ptr);
// Whether any of the `nbytes` at `ptr` lie in memory the GPU can write: a Metal
// buffer, or CPU memory in a region with a buffer (metal_register_cpu_view).
// Unlike metal_is_device_pointer, this does not need `ptr` to be a buffer start
// or a registered view. It scans every buffer, so it is for rare paths.
bool metal_overlaps_gpu_memory(const void* ptr, size_t nbytes);
int metal_copy_memory(
    void* dst,
    const void* src,
    size_t nbytes,
    bool src_is_device,
    bool dst_is_device);
void metal_cleanup_resources();
bool metal_buffer_nocopy(void* ptr, size_t nbytes, bool map_ptr_to_buffer);

// Records that `view_ptr` points inside the Metal buffer that owns `base_ptr`,
// so the view is bound as that buffer plus an offset. Giving a view its own
// MTLBuffer over the same memory does not work: Metal treats the two buffers as
// unrelated, and a write through one is not seen by a read of the other in the
// same command buffer. Registrations are counted: every tensor handle at
// `view_ptr` holds one, taken with metal_register_view when the view is created
// or with metal_retain_view when another handle is made for the same address,
// and gives it back with metal_unregister_view. metal_retain_view does nothing
// for an address that is not a registered view.
bool metal_register_view(void* view_ptr, void* base_ptr);
void metal_retain_view(void* view_ptr);
// Returns whether that was the last handle registered at `view_ptr`.
bool metal_unregister_view(void* view_ptr);

// Records that `view_ptr`, `view_nbytes` long, is a view of the CPU memory
// that starts at `region` and is `region_nbytes` long. A region's views and its
// start are bound into one no-copy buffer over it, the way views of a Metal
// buffer are bound into that buffer, so that Metal orders their uses. The
// buffer is made with the region's first view. For memory the runtime
// allocated (`owned`), it is kept until metal_release_cpu_region(), since views
// come and go and a new buffer for the same memory would not be ordered with
// work queued on the old one. Other memory can be freed and reused behind the
// runtime's back, so its buffer is released, after a wait, with its last view.
// Views are counted and released like views of a Metal buffer. If a view
// reaches past `region_nbytes` (the extent is not always known up front), the
// buffer is replaced by a longer one after queued work is done.
bool metal_register_cpu_view(
    void* view_ptr,
    size_t view_nbytes,
    void* region,
    size_t region_nbytes,
    bool owned);
bool metal_is_cpu_view(void* ptr);
// Whether `ptr` is a view of a CPU region or the start of one.
bool metal_is_cpu_memory(void* ptr);
// The start of the CPU region `ptr` is a view of, if it is one.
bool metal_cpu_view_region(void* ptr, void** region);
// Releases the buffer of the CPU region starting at `region`, after waiting for
// queued work; call before freeing its memory. Returns whether there was one.
bool metal_release_cpu_region(void* region);

// Helper functions to access Metal objects
MTLDevice_t get_metal_device();
MTLCommandQueue_t get_metal_command_queue();

#ifdef __cplusplus
}

// C++ only - expose the Metal buffer mapping
#ifdef __OBJC__
extern std::unordered_map<void*, MTLBuffer_t> ptr_to_mtl_buffer;

// Finds the Metal buffer holding `ptr` and how far into it `ptr` is. Handles
// both a buffer's own address and a registered view. Returns false for memory
// Metal does not own.
bool metal_resolve_buffer(void* ptr, MTLBuffer_t* buffer, size_t* offset);
#endif

#endif

} // namespace metal
} // namespace backends
} // namespace executorch
