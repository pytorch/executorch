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

  // Command buffer management with lazy creation
  MPSCommandBuffer_t commandBuffer();
  MTLComputeCommandEncoder_t commandEncoder();

  void endKernelCoalescing();

  // MPSGraph execution
  void executeMPSGraph(
      MPSGraph_t mpsGraph,
      NSDictionary_t feeds,
      NSDictionary_t results,
      SyncType syncType = SyncType::COMMIT_ADAPTIVE);

  // Command buffer lifecycle management
  void commitCommandBuffer(MTLCommandBuffer_t commandBuffer);
  void flush();

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
// Metal backend timing statistics
// =======================
// Execute timing
double get_metal_backend_execute_total_ms();
int64_t get_metal_backend_execute_call_count();
// Returns map of method_name -> (total_ms, call_count)
std::unordered_map<std::string, std::pair<double, int64_t>>
get_metal_backend_per_method_stats();

// Init timing
double get_metal_backend_init_total_ms();
int64_t get_metal_backend_init_call_count();
// Returns map of method_name -> (total_ms, call_count) for init
std::unordered_map<std::string, std::pair<double, int64_t>>
get_metal_backend_init_per_method_stats();

// Reset all timing stats
void reset_metal_backend_execute_stats();

// =======================
// Metal helper functions (C interface)
// =======================
#ifdef __cplusplus
extern "C" {
#endif

// Memory management functions for Metal
void* metal_allocate_buffer(long bytes);
void metal_deallocate_buffer(void* ptr);
bool metal_is_device_pointer(void* ptr);
int metal_copy_memory(
    void* dst,
    const void* src,
    size_t nbytes,
    bool src_is_device,
    bool dst_is_device);
void metal_cleanup_resources();
bool metal_buffer_nocopy(void* ptr, size_t nbytes, bool map_ptr_to_buffer);

// Helper functions to access Metal objects
MTLDevice_t get_metal_device();
MTLCommandQueue_t get_metal_command_queue();

#ifdef __cplusplus
}

// C++ only - expose the Metal buffer mapping
#ifdef __OBJC__
extern std::unordered_map<void*, MTLBuffer_t> ptr_to_mtl_buffer;
#endif

#endif

} // namespace metal
} // namespace backends
} // namespace executorch
