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
class ETMetalKernelFunction {
 public:
  ETMetalKernelFunction(MTLComputePipelineState_t cps, MTLFunction_t func);
  ~ETMetalKernelFunction();

  void startEncoding();
  void setArg(unsigned idx, const executorch::runtime::etensor::Tensor& tensor);
  void setArg(unsigned idx, int64_t val);

  void dispatchSingle(uint64_t length);
  void dispatchSingleWithGroupSize(uint64_t length, uint64_t group_size);
  void dispatchArray(const uint64_t* length, size_t length_size);
  void dispatchArrayWithGroupSize(
      const uint64_t* length,
      size_t length_size,
      const uint64_t* group_size,
      size_t group_size_size);

  void runCommandBlock(std::function<void(void)> f);

 private:
  MTLComputePipelineState_t cps_;
  MTLFunction_t func_;
  MTLComputeCommandEncoder_t encoder_;
};

// =======================
// ETMetalStream - Metal command buffer and synchronization management
// =======================
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
// Metal helper functions (C interface)
// =======================
#ifdef __cplusplus
extern "C" {
#endif

// Memory management functions for Metal
void* metal_allocate_buffer(long bytes);
bool metal_is_device_pointer(void* ptr);
int metal_copy_memory(
    void* dst,
    const void* src,
    size_t nbytes,
    bool src_is_device,
    bool dst_is_device);
void metal_cleanup_resources();

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
