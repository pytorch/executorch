/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef __OBJC__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <dispatch/dispatch.h>
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
#endif

#include <unordered_map>
#include <memory>
#include <string>
#include <functional>
#include <vector>

namespace executorch {
namespace runtime {
namespace etensor {
class Tensor;
}
}
}

namespace executorch {
namespace backends {
namespace aoti {

// Forward declarations
class ETMetalKernelFunction;
class ETMetalStream;

// =======================
// SyncType - Metal synchronization options
// =======================
enum class SyncType {
    NONE,                // no commit to command buffer
    COMMIT,              // commit and flush the command buffer
    COMMIT_AND_WAIT,     // flush and wait for command buffer execution to finish
    COMMIT_AND_CONTINUE, // commit and continue with a new underlying command buffer
    COMMIT_ADAPTIVE,     // commit adaptively based on available memory
};

// =======================
// ETMetalShaderLibrary - ExecuTorch Metal shader library management
// =======================
class ETMetalShaderLibrary {
public:
    ETMetalShaderLibrary(const std::string& source);
    ~ETMetalShaderLibrary();

    std::shared_ptr<ETMetalKernelFunction> getKernelFunction(const std::string& name);

private:
    void compileLibrary();
#ifdef __OBJC__
    std::pair<id<MTLComputePipelineState>, id<MTLFunction>> getLibraryPipelineState(const std::string& functionName);

    friend class ETMetalKernelFunction;

    std::string shaderSource_;
    id<MTLLibrary> library_;
    std::unordered_map<std::string, std::pair<id<MTLComputePipelineState>, id<MTLFunction>>> pipelineStates_;
#else
    void* getLibraryPipelineState(const std::string& functionName);

    friend class ETMetalKernelFunction;

    std::string shaderSource_;
    void* library_;
    std::unordered_map<std::string, void*> pipelineStates_;
#endif
};

// =======================
// ETMetalKernelFunction - ExecuTorch Metal kernel function execution
// =======================
class ETMetalKernelFunction {
public:
#ifdef __OBJC__
    ETMetalKernelFunction(id<MTLComputePipelineState> cps, id<MTLFunction> func);
#else
    ETMetalKernelFunction(void* cps, void* func);
#endif
    ~ETMetalKernelFunction();

    void startEncoding();
    void setArg(unsigned idx, const executorch::runtime::etensor::Tensor& tensor);
    void setArg(unsigned idx, int64_t val);

    void dispatchSingle(uint64_t length);
    void dispatchSingleWithGroupSize(uint64_t length, uint64_t group_size);
    void dispatchArray(const uint64_t* length, size_t length_size);
    void dispatchArrayWithGroupSize(const uint64_t* length, size_t length_size,
                                   const uint64_t* group_size, size_t group_size_size);

    void runCommandBlock(std::function<void(void)> f);

private:
#ifdef __OBJC__
    id<MTLComputePipelineState> cps_;
    id<MTLFunction> func_;
    id<MTLComputeCommandEncoder> encoder_;
#else
    void* cps_;
    void* func_;
    void* encoder_;
#endif
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

#ifdef __OBJC__
    // Device and queue access
    id<MTLDevice> device() const { return device_; }
    id<MTLCommandQueue> commandQueue() const { return commandQueue_; }
    dispatch_queue_t queue() const { return serialQueue_; }

    // Command buffer management with lazy creation
    id<MTLCommandBuffer> commandBuffer();
    id<MTLComputeCommandEncoder> commandEncoder();

    // Synchronization methods
    void synchronize(SyncType syncType = SyncType::COMMIT_AND_WAIT);
    void synchronize(); // Overload for backward compatibility
    void endKernelCoalescing();

    // Command buffer lifecycle management
    void commitCommandBuffer(id<MTLCommandBuffer> commandBuffer);
    void flush();
    bool isEmpty() const;

    // Memory operations
    void fill(id<MTLBuffer> buffer, uint8_t value, size_t length, size_t offset, SyncType syncType = SyncType::NONE);
    void copy(id<MTLBuffer> srcBuffer, id<MTLBuffer> dstBuffer, size_t length,
             size_t srcOffset, size_t dstOffset, SyncType syncType = SyncType::NONE);

private:
    // Private members
    id<MTLDevice> device_;
    id<MTLCommandQueue> commandQueue_;
    id<MTLCommandBuffer> commandBuffer_;
    id<MTLCommandBuffer> prevCommandBuffer_;  // For commit-and-continue pattern
    id<MTLComputeCommandEncoder> commandEncoder_;
    dispatch_queue_t serialQueue_;  // For thread safety

    // Configuration
    bool enableCommitAndContinue_;

    // Private synchronization methods
    void commit();
    void commitAndWait();
    void commitAndContinue();
#else
    // C++ only interface - limited functionality
    void* device() const { return device_; }
    void* commandQueue() const { return commandQueue_; }
    void* queue() const { return serialQueue_; }

    // Basic methods for C++ compilation
    void synchronize(SyncType syncType = SyncType::COMMIT_AND_WAIT);
    void synchronize();
    bool isEmpty() const;

private:
    // Private members (void* for C++ compatibility)
    void* device_;
    void* commandQueue_;
    void* commandBuffer_;
    void* prevCommandBuffer_;
    void* commandEncoder_;
    void* serialQueue_;

    // Configuration
    bool enableCommitAndContinue_;
#endif

    // Singleton instance
    static ETMetalStream* defaultStream_;
};

// =======================
// Global storage management functions
// =======================
void storeFunctionHandle(ETMetalKernelFunction* raw_function, std::shared_ptr<ETMetalKernelFunction> function_shared_ptr);
void storeLibraryHandle(ETMetalShaderLibrary* raw_library, std::unique_ptr<ETMetalShaderLibrary> library);
bool removeFunctionHandle(ETMetalKernelFunction* raw_function);
bool removeLibraryHandle(ETMetalShaderLibrary* raw_library);

// =======================
// Global stream access functions
// =======================
ETMetalStream* getCurrentMetalStream();
void setCurrentMetalStream(ETMetalStream* stream);

// =======================
// Metal helper functions (C interface)
// =======================
#ifdef __cplusplus
extern "C" {
#endif

// Metal initialization and management
void metal_init_if_needed();
void* metal_allocate_buffer(long bytes);
void metal_cleanup_resources();

// Memory management functions for Metal
bool metal_is_device_pointer(void* ptr);
int metal_copy_memory(void* dst, const void* src, size_t nbytes, bool src_is_device, bool dst_is_device);

// Helper functions to access Metal objects
#ifdef __OBJC__
id<MTLDevice> get_metal_device();
id<MTLCommandQueue> get_metal_command_queue();
#else
void* get_metal_device();
void* get_metal_command_queue();
#endif

#ifdef __cplusplus
}

// C++ only - expose the Metal buffer mapping
#ifdef __OBJC__
extern std::unordered_map<void*, id<MTLBuffer>> ptr_to_mtl_buffer;
#endif

#endif

} // namespace aoti
} // namespace backends
} // namespace executorch
