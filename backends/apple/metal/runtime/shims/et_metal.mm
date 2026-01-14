/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <Foundation/Foundation.h>
#include <simd/simd.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/backends/apple/metal/runtime/shims/et_metal.h>
#include <algorithm>
#include <optional>
#include <exception>

namespace executorch {
namespace backends {
namespace metal {

// =======================
// Exception-Safe Dispatch Function (similar to PyTorch MPS)
// =======================

void dispatch_sync_with_rethrow(dispatch_queue_t queue, void (^block)()) {
    __block std::optional<std::exception_ptr> block_exception;
    dispatch_sync(queue, ^() {
        try {
            block();
        } catch (...) {
            block_exception = std::current_exception();
        }
    });
    if (block_exception) {
        std::rethrow_exception(*block_exception);
    }
}

// =======================
// Global Variables and Storage
// ================


// Global Metal buffer mapping - accessible for MPS shim
std::unordered_map<void*, id<MTLBuffer>> ptr_to_mtl_buffer;

// Global storage to keep shared_ptr alive while raw pointers are used
static std::unordered_map<ETMetalKernelFunction*, std::shared_ptr<ETMetalKernelFunction>> function_storage;
static std::unordered_map<ETMetalShaderLibrary*, std::unique_ptr<ETMetalShaderLibrary>> library_storage;

// Static singleton instance for default stream
ETMetalStream* ETMetalStream::defaultStream_ = nullptr;

// Thread-local current stream
static thread_local ETMetalStream* currentStream_ = nullptr;

// =======================
// Metal Helper Functions (C Interface)
// =======================

extern "C" {

void* metal_allocate_buffer(long bytes) {
    ETMetalStream* stream = getCurrentMetalStream();
    id<MTLDevice> device = stream->device();
    if (!device) {
        ET_LOG(Error, "Failed to get Metal device from stream");
        return nullptr;
    }

    @autoreleasepool {
        id<MTLBuffer> buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!buffer) {
            ET_LOG(Error, "Failed to allocate %ld bytes on Metal device", bytes);
            return nullptr;
        }

        void* ptr = [buffer contents];
        ptr_to_mtl_buffer[ptr] = buffer;

        ET_LOG(Debug, "Allocated %ld bytes on Metal device", bytes);
        return ptr;
    }
}

void metal_deallocate_buffer(void* ptr) {
    @autoreleasepool {
        auto it = ptr_to_mtl_buffer.find(ptr);
        if (it != ptr_to_mtl_buffer.end()) {
            id<MTLBuffer> buffer = it->second;
            [buffer release];
            ptr_to_mtl_buffer.erase(it);
            ET_LOG(Debug, "Deallocated Metal buffer for pointer %p", ptr);
            ptr = nullptr;
        } else {
            ET_LOG(Error, "Failed to find Metal buffer for pointer %p", ptr);
        }
    }
}

void metal_cleanup_resources() {
    if (!ptr_to_mtl_buffer.empty()) {
        @autoreleasepool {
            for (auto& pair : ptr_to_mtl_buffer) {
                pair.second = nil;
            }
            ptr_to_mtl_buffer.clear();
        }
    }
}

bool metal_buffer_nocopy(void* ptr, size_t nbytes, bool map_ptr_to_buffer) {
    id<MTLDevice> device = get_metal_device();
    id<MTLBuffer> subBuffer = [device newBufferWithBytesNoCopy:ptr
                                                        length:nbytes
                                                        options:MTLResourceCPUCacheModeWriteCombined | MTLResourceStorageModeShared
                                                    deallocator:nil];
    if (!subBuffer) {
        ET_LOG(Error, "metal_buffer_nocopy: Failed to create no-copy buffer (ptr=%p, nbytes=%zu)", ptr, nbytes);
        return false;
    }

    if (map_ptr_to_buffer) {
        ptr_to_mtl_buffer[ptr] = subBuffer;  // Map contents to buffer
    }

    return true;
}

bool metal_is_device_pointer(void* ptr) {
    return ptr_to_mtl_buffer.find(ptr) != ptr_to_mtl_buffer.end();
}

int metal_copy_memory(void* dst, const void* src, size_t nbytes, bool src_is_device, bool dst_is_device) {
    if (!src || !dst || nbytes == 0) {
        ET_LOG(Error, "Metal copy: Invalid parameters");
        return -1;
    }

    @autoreleasepool {
        // Case 1: Device-to-device copy - use GPU blit encoder (most efficient)
        if (src_is_device && dst_is_device) {
            auto src_it = ptr_to_mtl_buffer.find(const_cast<void*>(src));
            auto dst_it = ptr_to_mtl_buffer.find(dst);

            if (src_it != ptr_to_mtl_buffer.end() && dst_it != ptr_to_mtl_buffer.end()) {
                id<MTLBuffer> srcBuffer = src_it->second;
                id<MTLBuffer> dstBuffer = dst_it->second;

                // Calculate offsets relative to buffer base
                size_t srcOffset = static_cast<const uint8_t*>(src) - static_cast<const uint8_t*>([srcBuffer contents]);
                size_t dstOffset = static_cast<uint8_t*>(dst) - static_cast<uint8_t*>([dstBuffer contents]);

                // Use Metal's blit encoder for GPU-accelerated copy
                ETMetalStream* stream = getCurrentMetalStream();
                stream->copy(srcBuffer, dstBuffer, nbytes, srcOffset, dstOffset, SyncType::NONE);

                ET_LOG(Debug, "Metal device-to-device copy (GPU blit): %zu bytes", nbytes);
                return 0;
            }

            ET_LOG(Error, "Metal copy: Device pointers not found in buffer map");
            return -1;
        }

        // Case 2: Host-to-device or device-to-host - use memcpy with shared memory
        // Since Metal uses shared storage mode, CPU and GPU access the same memory
        std::memcpy(dst, src, nbytes);

        // Synchronize only if we need to ensure GPU operations complete before CPU reads
        // (device-to-host case where GPU may have written data)
        if (src_is_device && !dst_is_device) {
            // Ensure any pending GPU writes to source complete before CPU reads
            ETMetalStream* stream = getCurrentMetalStream();
            stream->synchronize(SyncType::COMMIT_AND_WAIT);
        }

        ET_LOG(Debug, "Metal memory copy (memcpy): %zu bytes, src_device=%d, dst_device=%d",
               nbytes, src_is_device, dst_is_device);
    }

    return 0;
}

id<MTLDevice> get_metal_device() {
    // Use stream-based device access
    ETMetalStream* stream = getCurrentMetalStream();
    return stream->device();
}

id<MTLCommandQueue> get_metal_command_queue() {
    // Use stream-based queue access
    ETMetalStream* stream = getCurrentMetalStream();
    return stream->commandQueue();
}

} // extern "C"

// =======================
// ETMetalShaderLibrary Implementation
// =======================

ETMetalShaderLibrary::ETMetalShaderLibrary(const std::string& source) : shaderSource_(source) {
    compileLibrary();
}

ETMetalShaderLibrary::~ETMetalShaderLibrary() {
    @autoreleasepool {
        if (library_) {
            [library_ release];
            library_ = nil;
        }

        for (auto& pair : pipelineStates_) {
            [pair.second.first release];
            [pair.second.second release];
        }
        pipelineStates_.clear();
    }
}

void ETMetalShaderLibrary::compileLibrary() {
    @autoreleasepool {
        id<MTLDevice> device = get_metal_device();
        if (!device) {
            ET_LOG(Error, "ETMetalShaderLibrary: Failed to get Metal device");
            return;
        }

        NSString* sourceString = [NSString stringWithUTF8String:shaderSource_.c_str()];
        NSError* error = nil;

        library_ = [device newLibraryWithSource:sourceString options:nil error:&error];
        if (!library_ || error) {
            ET_LOG(Error, "ETMetalShaderLibrary: Failed to compile shader library: %s",
                   error ? [[error localizedDescription] UTF8String] : "unknown error");
            return;
        }

        [library_ retain];
        ET_LOG(Debug, "ETMetalShaderLibrary: Successfully compiled shader library");
    }
}

std::pair<id<MTLComputePipelineState>, id<MTLFunction>> ETMetalShaderLibrary::getLibraryPipelineState(const std::string& functionName) {
    auto it = pipelineStates_.find(functionName);
    if (it != pipelineStates_.end()) {
        return it->second;
    }

    @autoreleasepool {
        if (!library_) {
            ET_LOG(Error, "ETMetalShaderLibrary: Library not compiled");
            return {nil, nil};
        }

        id<MTLDevice> device = get_metal_device();
        if (!device) {
            ET_LOG(Error, "ETMetalShaderLibrary: Failed to get Metal device");
            return {nil, nil};
        }

        NSString* funcName = [NSString stringWithUTF8String:functionName.c_str()];
        id<MTLFunction> function = [library_ newFunctionWithName:funcName];
        if (!function) {
            ET_LOG(Error, "ETMetalShaderLibrary: Failed to get function '%s'", functionName.c_str());
            return {nil, nil};
        }

        NSError* error = nil;
        id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipelineState || error) {
            ET_LOG(Error, "ETMetalShaderLibrary: Failed to create pipeline state for '%s': %s",
                   functionName.c_str(), error ? [[error localizedDescription] UTF8String] : "unknown error");
            [function release];
            return {nil, nil};
        }

        [pipelineState retain];
        [function retain];
        pipelineStates_[functionName] = {pipelineState, function};

        ET_LOG(Debug, "ETMetalShaderLibrary: Created pipeline state for function '%s'", functionName.c_str());
        return {pipelineState, function};
    }
}

std::shared_ptr<ETMetalKernelFunction> ETMetalShaderLibrary::getKernelFunction(const std::string& name) {
    auto pipelineStatePair = getLibraryPipelineState(name);
    if (!pipelineStatePair.first || !pipelineStatePair.second) {
        ET_LOG(Error, "ETMetalShaderLibrary::getKernelFunction: Failed to get pipeline state for '%s'", name.c_str());
        return nullptr;
    }

    return std::make_shared<ETMetalKernelFunction>(pipelineStatePair.first, pipelineStatePair.second);
}

// =======================
// ETMetalKernelFunction Implementation
// =======================

ETMetalKernelFunction::ETMetalKernelFunction(id<MTLComputePipelineState> cps, id<MTLFunction> func)
    : cps_(cps), func_(func), encoder_(nil) {
    if (cps_) [cps_ retain];
    if (func_) [func_ retain];
}

ETMetalKernelFunction::~ETMetalKernelFunction() {
    @autoreleasepool {
        // Don't release encoder_ here - the stream owns it
        // Only clean up our own references
        if (cps_) {
            [cps_ release];
            cps_ = nil;
        }
        if (func_) {
            [func_ release];
            func_ = nil;
        }

        encoder_ = nil; // Clear reference without releasing
    }
}

void ETMetalKernelFunction::startEncoding() {
    @autoreleasepool {
        // Don't retain/release the encoder - just get reference from stream
        ETMetalStream* stream = getCurrentMetalStream();
        encoder_ = stream->commandEncoder(); // Use stream's managed encoder
        if (!encoder_) {
            ET_LOG(Error, "ETMetalKernelFunction: Failed to get encoder from stream");
            return;
        }

        // Don't retain - stream owns the encoder
        [encoder_ setComputePipelineState:cps_];

        ET_LOG(Debug, "ETMetalKernelFunction: Started encoding with stream-managed encoder");
    }
}

void ETMetalKernelFunction::setArg(unsigned idx, const executorch::runtime::etensor::Tensor& tensor) {
    if (!encoder_) {
        ET_LOG(Error, "ETMetalKernelFunction::setArg: No active encoder");
        return;
    }

    void* data_ptr = tensor.mutable_data_ptr();
    size_t totalSize = tensor.numel() * tensor.element_size();

    auto it = ptr_to_mtl_buffer.find(data_ptr);
    if (it != ptr_to_mtl_buffer.end()) {
        // Use existing Metal buffer
        id<MTLBuffer> mtlBuffer = it->second;
        [encoder_ setBuffer:mtlBuffer offset:0 atIndex:idx];
        ET_LOG(Debug, "ETMetalKernelFunction::setArg: Set Metal buffer at index %u (size: %zu)", idx, totalSize);
    } else {
        // Handle CPU tensor data
        if (totalSize <= 4096) {
            // Use setBytes for small data (more efficient)
            [encoder_ setBytes:data_ptr length:totalSize atIndex:idx];
            ET_LOG(Debug, "ETMetalKernelFunction::setArg: Set CPU tensor via setBytes at index %u (size: %zu)", idx, totalSize);
        } else {
            // Create temporary buffer for large data (should be rare)
            @autoreleasepool {
                id<MTLDevice> device = get_metal_device();
                if (device) {
                    id<MTLBuffer> tempBuffer = [device newBufferWithBytes:data_ptr
                                                                   length:totalSize
                                                                  options:MTLResourceStorageModeShared];
                    if (tempBuffer) {
                        [encoder_ setBuffer:tempBuffer offset:0 atIndex:idx];
                        ET_LOG(Debug, "ETMetalKernelFunction::setArg: Set large CPU tensor via temporary buffer at index %u (size: %zu)", idx, totalSize);
                    } else {
                        ET_LOG(Error, "ETMetalKernelFunction::setArg: Failed to create temporary buffer for index %u", idx);
                    }
                } else {
                    ET_LOG(Error, "ETMetalKernelFunction::setArg: No Metal device available for index %u", idx);
                }
            }
        }
    }
}

void ETMetalKernelFunction::setArg(unsigned idx, int64_t val) {
    if (!encoder_) {
        ET_LOG(Error, "ETMetalKernelFunction::setArg: No active encoder");
        return;
    }

    [encoder_ setBytes:&val length:sizeof(int64_t) atIndex:idx];
    ET_LOG(Debug, "ETMetalKernelFunction::setArg: Set int64_t value %lld at index %u", val, idx);
}

void ETMetalKernelFunction::setArg(unsigned idx, uint32_t val) {
    if (!encoder_) {
        ET_LOG(Error, "ETMetalKernelFunction::setArg: No active encoder");
        return;
    }

    [encoder_ setBytes:&val length:sizeof(uint32_t) atIndex:idx];
    ET_LOG(Debug, "ETMetalKernelFunction::setArg: Set uint32_t value %u at index %u", val, idx);
}

void ETMetalKernelFunction::setArg(unsigned idx, float val) {
    if (!encoder_) {
        ET_LOG(Error, "ETMetalKernelFunction::setArg: No active encoder");
        return;
    }

    [encoder_ setBytes:&val length:sizeof(float) atIndex:idx];
    ET_LOG(Debug, "ETMetalKernelFunction::setArg: Set float value %f at index %u", val, idx);
}

void ETMetalKernelFunction::setArg(unsigned idx, bool val) {
    if (!encoder_) {
        ET_LOG(Error, "ETMetalKernelFunction::setArg: No active encoder");
        return;
    }

    [encoder_ setBytes:&val length:sizeof(bool) atIndex:idx];
    ET_LOG(Debug, "ETMetalKernelFunction::setArg: Set bool value %s at index %u", val ? "true" : "false", idx);
}

void ETMetalKernelFunction::setArg(unsigned idx, const void* data, size_t size) {
    if (!encoder_) {
        ET_LOG(Error, "ETMetalKernelFunction::setArg: No active encoder");
        return;
    }

    [encoder_ setBytes:data length:size atIndex:idx];
    ET_LOG(Debug, "ETMetalKernelFunction::setArg: Set bytes at index %u (size: %zu)", idx, size);
}

void ETMetalKernelFunction::setArgUint3(unsigned idx, uint32_t x, uint32_t y, uint32_t z) {
    if (!encoder_) {
        ET_LOG(Error, "ETMetalKernelFunction::setArgUint3: No active encoder");
        return;
    }

    // Use SIMD library's uint3 type which matches Metal shader's uint3 layout
    simd_uint3 val = {x, y, z};
    [encoder_ setBytes:&val length:sizeof(simd_uint3) atIndex:idx];
    ET_LOG(Debug, "ETMetalKernelFunction::setArgUint3: Set uint3{%u, %u, %u} at index %u", x, y, z, idx);
}

void ETMetalKernelFunction::dispatchSingle(uint64_t length) {
    if (!encoder_) {
        ET_LOG(Error, "ETMetalKernelFunction::dispatchSingle: No active encoder");
        return;
    }

    const auto maxThreadsPerGroup = static_cast<uint64_t>([cps_ maxTotalThreadsPerThreadgroup]);
    uint64_t actualGroupSize = std::min(maxThreadsPerGroup, length);

    auto size = MTLSizeMake(length, 1, 1);
    auto threadGroupSize = MTLSizeMake(actualGroupSize, 1, 1);

    [encoder_ dispatchThreads:size threadsPerThreadgroup:threadGroupSize];
    ET_LOG(Debug, "ETMetalKernelFunction::dispatchSingle: Dispatched with length %llu, group size %llu", length, actualGroupSize);

}

void ETMetalKernelFunction::dispatchSingleWithGroupSize(uint64_t length, uint64_t group_size) {
    if (!encoder_) {
        ET_LOG(Error, "ETMetalKernelFunction::dispatchSingleWithGroupSize: No active encoder");
        return;
    }

    const auto maxThreadsPerGroup = static_cast<uint64_t>([cps_ maxTotalThreadsPerThreadgroup]);
    uint64_t actualGroupSize = group_size > 0 ? std::min(group_size, maxThreadsPerGroup) : std::min(maxThreadsPerGroup, length);

    auto size = MTLSizeMake(length, 1, 1);
    auto threadGroupSize = MTLSizeMake(actualGroupSize, 1, 1);

    [encoder_ dispatchThreads:size threadsPerThreadgroup:threadGroupSize];
    ET_LOG(Debug, "ETMetalKernelFunction::dispatchSingleWithGroupSize: Dispatched with length %llu, group size %llu", length, actualGroupSize);

}

void ETMetalKernelFunction::dispatchArray(const uint64_t* length, size_t length_size) {
    if (!encoder_) {
        ET_LOG(Error, "ETMetalKernelFunction::dispatchArray: No active encoder");
        return;
    }

    if (!length || length_size == 0) {
        ET_LOG(Error, "ETMetalKernelFunction::dispatchArray: Invalid length array");
        return;
    }

    const auto maxThreadsPerGroup = static_cast<uint64_t>([cps_ maxTotalThreadsPerThreadgroup]);

    MTLSize size, threadGroupSize;

    if (length_size == 1) {
        size = MTLSizeMake(length[0], 1, 1);
        uint64_t actualGroupSize = std::min(maxThreadsPerGroup, length[0]);
        threadGroupSize = MTLSizeMake(actualGroupSize, 1, 1);
    } else if (length_size == 2) {
        size = MTLSizeMake(length[0], length[1], 1);
        uint64_t groupX = std::min(static_cast<uint64_t>(32), length[0]);
        uint64_t groupY = maxThreadsPerGroup / groupX;
        threadGroupSize = MTLSizeMake(groupX, groupY, 1);
    } else {
        size = MTLSizeMake(length[0], length[1], length_size > 2 ? length[2] : 1);
        uint64_t groupX = std::min(static_cast<uint64_t>(8), length[0]);
        uint64_t groupY = std::min(static_cast<uint64_t>(8), length[1]);
        uint64_t groupZ = maxThreadsPerGroup / (groupX * groupY);
        threadGroupSize = MTLSizeMake(groupX, groupY, groupZ);
    }

    [encoder_ dispatchThreads:size threadsPerThreadgroup:threadGroupSize];
    ET_LOG(Debug, "ETMetalKernelFunction::dispatchArray: Dispatched %zuD with size [%lu, %lu, %lu], group [%lu, %lu, %lu]",
           length_size, size.width, size.height, size.depth,
           threadGroupSize.width, threadGroupSize.height, threadGroupSize.depth);

}

void ETMetalKernelFunction::dispatchArrayWithGroupSize(const uint64_t* length, size_t length_size,
                                                      const uint64_t* group_size, size_t group_size_size) {
    if (!encoder_) {
        ET_LOG(Error, "ETMetalKernelFunction::dispatchArrayWithGroupSize: No active encoder");
        return;
    }

    if (!length || length_size == 0) {
        ET_LOG(Error, "ETMetalKernelFunction::dispatchArrayWithGroupSize: Invalid length array");
        return;
    }

    const auto maxThreadsPerGroup = static_cast<uint64_t>([cps_ maxTotalThreadsPerThreadgroup]);

    MTLSize size, threadGroupSize;

    if (length_size == 1) {
        size = MTLSizeMake(length[0], 1, 1);
        uint64_t actualGroupSize = maxThreadsPerGroup;
        if (group_size && group_size_size > 0) {
            actualGroupSize = std::min(maxThreadsPerGroup, group_size[0]);
        }
        threadGroupSize = MTLSizeMake(actualGroupSize, 1, 1);
    } else if (length_size == 2) {
        size = MTLSizeMake(length[0], length[1], 1);
        uint64_t groupX = std::min(static_cast<uint64_t>(32), length[0]);
        uint64_t groupY = maxThreadsPerGroup / groupX;
        if (group_size && group_size_size >= 2) {
            groupX = std::min(static_cast<uint64_t>(group_size[0]), length[0]);
            groupY = std::min(static_cast<uint64_t>(group_size[1]), length[1]);
        }
        threadGroupSize = MTLSizeMake(groupX, groupY, 1);
    } else {
        size = MTLSizeMake(length[0], length[1], length_size > 2 ? length[2] : 1);
        uint64_t groupX = std::min(static_cast<uint64_t>(8), length[0]);
        uint64_t groupY = std::min(static_cast<uint64_t>(8), length[1]);
        uint64_t groupZ = maxThreadsPerGroup / (groupX * groupY);
        if (group_size && group_size_size >= 3) {
            groupX = std::min(static_cast<uint64_t>(group_size[0]), length[0]);
            groupY = std::min(static_cast<uint64_t>(group_size[1]), length[1]);
            groupZ = std::min(static_cast<uint64_t>(group_size[2]), length_size > 2 ? length[2] : 1);
        }
        threadGroupSize = MTLSizeMake(groupX, groupY, groupZ);
    }

    [encoder_ dispatchThreads:size threadsPerThreadgroup:threadGroupSize];
    ET_LOG(Debug, "ETMetalKernelFunction::dispatchArrayWithGroupSize: Dispatched %zuD with size [%lu, %lu, %lu], group [%lu, %lu, %lu]",
           length_size, size.width, size.height, size.depth,
           threadGroupSize.width, threadGroupSize.height, threadGroupSize.depth);

}

void ETMetalKernelFunction::dispatchThreadgroups(uint64_t gridX, uint64_t gridY, uint64_t gridZ,
                                                  uint64_t threadsX, uint64_t threadsY, uint64_t threadsZ) {
    if (!encoder_) {
        ET_LOG(Error, "ETMetalKernelFunction::dispatchThreadgroups: No active encoder");
        return;
    }

    if (!cps_) {
        ET_LOG(Error, "ETMetalKernelFunction::dispatchThreadgroups: No compute pipeline state");
        return;
    }

    // Calculate total threads per threadgroup
    uint64_t totalThreads = threadsX * threadsY * threadsZ;

    const auto maxThreadsPerGroup = static_cast<uint64_t>([cps_ maxTotalThreadsPerThreadgroup]);

    // Validate total thread count
    if (totalThreads > maxThreadsPerGroup) {
        ET_LOG(Error, "ETMetalKernelFunction::dispatchThreadgroups: Requested %llu total threads per threadgroup exceeds device maximum of %llu",
               (unsigned long long)totalThreads, (unsigned long long)maxThreadsPerGroup);
        return;
    }

    MTLSize threadgroupsPerGrid = MTLSizeMake(gridX, gridY, gridZ);
    MTLSize threadsPerThreadgroup = MTLSizeMake(threadsX, threadsY, threadsZ);

    [encoder_ dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];

    ET_LOG(Debug, "ETMetalKernelFunction::dispatchThreadgroups: Dispatched grid [%llu, %llu, %llu] with threadgroup [%llu, %llu, %llu]",
           (unsigned long long)gridX, (unsigned long long)gridY, (unsigned long long)gridZ,
           (unsigned long long)threadsX, (unsigned long long)threadsY, (unsigned long long)threadsZ);
}

void ETMetalKernelFunction::runCommandBlock(std::function<void(void)> f) {
    // Use dispatch_sync with the stream's serial queue for thread safety and synchronization
    // This matches PyTorch's approach: dispatch_sync_with_rethrow(getCurrentMPSStream()->queue(), ...)
    ETMetalStream* stream = getCurrentMetalStream();
    dispatch_sync_with_rethrow(stream->queue(), ^() {
        @autoreleasepool {
            f();
        }
    });

    ET_LOG(Debug, "ETMetalKernelFunction::runCommandBlock: Executed command block with dispatch_sync");
}

// =======================
// ETMetalStream Implementation
// =======================

ETMetalStream::ETMetalStream()
    : device_(nil), commandQueue_(nil), commandBuffer_(nil), prevCommandBuffer_(nil),
      commandEncoder_(nil), serialQueue_(nullptr), enableCommitAndContinue_(true) {
    @autoreleasepool {
        // Create device and command queue
        device_ = MTLCreateSystemDefaultDevice();
        if (!device_) {
            ET_LOG(Error, "ETMetalStream: Failed to create Metal device");
            return;
        }
        [device_ retain];

        commandQueue_ = [device_ newCommandQueue];
        if (!commandQueue_) {
            ET_LOG(Error, "ETMetalStream: Failed to create Metal command queue");
            return;
        }
        [commandQueue_ retain];

        // Create serial queue for thread safety
        serialQueue_ = dispatch_queue_create("metal gpu stream", nullptr);

        ET_LOG(Debug, "ETMetalStream: Created stream with device %p, queue %p", device_, commandQueue_);
    }
}

ETMetalStream::~ETMetalStream() {
    @autoreleasepool {
        // Synchronize before cleanup
        synchronize(SyncType::COMMIT_AND_WAIT);

        // Clean up command encoder
        if (commandEncoder_) {
            [commandEncoder_ release];
            commandEncoder_ = nil;
        }

        // Clean up command buffers
        if (commandBuffer_) {
            [commandBuffer_ release];
            commandBuffer_ = nil;
        }
        if (prevCommandBuffer_) {
            [prevCommandBuffer_ release];
            prevCommandBuffer_ = nil;
        }

        // Clean up command queue and device
        if (commandQueue_) {
            [commandQueue_ release];
            commandQueue_ = nil;
        }
        if (device_) {
            [device_ release];
            device_ = nil;
        }

        // Clean up serial queue
        if (serialQueue_) {
            dispatch_release(serialQueue_);
            serialQueue_ = nullptr;
        }

        ET_LOG(Debug, "ETMetalStream: Destroyed stream");
    }
}

ETMetalStream* ETMetalStream::getDefaultStream() {
    if (!defaultStream_) {
        defaultStream_ = new ETMetalStream();
    }
    return defaultStream_;
}

// Lazy command buffer creation (use MPSCommandBuffer like PyTorch)
MPSCommandBuffer* ETMetalStream::commandBuffer() {
    if (!commandBuffer_) {
        if (!commandQueue_) {
            ET_LOG(Error, "ETMetalStream::commandBuffer: No command queue available");
            return nil;
        }

        commandBuffer_ = [MPSCommandBuffer commandBufferFromCommandQueue:commandQueue_];
        if (!commandBuffer_) {
            ET_LOG(Error, "ETMetalStream::commandBuffer: Failed to create command buffer");
            return nil;
        }
        [commandBuffer_ retain];

        ET_LOG(Debug, "ETMetalStream::commandBuffer: Created lazy command buffer %p", commandBuffer_);
    }

    return commandBuffer_;
}

// Lazy command encoder creation
id<MTLComputeCommandEncoder> ETMetalStream::commandEncoder() {
    if (!commandEncoder_) {
        MPSCommandBuffer* cmdBuffer = commandBuffer();
        if (!cmdBuffer) {
            ET_LOG(Error, "ETMetalStream::commandEncoder: Failed to get command buffer");
            return nil;
        }

        commandEncoder_ = [cmdBuffer computeCommandEncoder];
        if (!commandEncoder_) {
            ET_LOG(Error, "ETMetalStream::commandEncoder: Failed to create command encoder");
            return nil;
        }
        [commandEncoder_ retain];

        ET_LOG(Debug, "ETMetalStream::commandEncoder: Created lazy command encoder %p", commandEncoder_);
    }

    return commandEncoder_;
}

// Synchronization with SyncType - matches PyTorch's approach (no dispatch_sync here)
void ETMetalStream::synchronize(SyncType syncType) {
    endKernelCoalescing();

    switch (syncType) {
        case SyncType::NONE:
            // Do nothing - no commit
            break;
        case SyncType::COMMIT:
            commit();
            break;
        case SyncType::COMMIT_AND_WAIT:
            commitAndWait();
            break;
        case SyncType::COMMIT_AND_CONTINUE:
            if (enableCommitAndContinue_) {
                commitAndContinue();
            } else {
                ET_LOG(Error, "ETMetalStream::synchronize: CommitAndContinue requested but disabled");
                commit();
            }
            break;
        case SyncType::COMMIT_ADAPTIVE:
            // Simple adaptive policy - could be enhanced with memory pressure detection
            // TODO: Could add memory pressure detection like PyTorch does
            commit();
            break;
    }

    ET_LOG(Debug, "ETMetalStream::synchronize: Completed with SyncType %d", static_cast<int>(syncType));
}

// Encoder coalescing management
void ETMetalStream::endKernelCoalescing() {
    if (commandEncoder_) {
        [commandEncoder_ endEncoding];
        [commandEncoder_ release];
        commandEncoder_ = nil;
        ET_LOG(Debug, "ETMetalStream::endKernelCoalescing: Ended encoder coalescing");
    }
}

// Commit methods
void ETMetalStream::commit() {
    if (!commandBuffer_) {
        ET_LOG(Error, "ETMetalStream::commit: No command buffer to commit");
        return;
    }

    [commandBuffer_ commit];
    ET_LOG(Debug, "ETMetalStream::commit: Committed buffer %p", commandBuffer_);

    [commandBuffer_ release];
    commandBuffer_ = nil;
}

void ETMetalStream::commitAndWait() {
    // Handle previous command buffer first
    if (prevCommandBuffer_) {
        [prevCommandBuffer_ waitUntilCompleted];
        [prevCommandBuffer_ release];
        prevCommandBuffer_ = nil;
    }

    // Handle current command buffer
    if (commandBuffer_) {
        [commandBuffer_ commit];
        [commandBuffer_ waitUntilCompleted];
        [commandBuffer_ release];
        commandBuffer_ = nil;
    }

    ET_LOG(Debug, "ETMetalStream::commitAndWait: Committed and waited for completion");
}

void ETMetalStream::commitAndContinue() {
    if (!commandBuffer_) {
        ET_LOG(Error, "ETMetalStream::commitAndContinue: No command buffer to commit");
        return;
    }

    // Commit buffer and allow immediate reuse for better performance
    [commandBuffer_ commit];
    ET_LOG(Debug, "ETMetalStream::commitAndContinue: Committed buffer %p with continue", commandBuffer_);

    // The buffer handles synchronization internally for commit-and-continue
}

void ETMetalStream::flush() {
    if (commandBuffer_) {
        [commandBuffer_ commit];

        if (!enableCommitAndContinue_) {
            // Keep the command buffer for later waiting if commit-and-continue is disabled
            prevCommandBuffer_ = commandBuffer_;
        } else {
            [commandBuffer_ release];
        }
        commandBuffer_ = nil;

        ET_LOG(Debug, "ETMetalStream::flush: Flushed command buffer");
    }
}

// Memory operations
void ETMetalStream::fill(id<MTLBuffer> buffer, uint8_t value, size_t length, size_t offset, SyncType syncType) {
    if (length == 0) {
        return;
    }

    dispatch_sync(serialQueue_, ^{
        @autoreleasepool {
            endKernelCoalescing();
            id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer() blitCommandEncoder];

            [blitEncoder fillBuffer:buffer range:NSMakeRange(offset, length) value:value];
            [blitEncoder endEncoding];
            synchronize(syncType);

            ET_LOG(Debug, "ETMetalStream::fill: Filled buffer with value %u, length %zu, offset %zu", value, length, offset);
        }
    });
}

void ETMetalStream::copy(id<MTLBuffer> srcBuffer, id<MTLBuffer> dstBuffer, size_t length,
                        size_t srcOffset, size_t dstOffset, SyncType syncType) {

    if (length == 0) {
        return;
    }

    // Check that offsets are within buffer bounds before copying
    if (!srcBuffer || !dstBuffer) {
        ET_LOG(Error, "ETMetalStream::copy: Source or destination buffer is nil");
        return;
    }
    NSUInteger srcBufferLength = [srcBuffer length];
    NSUInteger dstBufferLength = [dstBuffer length];
    if (srcOffset + length > srcBufferLength) {
        ET_LOG(Error, "ETMetalStream::copy: Source offset (%zu) + length (%zu) exceeds source buffer size (%zu)", srcOffset, length, srcBufferLength);
        return;
    }
    if (dstOffset + length > dstBufferLength) {
        ET_LOG(Error, "ETMetalStream::copy: Destination offset (%zu) + length (%zu) exceeds destination buffer size (%zu)", dstOffset, length, dstBufferLength);
        return;
    }

    dispatch_sync(serialQueue_, ^{
        @autoreleasepool {
            endKernelCoalescing();
            id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer() blitCommandEncoder];

            // Handle large copies in chunks
            constexpr size_t max_copy_size = 0x80000000; // 2GB
            size_t bytes_copied = 0;
            size_t bytes_remaining = length;

            while (bytes_remaining > 0) {
                NSUInteger bytes_to_copy = std::min(max_copy_size, bytes_remaining);
                [blitEncoder copyFromBuffer:srcBuffer
                               sourceOffset:(NSUInteger)srcOffset + bytes_copied
                                   toBuffer:dstBuffer
                          destinationOffset:(NSUInteger)dstOffset + bytes_copied
                                       size:bytes_to_copy];
                bytes_copied += bytes_to_copy;
                bytes_remaining -= bytes_to_copy;
            }

            [blitEncoder endEncoding];
            synchronize(syncType);

            ET_LOG(Debug, "ETMetalStream::copy: Copied %zu bytes from offset %zu to offset %zu", length, srcOffset, dstOffset);
        }
    });
}


void ETMetalStream::synchronize() {
    synchronize(SyncType::COMMIT_AND_WAIT);
}

bool ETMetalStream::isEmpty() const {
    return !commandBuffer_ && !commandEncoder_;
}

void ETMetalStream::executeMPSGraph(MPSGraph* mpsGraph, NSDictionary* feeds, NSDictionary* results, SyncType syncType) {
    // Use dispatch_sync_with_rethrow exactly like PyTorch does for MPSGraph execution
    dispatch_sync_with_rethrow(serialQueue_, ^() {
        @autoreleasepool {
            endKernelCoalescing();

            [mpsGraph encodeToCommandBuffer:commandBuffer()
                                      feeds:feeds
                           targetOperations:nil
                          resultsDictionary:results
                        executionDescriptor:nil];
        }
    });
}

// =======================
// Global Storage Management Functions
// =======================

void storeFunctionHandle(ETMetalKernelFunction* raw_function, std::shared_ptr<ETMetalKernelFunction> function_shared_ptr) {
    function_storage[raw_function] = function_shared_ptr;
}

void storeLibraryHandle(ETMetalShaderLibrary* raw_library, std::unique_ptr<ETMetalShaderLibrary> library) {
    library_storage[raw_library] = std::move(library);
}

bool removeFunctionHandle(ETMetalKernelFunction* raw_function) {
    auto it = function_storage.find(raw_function);
    if (it != function_storage.end()) {
        function_storage.erase(it);
        return true;
    }
    return false;
}

bool removeLibraryHandle(ETMetalShaderLibrary* raw_library) {
    auto it = library_storage.find(raw_library);
    if (it != library_storage.end()) {
        library_storage.erase(it);
        return true;
    }
    return false;
}

// =======================
// Global Stream Access Functions
// =======================

ETMetalStream* getCurrentMetalStream() {
    if (!currentStream_) {
        currentStream_ = ETMetalStream::getDefaultStream();
    }
    return currentStream_;
}

void setCurrentMetalStream(ETMetalStream* stream) {
    currentStream_ = stream;
}

// =======================
// Metal Stream Synchronization Functions
// =======================

void synchronize_metal_stream() {
    @autoreleasepool {
        // Use the ETMetalStream for proper synchronization
        ETMetalStream* stream = getCurrentMetalStream();
        stream->synchronize(SyncType::COMMIT_AND_WAIT);

        ET_LOG(Debug, "synchronize_metal_stream: Stream synchronized with COMMIT_AND_WAIT");
    }
}

void synchronize_metal_stream_with_type(int sync_type) {
    @autoreleasepool {
        ETMetalStream* stream = getCurrentMetalStream();
        SyncType syncTypeEnum = static_cast<SyncType>(sync_type);
        stream->synchronize(syncTypeEnum);

        ET_LOG(Debug, "synchronize_metal_stream_with_type: Stream synchronized with SyncType %d", sync_type);
    }
}

} // namespace metal
} // namespace backends
} // namespace executorch
