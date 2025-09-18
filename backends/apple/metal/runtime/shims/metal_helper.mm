/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <executorch/runtime/platform/log.h>
#include <unordered_map>
#include "metal_helper.h"

// Metal-specific globals
namespace {
  id<MTLDevice> metalDevice = nil;
  id<MTLCommandQueue> metalCommandQueue = nil;
}

// Make this globally accessible for the MPS shim
std::unordered_map<void*, id<MTLBuffer>> ptr_to_mtl_buffer;

extern "C" {

void metal_init_if_needed() {
  if (!metalDevice) {
    @autoreleasepool {
      metalDevice = MTLCreateSystemDefaultDevice();
      if (!metalDevice) {
        ET_LOG(Error, "Failed to create Metal device");
        return;
      }

      metalCommandQueue = [metalDevice newCommandQueue];
      if (!metalCommandQueue) {
        ET_LOG(Error, "Failed to create Metal command queue");
        return;
      }
      ET_LOG(Info, "Metal initialized successfully");
    }
  }
}

void* metal_allocate_buffer(long bytes) {
  metal_init_if_needed();
  if (!metalDevice) {
    ET_LOG(Error, "Failed to initialize Metal device");
    return nullptr;
  }

  @autoreleasepool {
    // Create a Metal buffer
    id<MTLBuffer> buffer = [metalDevice newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!buffer) {
      ET_LOG(Error, "Failed to allocate %ld bytes on Metal device", bytes);
      return nullptr;
    }

    // Store the Metal buffer pointer - this is accessible from CPU
    void* ptr = [buffer contents];

    // Store the mapping from pointer to Metal buffer for cleanup
    ptr_to_mtl_buffer[ptr] = buffer;

    ET_LOG(Debug, "Allocated %ld bytes on Metal device", bytes);
    return ptr;
  }
}

void metal_cleanup_resources() {
  if (!ptr_to_mtl_buffer.empty()) {
    @autoreleasepool {
      for (auto& pair : ptr_to_mtl_buffer) {
        // Release all MTLBuffer references
        pair.second = nil;
      }
      ptr_to_mtl_buffer.clear();
    }
  }

  // Release Metal objects
  metalCommandQueue = nil;
  metalDevice = nil;
}

bool metal_is_device_pointer(void* ptr) {
  // Check if pointer is associated with a Metal buffer
  return ptr_to_mtl_buffer.find(ptr) != ptr_to_mtl_buffer.end();
}

int metal_copy_memory(void* dst, const void* src, size_t nbytes, bool src_is_device, bool dst_is_device) {
  // For Metal (MPS), since we use MTLResourceStorageModeShared,
  // both device and host share the same memory space.
  // This means we can use simple memcpy for all cases.

  if (!src || !dst || nbytes == 0) {
    ET_LOG(Error, "Metal copy: Invalid parameters");
    return -1;
  }

  // Debug logging to trace the copy operation
  float* src_float = (float*)src;
  float* dst_float = (float*)dst;
  ET_LOG(Debug, "Metal copy: src=%p [%.3f, %.3f, %.3f, ...] -> dst=%p (before copy)",
         src, src_float[0], src_float[1], src_float[2], dst);

  @autoreleasepool {
    // Since Metal buffers use shared storage mode, we can directly copy
    std::memcpy(dst, src, nbytes);

    // Log destination after copy
    ET_LOG(Debug, "Metal copy: dst=%p [%.3f, %.3f, %.3f, ...] (after memcpy)",
           dst, dst_float[0], dst_float[1], dst_float[2]);

    // If either source or destination is a Metal buffer, we may need to synchronize
    if (src_is_device || dst_is_device) {
      // For Metal, we don't need explicit synchronization for shared memory
      // but we can add a command buffer commit if needed for GPU operations
      if (metalCommandQueue) {
        id<MTLCommandBuffer> commandBuffer = [metalCommandQueue commandBuffer];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Log destination after synchronization
        ET_LOG(Debug, "Metal copy: dst=%p [%.3f, %.3f, %.3f, ...] (after sync)",
               dst, dst_float[0], dst_float[1], dst_float[2]);
      }
    }
  }

  ET_LOG(Debug, "Metal memory copy completed: %zu bytes", nbytes);
  return 0;
}

} // extern "C"
