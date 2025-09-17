/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef AOTI_METAL

#ifdef __cplusplus
extern "C" {
#endif

// Metal helper functions - these are implemented in Objective-C++
void metal_init_if_needed();
void* metal_allocate_buffer(long bytes);
void metal_cleanup_resources();

// Memory management functions for Metal
bool metal_is_device_pointer(void* ptr);
int metal_copy_memory(void* dst, const void* src, size_t nbytes, bool src_is_device, bool dst_is_device);

#ifdef __cplusplus
}

// C++ only - expose the Metal buffer mapping for MPS shim
#ifdef __OBJC__
#import <Metal/Metal.h>
#include <unordered_map>
extern std::unordered_map<void*, id<MTLBuffer>> ptr_to_mtl_buffer;
#endif

#endif

#endif // AOTI_METAL
