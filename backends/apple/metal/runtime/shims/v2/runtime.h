/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Pure infrastructure layer for the v2 AOTI Metal backend.
//
// Wraps MetalStream (from portable/runtime/metal_v2/) and exposes:
//   - getMetalStream() / getMetalDevice() / metal_set_flush_interval()
//   - the "metal_*" C ABI for buffer management used by the AOTI shims
//   - synchronize_metal_stream() drain
//
// No AOTI knowledge here. The aoti_* shims (aoti_tensor, aoti_kernel,
// aoti_ops) sit on top of this file.

#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __OBJC__
#import <Metal/Metal.h>
typedef id<MTLDevice> MTLDevice_t;
#else
typedef void* MTLDevice_t;
#endif

namespace executorch::backends::metal_v2 {
class MetalStream;
} // namespace executorch::backends::metal_v2

namespace executorch {
namespace backends {
namespace metal {

// The single MetalStream backing all v2 AOTI Metal execution (thread-local).
metal_v2::MetalStream* getMetalStream();

// Free-function wrappers that .cpp files can call without pulling in
// MetalStream.h (which transitively imports Metal/Metal.h and won't
// compile in non-ObjC++ translation units).
void metal_set_flush_interval(int dispatches);

MTLDevice_t getMetalDevice();

#ifdef __cplusplus
extern "C" {
#endif

void* metal_allocate_buffer(long bytes);
// Like metal_allocate_buffer but does NOT register the returned pointer as
// a device pointer in g_device_ptrs. metal_is_device_pointer() will return
// false for the returned pointer. Used by aoti_torch_mps_malloc to match
// the pre-reorg behavior where mps_malloc'd buffers weren't tracked.
void* metal_allocate_buffer_untracked(long bytes);
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
// Like metal_buffer_nocopy but does NOT add the pointer to the
// device-pointer tracking set. Used by aoti_torch_mps_memcpy to register
// constant sub-regions without affecting metal_is_device_pointer().
bool metal_register_external_buffer_only(void* ptr, size_t nbytes);
void synchronize_metal_stream();

#ifdef __cplusplus
}
#endif

} // namespace metal
} // namespace backends
} // namespace executorch
