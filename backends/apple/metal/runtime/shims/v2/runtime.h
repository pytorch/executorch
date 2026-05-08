/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Infrastructure layer for the v2 AOTI Metal backend.
//
// Wraps metal_v2::MetalStream and exposes:
//   - getMetalStream() / getMetalDevice() / metal_set_flush_interval()
//   - the metal_* C ABI for buffer management used by the AOTI shims
//   - synchronize_metal_stream() drain
//
// No AOTI knowledge lives here; aoti_tensor / aoti_kernel /
// aoti_fallback_op sit on top.

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
}  // namespace executorch::backends::metal_v2

namespace executorch {
namespace backends {
namespace metal {

// Thread-local MetalStream backing all v2 AOTI Metal execution.
metal_v2::MetalStream* getMetalStream();

// Free-function wrappers usable from non-ObjC++ TUs (which can't include
// MetalStream.h transitively due to its <Metal/Metal.h> import).
void metal_set_flush_interval(int dispatches);
MTLDevice_t getMetalDevice();

#ifdef __cplusplus
extern "C" {
#endif

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
bool metal_buffer_nocopy(void* ptr, size_t nbytes);

// snake_case alias for getMetalDevice(). Required by op_convolution.mm
// (compiled from v1 alongside v2 because v2 has no ConvOp yet).
void* get_metal_device();

#ifdef __cplusplus
}
#endif

// C++-mangled (NOT extern "C") to match v1's et_metal.h declaration —
// metal_backend.cpp from v1 calls this via the mangled symbol.
void synchronize_metal_stream();

}  // namespace metal
}  // namespace backends
}  // namespace executorch
