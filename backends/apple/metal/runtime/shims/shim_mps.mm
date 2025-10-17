/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/backends/apple/metal/runtime/shims/shim_mps.h>
#include <executorch/backends/apple/metal/runtime/shims/et_metal.h>
#include <functional>
#include <unordered_map>

namespace executorch {
namespace backends {
namespace metal {

// Declare the global mapping from et_metal.mm
extern std::unordered_map<void*, id<MTLBuffer>> ptr_to_mtl_buffer;

extern "C" {

// MetalShaderLibrary functions
AOTITorchError aoti_torch_mps_create_shader_library(
    const char* metal_shader_source,
    AOTIMetalShaderLibraryHandle* library_handle) {

    if (!metal_shader_source || !library_handle) {
        ET_LOG(Error, "aoti_torch_mps_create_shader_library: null arguments");
        return Error::InvalidArgument;
    }

    @autoreleasepool {
        try {
            auto library = std::make_unique<ETMetalShaderLibrary>(std::string(metal_shader_source));
            auto* raw_library = library.get();

            // Store the unique_ptr to keep the object alive
            storeLibraryHandle(raw_library, std::move(library));

            // Return raw pointer to match existing API
            *library_handle = reinterpret_cast<AOTIMetalShaderLibraryHandle>(raw_library);

            ET_LOG(Debug, "aoti_torch_mps_create_shader_library: Created shader library %p", raw_library);
            return Error::Ok;

        } catch (const std::exception& e) {
            ET_LOG(Error, "aoti_torch_mps_create_shader_library exception: %s", e.what());
            return Error::Internal;
        } catch (...) {
            ET_LOG(Error, "aoti_torch_mps_create_shader_library: unknown exception");
            return Error::Internal;
        }
    }
}

AOTITorchError aoti_torch_mps_delete_shader_library(
    AOTIMetalShaderLibraryHandle library_handle) {

    if (!library_handle) {
        ET_LOG(Error, "aoti_torch_mps_delete_shader_library: null library handle");
        return Error::InvalidArgument;
    }

    try {
        auto* library = reinterpret_cast<ETMetalShaderLibrary*>(library_handle);
        if (removeLibraryHandle(library)) {
            ET_LOG(Debug, "aoti_torch_mps_delete_shader_library: Deleted shader library %p", library);
        } else {
            ET_LOG(Error, "aoti_torch_mps_delete_shader_library: Library not found in storage");
            return Error::InvalidArgument;
        }

        return Error::Ok;

    } catch (const std::exception& e) {
        ET_LOG(Error, "aoti_torch_mps_delete_shader_library exception: %s", e.what());
        return Error::Internal;
    } catch (...) {
        ET_LOG(Error, "aoti_torch_mps_delete_shader_library: unknown exception");
        return Error::Internal;
    }
}

AOTITorchError aoti_torch_mps_get_kernel_function(
    AOTIMetalShaderLibraryHandle library_handle,
    const char* kernel_name,
    AOTIMetalKernelFunctionHandle* function_handle) {

    if (!library_handle || !kernel_name || !function_handle) {
        ET_LOG(Error, "aoti_torch_mps_get_kernel_function: null arguments");
        return Error::InvalidArgument;
    }

    @autoreleasepool {
        try {
            auto* library = reinterpret_cast<ETMetalShaderLibrary*>(library_handle);
            auto function_shared_ptr = library->getKernelFunction(std::string(kernel_name));
            if (!function_shared_ptr) {
                ET_LOG(Error, "aoti_torch_mps_get_kernel_function: Failed to get kernel function '%s'", kernel_name);
                return Error::Internal;
            }

            auto* raw_function = function_shared_ptr.get();

            // Store the shared_ptr to keep the object alive
            storeFunctionHandle(raw_function, function_shared_ptr);

            // Return raw pointer to match existing API
            *function_handle = reinterpret_cast<AOTIMetalKernelFunctionHandle>(raw_function);

            ET_LOG(Debug, "aoti_torch_mps_get_kernel_function: Got kernel function '%s' -> %p", kernel_name, raw_function);
            return Error::Ok;

        } catch (const std::exception& e) {
            ET_LOG(Error, "aoti_torch_mps_get_kernel_function exception: %s", e.what());
            return Error::Internal;
        } catch (...) {
            ET_LOG(Error, "aoti_torch_mps_get_kernel_function: unknown exception");
            return Error::Internal;
        }
    }
}

AOTITorchError aoti_torch_mps_start_encoding(
    AOTIMetalKernelFunctionHandle func) {

    if (!func) {
        ET_LOG(Error, "aoti_torch_mps_start_encoding: null function handle");
        return Error::InvalidArgument;
    }

    @autoreleasepool {
        try {
            auto* function = reinterpret_cast<ETMetalKernelFunction*>(func);
            function->startEncoding();

            ET_LOG(Debug, "aoti_torch_mps_start_encoding: Started encoding for function %p", function);
            return Error::Ok;

        } catch (const std::exception& e) {
            ET_LOG(Error, "aoti_torch_mps_start_encoding exception: %s", e.what());
            return Error::Internal;
        } catch (...) {
            ET_LOG(Error, "aoti_torch_mps_start_encoding: unknown exception");
            return Error::Internal;
        }
    }
}

AOTITorchError aoti_torch_mps_set_arg_tensor(
    AOTIMetalKernelFunctionHandle func,
    unsigned idx,
    AOTITensorHandle tensor) {

    if (!func || !tensor) {
        ET_LOG(Error, "aoti_torch_mps_set_arg_tensor: null function handle or tensor");
        return Error::InvalidArgument;
    }

    @autoreleasepool {
        try {
            auto* function = reinterpret_cast<ETMetalKernelFunction*>(func);
            auto* et_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(tensor);

            function->setArg(idx, *et_tensor);

            ET_LOG(Debug, "aoti_torch_mps_set_arg_tensor: Set tensor argument at index %u", idx);
            return Error::Ok;

        } catch (const std::exception& e) {
            ET_LOG(Error, "aoti_torch_mps_set_arg_tensor exception: %s", e.what());
            return Error::Internal;
        } catch (...) {
            ET_LOG(Error, "aoti_torch_mps_set_arg_tensor: unknown exception");
            return Error::Internal;
        }
    }
}

AOTITorchError aoti_torch_mps_set_arg_int(
    AOTIMetalKernelFunctionHandle func,
    unsigned idx,
    int64_t val) {

    if (!func) {
        ET_LOG(Error, "aoti_torch_mps_set_arg_int: null function handle");
        return Error::InvalidArgument;
    }

    try {
        auto* function = reinterpret_cast<ETMetalKernelFunction*>(func);
        function->setArg(idx, val);

        ET_LOG(Debug, "aoti_torch_mps_set_arg_int: Set int64_t value %lld at index %u", val, idx);
        return Error::Ok;

    } catch (const std::exception& e) {
        ET_LOG(Error, "aoti_torch_mps_set_arg_int exception: %s", e.what());
        return Error::Internal;
    } catch (...) {
        ET_LOG(Error, "aoti_torch_mps_set_arg_int: unknown exception");
        return Error::Internal;
    }
}

// Pure C dispatch functions - single value versions
AOTITorchError aoti_torch_mps_dispatch_single(
    AOTIMetalKernelFunctionHandle func,
    uint64_t length) {

    if (!func) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_single: null function handle");
        return Error::InvalidArgument;
    }

    try {
        auto* function = reinterpret_cast<ETMetalKernelFunction*>(func);
        function->dispatchSingle(length);

        ET_LOG(Debug, "aoti_torch_mps_dispatch_single: Dispatched function %p with length %llu", function, length);
        return Error::Ok;

    } catch (const std::exception& e) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_single exception: %s", e.what());
        return Error::Internal;
    } catch (...) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_single: unknown exception");
        return Error::Internal;
    }
}

AOTITorchError aoti_torch_mps_dispatch_single_with_group_size(
    AOTIMetalKernelFunctionHandle func,
    uint64_t length,
    uint64_t group_size) {

    if (!func) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_single_with_group_size: null function handle");
        return Error::InvalidArgument;
    }

    try {
        auto* function = reinterpret_cast<ETMetalKernelFunction*>(func);
        function->dispatchSingleWithGroupSize(length, group_size);

        ET_LOG(Debug, "aoti_torch_mps_dispatch_single_with_group_size: Dispatched function %p with length %llu, group size %llu", function, length, group_size);
        return Error::Ok;

    } catch (const std::exception& e) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_single_with_group_size exception: %s", e.what());
        return Error::Internal;
    } catch (...) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_single_with_group_size: unknown exception");
        return Error::Internal;
    }
}

// Pure C dispatch functions - array versions
AOTITorchError aoti_torch_mps_dispatch_array(
    AOTIMetalKernelFunctionHandle func,
    const uint64_t* length,
    size_t length_size) {

    if (!func) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_array: null function handle");
        return Error::InvalidArgument;
    }

    if (!length) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_array_with_group_size: null length pointer");
        return Error::InvalidArgument;
    }

    try {
        auto* function = reinterpret_cast<ETMetalKernelFunction*>(func);
        function->dispatchArray(length, length_size);

        ET_LOG(Debug, "aoti_torch_mps_dispatch_array: Dispatched function %p with %zu dimensions", function, length_size);
        return Error::Ok;

    } catch (const std::exception& e) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_array exception: %s", e.what());
        return Error::Internal;
    } catch (...) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_array: unknown exception");
        return Error::Internal;
    }
}

AOTITorchError aoti_torch_mps_dispatch_array_with_group_size(
    AOTIMetalKernelFunctionHandle func,
    const uint64_t* length,
    size_t length_size,
    const uint64_t* group_size,
    size_t group_size_size) {

    if (!func) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_array_with_group_size: null function handle");
        return Error::InvalidArgument;
    }

    if (!length) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_array_with_group_size: null length pointer");
        return Error::InvalidArgument;
    }

    try {
        auto* function = reinterpret_cast<ETMetalKernelFunction*>(func);
        function->dispatchArrayWithGroupSize(length, length_size, group_size, group_size_size);

        ET_LOG(Debug, "aoti_torch_mps_dispatch_array_with_group_size: Dispatched function %p with %zu dimensions", function, length_size);
        return Error::Ok;

    } catch (const std::exception& e) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_array_with_group_size exception: %s", e.what());
        return Error::Internal;
    } catch (...) {
        ET_LOG(Error, "aoti_torch_mps_dispatch_array_with_group_size: unknown exception");
        return Error::Internal;
    }
}

AOTITorchError aoti_torch_mps_malloc(void** buffer, size_t num_bytes) {
    if (num_bytes == 0) {
        *buffer = nullptr;
        return Error::Ok;
    }

    if (!buffer) {
        ET_LOG(Error, "aoti_torch_mps_malloc: null buffer pointer");
        return Error::InvalidArgument;
    }

    @autoreleasepool {
        try {
            id<MTLDevice> device = get_metal_device();
            if (!device) {
                ET_LOG(Error, "aoti_torch_mps_malloc: Failed to get Metal device");
                return Error::Internal;
            }

            id<MTLBuffer> metal_buffer = [device newBufferWithLength:num_bytes
                                                             options:MTLResourceCPUCacheModeWriteCombined | MTLResourceStorageModeShared];
            if (!metal_buffer) {
                ET_LOG(Error, "aoti_torch_mps_malloc: Failed to allocate Metal buffer of size %zu", num_bytes);
                return Error::Internal;
            }

            // FIX: Return contents pointer, not buffer object
            void* contents_ptr = [metal_buffer contents];
            ptr_to_mtl_buffer[contents_ptr] = metal_buffer;  // Map contents to buffer
            *buffer = contents_ptr;  // Return contents pointer

            ET_LOG(Debug, "aoti_torch_mps_malloc: Allocated Metal buffer %p with contents %p of size %zu",
                   metal_buffer, contents_ptr, num_bytes);
            return Error::Ok;

        } catch (const std::exception& e) {
            ET_LOG(Error, "aoti_torch_mps_malloc exception: %s", e.what());
            return Error::Internal;
        } catch (...) {
            ET_LOG(Error, "aoti_torch_mps_malloc: unknown exception");
            return Error::Internal;
        }
    }
}

AOTITorchError aoti_torch_mps_free(void* ptr) {
    if (!ptr) {
        return Error::Ok;  // Nothing to free
    }

    @autoreleasepool {
        try {
            // FIX: ptr is now the contents pointer, not the buffer object
            // Look up the buffer from the mapping and clean up
            auto it = ptr_to_mtl_buffer.find(ptr);
            if (it != ptr_to_mtl_buffer.end()) {
                id<MTLBuffer> metal_buffer = it->second;
                [metal_buffer release];
                ptr_to_mtl_buffer.erase(it);
                ET_LOG(Debug, "aoti_torch_mps_free: Freed Metal buffer for contents %p", ptr);
            } else {
                ET_LOG(Error, "aoti_torch_mps_free: Buffer not found for contents pointer %p", ptr);
                return Error::InvalidArgument;
            }

            return Error::Ok;

        } catch (const std::exception& e) {
            ET_LOG(Error, "aoti_torch_mps_free exception: %s", e.what());
            return Error::Internal;
        } catch (...) {
            ET_LOG(Error, "aoti_torch_mps_free: unknown exception");
            return Error::Internal;
        }
    }
}

AOTITorchError aoti_torch_mps_memcpy(
    void* buffer,
    size_t constant_offset,
    size_t bytes_read,
    size_t data_size,
    uint8_t* constants_start) {

    if (!buffer || !constants_start) {
        ET_LOG(Error, "aoti_torch_mps_memcpy: null buffer or constants_start");
        return Error::InvalidArgument;
    }

    @autoreleasepool {
        try {
            // FIX: buffer is now the contents pointer, not the buffer object
            auto buffer_pointer = static_cast<uint8_t*>(buffer);

            memcpy(buffer_pointer + constant_offset, constants_start + bytes_read, data_size);

            id<MTLDevice> device = get_metal_device();
            if (!device) {
                ET_LOG(Error, "aoti_torch_mps_memcpy: Failed to get Metal device");
                return Error::Internal;
            }
            id<MTLBuffer> subBuffer = [device newBufferWithBytesNoCopy:buffer_pointer + constant_offset
                                                                length:data_size
                                                               options:MTLResourceCPUCacheModeWriteCombined | MTLResourceStorageModeShared
                                                           deallocator:nil];

            if (constant_offset != 0) {
                ptr_to_mtl_buffer[buffer_pointer + constant_offset] = subBuffer;  // Map contents to buffer
            }

            ET_LOG(Debug, "aoti_torch_mps_memcpy: Copied %zu bytes from offset %zu to buffer offset %zu",
                   data_size, bytes_read, constant_offset);
            return Error::Ok;

        } catch (const std::exception& e) {
            ET_LOG(Error, "aoti_torch_mps_memcpy exception: %s", e.what());
            return Error::Internal;
        } catch (...) {
            ET_LOG(Error, "aoti_torch_mps_memcpy: unknown exception");
            return Error::Internal;
        }
    }
}

AOTITorchError aoti_torch_mps_copy_buffer(
    void* src_buffer,
    void* dst_buffer,
    size_t data_size,
    size_t src_offset,
    size_t dst_offset) {

    if (!src_buffer || !dst_buffer) {
        ET_LOG(Error, "aoti_torch_mps_copy_buffer: null buffer");
        return Error::InvalidArgument;
    }

    @autoreleasepool {
        try {
            auto src_mtl_buffer = (id<MTLBuffer>)src_buffer;
            auto dst_mtl_buffer = (id<MTLBuffer>)dst_buffer;

            uint8_t* src_contents = static_cast<uint8_t*>([src_mtl_buffer contents]);
            uint8_t* dst_contents = static_cast<uint8_t*>([dst_mtl_buffer contents]);

            if (!src_contents || !dst_contents) {
                ET_LOG(Error, "aoti_torch_mps_copy_buffer: Failed to get buffer contents");
                return Error::Internal;
            }

            memcpy(dst_contents + dst_offset, src_contents + src_offset, data_size);

            ET_LOG(Debug, "aoti_torch_mps_copy_buffer: Copied %zu bytes from src+%zu to dst+%zu",
                   data_size, src_offset, dst_offset);
            return Error::Ok;

        } catch (const std::exception& e) {
            ET_LOG(Error, "aoti_torch_mps_copy_buffer exception: %s", e.what());
            return Error::Internal;
        } catch (...) {
            ET_LOG(Error, "aoti_torch_mps_copy_buffer: unknown exception");
            return Error::Internal;
        }
    }
}

// Shared callback function for std::function trampoline
void aoti_torch_mps_shared_callback(
    AOTIMetalKernelFunctionHandle func,
    void* user_data) {
    ET_LOG(Debug, "aoti_torch_mps_shared_callback: Called with func=%p, user_data=%p", func, user_data);

    auto* function_wrapper = static_cast<std::function<void(AOTIMetalKernelFunctionHandle)>*>(user_data);
    if (function_wrapper) {
        ET_LOG(Debug, "aoti_torch_mps_shared_callback: Calling function wrapper");
        (*function_wrapper)(func);
        ET_LOG(Debug, "aoti_torch_mps_shared_callback: Function wrapper completed");
    } else {
        ET_LOG(Error, "aoti_torch_mps_shared_callback: null function wrapper");
    }
}

// Pure C version using function pointer and user data for trampoline pattern
AOTITorchError aoti_torch_mps_run_command_block(
    AOTIMetalKernelFunctionHandle func,
    aoti_torch_mps_command_block_callback_t callback,
    void* user_data) {

    if (!func) {
        ET_LOG(Error, "aoti_torch_mps_run_command_block: null function handle");
        return Error::InvalidArgument;
    }

    if (!callback) {
        ET_LOG(Error, "aoti_torch_mps_run_command_block: null callback");
        return Error::InvalidArgument;
    }

    ET_LOG(Debug, "aoti_torch_mps_run_command_block: Starting command block for function %p, callback %p, user_data %p",
           func, callback, user_data);

    try {
        auto* function = reinterpret_cast<ETMetalKernelFunction*>(func);
        function->runCommandBlock([callback, func, user_data]() {
            ET_LOG(Debug, "aoti_torch_mps_run_command_block: Inside lambda, calling callback");
            callback(func, user_data);
            ET_LOG(Debug, "aoti_torch_mps_run_command_block: Callback completed");
        });

        ET_LOG(Debug, "aoti_torch_mps_run_command_block: Executed command block for function %p", function);
        return Error::Ok;

    } catch (const std::exception& e) {
        ET_LOG(Error, "aoti_torch_mps_run_command_block exception: %s", e.what());
        return Error::Internal;
    } catch (...) {
        ET_LOG(Error, "aoti_torch_mps_run_command_block: unknown exception");
        return Error::Internal;
    }
}

} // extern "C"


} // namespace metal
} // namespace backends
} // namespace executorch
