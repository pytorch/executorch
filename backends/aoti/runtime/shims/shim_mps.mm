/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef AOTI_METAL
#ifdef __APPLE__

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include "shim_mps.h"
#include "metal_helper.h"
#include "utils.h"

namespace executorch {
namespace backends {
namespace aoti {

// We need to match PyTorch's MetalKernelFunction structure to extract the encoder
// This is based on PyTorch's ATen/native/mps/MetalShaderLibrary.h
namespace {
  // Match the actual PyTorch MetalKernelFunction structure
  // From MetalShaderLibrary.h: cps, func, encoder (in that order)
  struct MetalKernelFunctionShim {
    id<MTLComputePipelineState> cps;    // First member
    id<MTLFunction> func;               // Second member
    id<MTLComputeCommandEncoder> encoder;  // Third member (what we need)
  };
}

extern "C" {

AOTITorchError aoti_torch_mps_set_arg_tensor(
    AOTIMetalKernelFunctionHandle func,
    unsigned idx,
    AtenTensorHandle tensor) {

  if (!func || !tensor) {
    ET_LOG(Error, "aoti_torch_mps_set_arg_tensor: null function handle or tensor");
    return Error::InvalidArgument;
  }

  @autoreleasepool {
    try {
      ET_LOG(Debug, "aoti_torch_mps_set_arg_tensor: Starting with func=%p, idx=%u, tensor=%p", func, idx, tensor);

      // Cast the opaque handle to our shim structure to access the encoder
      auto kernelFunc = reinterpret_cast<MetalKernelFunctionShim*>(func);
      ET_LOG(Debug, "aoti_torch_mps_set_arg_tensor: Cast to kernelFunc=%p", kernelFunc);

      id<MTLComputeCommandEncoder> encoder = kernelFunc->encoder;
      ET_LOG(Debug, "aoti_torch_mps_set_arg_tensor: Retrieved encoder=%p", encoder);

      if (!encoder) {
        ET_LOG(Error, "aoti_torch_mps_set_arg_tensor: null encoder");
        return Error::InvalidArgument;
      }

      // Convert the AtenTensorHandle to our ExecutorTorch tensor
      // In our case, AtenTensorHandle is just a pointer to our ExecutorTorch tensor
      auto et_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(tensor);
      ET_LOG(Debug, "aoti_torch_mps_set_arg_tensor: Cast to et_tensor=%p", et_tensor);

      // Get the data pointer
      void* data_ptr = et_tensor->mutable_data_ptr();
      ET_LOG(Debug, "aoti_torch_mps_set_arg_tensor: Retrieved data_ptr=%p", data_ptr);

      if (!data_ptr) {
        ET_LOG(Error, "aoti_torch_mps_set_arg_tensor: null data pointer");
        return Error::InvalidArgument;
      }

      // Check if this is a Metal device pointer using our existing helper
      bool is_metal = metal_is_device_pointer(data_ptr);
      ET_LOG(Debug, "aoti_torch_mps_set_arg_tensor: is_metal=%d", is_metal);

      if (is_metal) {
        // This is a Metal tensor - get the MTLBuffer from our mapping
        // Note: ptr_to_mtl_buffer is declared in metal_helper.h and defined in metal_helper.mm
        auto it = ptr_to_mtl_buffer.find(data_ptr);

        if (it == ptr_to_mtl_buffer.end()) {
          ET_LOG(Error, "aoti_torch_mps_set_arg_tensor: Metal pointer not found in buffer mapping");
          return Error::Internal;
        }

        id<MTLBuffer> mtlBuffer = it->second;
        ET_LOG(Debug, "aoti_torch_mps_set_arg_tensor: Retrieved mtlBuffer=%p", mtlBuffer);

        if (!mtlBuffer) {
          ET_LOG(Error, "aoti_torch_mps_set_arg_tensor: null MTLBuffer");
          return Error::Internal;
        }

        // Set the Metal buffer directly on the encoder
        // ExecutorTorch tensors don't have storage_offset, so we assume offset 0
        // This is fine because ExecutorTorch tensors are typically not views
        size_t offset = 0;
        ET_LOG(Debug, "aoti_torch_mps_set_arg_tensor: About to call setBuffer with idx=%u, offset=%zu", idx, offset);

        [encoder setBuffer:mtlBuffer offset:offset atIndex:idx];

        // Also log the buffer contents for debugging (first few bytes)
        void* bufferContents = [mtlBuffer contents];
        if (bufferContents) {
          float* floatData = (float*)bufferContents;
          ET_LOG(Debug, "aoti_torch_mps_set_arg_tensor: Buffer contents at idx %u: [%.3f, %.3f, %.3f, ...]",
                 idx, floatData[0], floatData[1], floatData[2]);
        }

        ET_LOG(Debug, "aoti_torch_mps_set_arg_tensor: Successfully set Metal buffer at index %u with offset %zu",
               idx, offset);

      } else {
        // This is a CPU tensor - handle as bytes
        int dims = et_tensor->dim();
        ET_LOG(Debug, "aoti_torch_mps_set_arg_tensor: CPU tensor with dims=%d", dims);

        if (dims != 0) {
          ET_LOG(Error, "aoti_torch_mps_set_arg_tensor: CPU tensor must be scalar (0-dim)");
          return Error::InvalidArgument;
        }

        // For CPU scalars, set as bytes
        size_t element_size = et_tensor->element_size();
        ET_LOG(Debug, "aoti_torch_mps_set_arg_tensor: About to call setBytes with idx=%u, element_size=%zu", idx, element_size);

        [encoder setBytes:data_ptr length:element_size atIndex:idx];

        ET_LOG(Debug, "aoti_torch_mps_set_arg_tensor: Successfully set CPU scalar at index %u with size %zu",
               idx, element_size);
      }

      ET_LOG(Debug, "aoti_torch_mps_set_arg_tensor: Completed successfully");
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

  @autoreleasepool {
    try {
      // Cast the opaque handle to our shim structure to access the encoder
      auto kernelFunc = reinterpret_cast<MetalKernelFunctionShim*>(func);
      id<MTLComputeCommandEncoder> encoder = kernelFunc->encoder;

      if (!encoder) {
        ET_LOG(Error, "aoti_torch_mps_set_arg_int: null encoder");
        return Error::InvalidArgument;
      }

      // Set the integer value as bytes
      [encoder setBytes:&val length:sizeof(int64_t) atIndex:idx];

      ET_LOG(Debug, "aoti_torch_mps_set_arg_int: set int64_t value %lld at index %u", val, idx);

      return Error::Ok;

    } catch (const std::exception& e) {
      ET_LOG(Error, "aoti_torch_mps_set_arg_int exception: %s", e.what());
      return Error::Internal;
    } catch (...) {
      ET_LOG(Error, "aoti_torch_mps_set_arg_int: unknown exception");
      return Error::Internal;
    }
  }
}

} // extern "C"

} // namespace aoti
} // namespace backends
} // namespace executorch

#else
// Stub implementations for non-Apple platforms with AOTI_METAL defined

namespace executorch {
namespace backends {
namespace aoti {

extern "C" {

AOTITorchError aoti_torch_mps_set_arg_tensor(
    AOTIMetalKernelFunctionHandle func,
    unsigned idx,
    AtenTensorHandle tensor) {
  ET_LOG(Error, "MPS support requested but not available on this platform");
  return Error::NotImplemented;
}

AOTITorchError aoti_torch_mps_set_arg_int(
    AOTIMetalKernelFunctionHandle func,
    unsigned idx,
    int64_t val) {
  ET_LOG(Error, "MPS support requested but not available on this platform");
  return Error::NotImplemented;
}

} // extern "C"

} // namespace aoti
} // namespace backends
} // namespace executorch

#endif // __APPLE__
#endif // AOTI_METAL
