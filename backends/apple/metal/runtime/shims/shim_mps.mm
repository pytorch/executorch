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

AOTITorchError aoti_torch_mps_addmm_out(
    AtenTensorHandle out,
    AtenTensorHandle self,
    AtenTensorHandle mat1,
    AtenTensorHandle mat2,
    double beta,
    double alpha) {

  ET_LOG(Debug, "aoti_torch_mps_addmm_out: Starting with out=%p, self=%p, mat1=%p, mat2=%p, beta=%f, alpha=%f",
         out, self, mat1, mat2, beta, alpha);

  if (!out || !self || !mat1 || !mat2) {
    ET_LOG(Error, "aoti_torch_mps_addmm_out: null tensor handles");
    return Error::InvalidArgument;
  }

  @autoreleasepool {
    try {
      // Convert AtenTensorHandle to ExecutorTorch tensors
      auto out_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(out);
      auto self_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(self);
      auto mat1_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(mat1);
      auto mat2_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(mat2);

      ET_LOG(Debug, "aoti_torch_mps_addmm_out: Converted tensor handles to ET tensors");

      // For now, just zero out the output tensor to get the right shape
      // TODO: Implement actual matrix multiplication: out = beta * self + alpha * (mat1 @ mat2)

      // Get output data pointer and size
      float* out_data = static_cast<float*>(out_tensor->mutable_data_ptr());
      size_t out_numel = out_tensor->numel();

      if (!out_data) {
        ET_LOG(Error, "aoti_torch_mps_addmm_out: null output data pointer");
        return Error::InvalidArgument;
      }

      // Zero out the output tensor
      std::memset(out_data, 0, out_numel * sizeof(float));

      ET_LOG(Debug, "aoti_torch_mps_addmm_out: Zeroed output tensor with %zu elements", out_numel);
      return Error::Ok;

    } catch (const std::exception& e) {
      ET_LOG(Error, "aoti_torch_mps_addmm_out exception: %s", e.what());
      return Error::Internal;
    } catch (...) {
      ET_LOG(Error, "aoti_torch_mps_addmm_out: unknown exception");
      return Error::Internal;
    }
  }
}

} // extern "C"

} // namespace aoti
} // namespace backends
} // namespace executorch
