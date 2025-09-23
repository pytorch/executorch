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
  ET_LOG(Error, "aoti_torch_mps_addmm_out: Starting with out=%p, self=%p, mat1=%p, mat2=%p, beta=%f, alpha=%f",
         out, self, mat1, mat2, beta, alpha);

  return Error:Ok;

  // if (!out || !self || !mat1 || !mat2) {
  //   ET_LOG(Error, "aoti_torch_mps_addmm_out: null tensor handles");
  //   return Error::InvalidArgument;
  // }

  // @autoreleasepool {
  //   try {
  //     // Convert AtenTensorHandle to ExecutorTorch tensors
  //     auto out_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(out);
  //     auto self_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(self);
  //     auto mat1_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(mat1);
  //     auto mat2_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(mat2);

  //     ET_LOG(Debug, "aoti_torch_mps_addmm_out: Converted tensor handles to ET tensors");

  //     // Verify tensor dimensions for matrix multiplication
  //     if (mat1_tensor->dim() != 2 || mat2_tensor->dim() != 2) {
  //       ET_LOG(Error, "aoti_torch_mps_addmm_out: mat1 and mat2 must be 2D tensors, got dims %d and %d",
  //              mat1_tensor->dim(), mat2_tensor->dim());
  //       return Error::InvalidArgument;
  //     }

  //     // Get tensor sizes
  //     auto mat1_sizes = mat1_tensor->sizes();
  //     auto mat2_sizes = mat2_tensor->sizes();

  //     ET_LOG(Debug, "aoti_torch_mps_addmm_out: mat1 size [%d, %d], mat2 size [%d, %d]",
  //            (int)mat1_sizes[0], (int)mat1_sizes[1],
  //            (int)mat2_sizes[0], (int)mat2_sizes[1]);

  //     // Verify matrix multiplication dimensions
  //     if (mat1_sizes[1] != mat2_sizes[0]) {
  //       ET_LOG(Error, "aoti_torch_mps_addmm_out: matrix dimension mismatch, mat1[%d,%d] x mat2[%d,%d]",
  //              (int)mat1_sizes[0], (int)mat1_sizes[1], (int)mat2_sizes[0], (int)mat2_sizes[1]);
  //       return Error::InvalidArgument;
  //     }

  //     // Get Metal buffers for each tensor
  //     void* out_data = out_tensor->mutable_data_ptr();
  //     void* self_data = self_tensor->mutable_data_ptr();
  //     void* mat1_data = mat1_tensor->mutable_data_ptr();
  //     void* mat2_data = mat2_tensor->mutable_data_ptr();

  //     // Get MTLBuffers from our buffer mapping
  //     auto out_it = ptr_to_mtl_buffer.find(out_data);
  //     auto self_it = ptr_to_mtl_buffer.find(self_data);
  //     auto mat1_it = ptr_to_mtl_buffer.find(mat1_data);
  //     auto mat2_it = ptr_to_mtl_buffer.find(mat2_data);

  //     if (out_it == ptr_to_mtl_buffer.end() ||
  //         self_it == ptr_to_mtl_buffer.end() ||
  //         mat1_it == ptr_to_mtl_buffer.end() ||
  //         mat2_it == ptr_to_mtl_buffer.end()) {
  //       ET_LOG(Error, "aoti_torch_mps_addmm_out: One or more Metal buffers not found in mapping");
  //       return Error::Internal;
  //     }

  //     id<MTLBuffer> out_buffer = out_it->second;
  //     id<MTLBuffer> self_buffer = self_it->second;
  //     id<MTLBuffer> mat1_buffer = mat1_it->second;
  //     id<MTLBuffer> mat2_buffer = mat2_it->second;

  //     // Get the Metal device and command queue
  //     id<MTLDevice> device = get_metal_device();
  //     if (!device) {
  //       ET_LOG(Error, "aoti_torch_mps_addmm_out: Failed to get Metal device");
  //       return Error::Internal;
  //     }

  //     id<MTLCommandQueue> commandQueue = get_metal_command_queue();
  //     if (!commandQueue) {
  //       ET_LOG(Error, "aoti_torch_mps_addmm_out: Failed to get Metal command queue");
  //       return Error::Internal;
  //     }

  //     id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
  //     if (!commandBuffer) {
  //       ET_LOG(Error, "aoti_torch_mps_addmm_out: Failed to create command buffer");
  //       return Error::Internal;
  //     }

  //     // Set up matrix descriptors for MetalPerformanceShaders
  //     int M = mat1_sizes[0];  // Rows of mat1 and output
  //     int K = mat1_sizes[1];  // Cols of mat1, rows of mat2
  //     int N = mat2_sizes[1];  // Cols of mat2 and output

  //     // Create matrix descriptors
  //     MPSMatrixDescriptor* mat1Desc = [MPSMatrixDescriptor matrixDescriptorWithRows:M
  //                                                                           columns:K
  //                                                                          rowBytes:K * sizeof(float)
  //                                                                          dataType:MPSDataTypeFloat32];

  //     MPSMatrixDescriptor* mat2Desc = [MPSMatrixDescriptor matrixDescriptorWithRows:K
  //                                                                           columns:N
  //                                                                          rowBytes:N * sizeof(float)
  //                                                                          dataType:MPSDataTypeFloat32];

  //     MPSMatrixDescriptor* outDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:M
  //                                                                          columns:N
  //                                                                         rowBytes:N * sizeof(float)
  //                                                                         dataType:MPSDataTypeFloat32];

  //     // Create MPS matrices
  //     MPSMatrix* mat1Matrix = [[MPSMatrix alloc] initWithBuffer:mat1_buffer descriptor:mat1Desc];
  //     MPSMatrix* mat2Matrix = [[MPSMatrix alloc] initWithBuffer:mat2_buffer descriptor:mat2Desc];
  //     MPSMatrix* outMatrix = [[MPSMatrix alloc] initWithBuffer:out_buffer descriptor:outDesc];

  //     // The operation is: out = beta * self + alpha * (mat1 @ mat2)

  //     // Step 1: Copy self to out and scale by beta
  //     if (beta != 0.0) {
  //       // Handle self tensor (could be scalar or matrix)
  //       if (self_tensor->numel() == 1) {
  //         // Scalar case: fill output with beta * scalar
  //         MPSMatrixDescriptor* selfDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:1
  //                                                                               columns:1
  //                                                                              rowBytes:sizeof(float)
  //                                                                              dataType:MPSDataTypeFloat32];
  //         MPSMatrix* selfMatrix = [[MPSMatrix alloc] initWithBuffer:self_buffer descriptor:selfDesc];

  //         // Use MPSMatrixUnaryKernel to broadcast and scale
  //         MPSMatrixUnaryKernel* scaleKernel = [[MPSMatrixUnaryKernel alloc] initWithDevice:device
  //                                                                               sourceRows:1
  //                                                                            sourceColumns:1
  //                                                                               resultRows:M
  //                                                                            resultColumns:N];
  //         scaleKernel.alpha = beta;
  //         scaleKernel.beta = 0.0;
  //         [scaleKernel encodeToCommandBuffer:commandBuffer
  //                               sourceMatrix:selfMatrix
  //                               resultMatrix:outMatrix];
  //       } else {
  //         // Matrix case: element-wise scale
  //         MPSMatrix* selfMatrix = [[MPSMatrix alloc] initWithBuffer:self_buffer descriptor:outDesc];
  //         MPSMatrixUnaryKernel* scaleKernel = [[MPSMatrixUnaryKernel alloc] initWithDevice:device
  //                                                                               sourceRows:M
  //                                                                            sourceColumns:N
  //                                                                               resultRows:M
  //                                                                            resultColumns:N];
  //         scaleKernel.alpha = beta;
  //         scaleKernel.beta = 0.0;
  //         [scaleKernel encodeToCommandBuffer:commandBuffer
  //                               sourceMatrix:selfMatrix
  //                               resultMatrix:outMatrix];
  //       }
  //     } else {
  //       // Zero the output if beta is 0
  //       id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
  //       [blitEncoder fillBuffer:out_buffer range:NSMakeRange(0, M * N * sizeof(float)) value:0];
  //       [blitEncoder endEncoding];
  //     }

  //     // Step 2: Add alpha * (mat1 @ mat2) to the result
  //     if (alpha != 0.0) {
  //       MPSMatrixMultiplication* matmul = [[MPSMatrixMultiplication alloc] initWithDevice:device
  //                                                                            transposeLeft:NO
  //                                                                           transposeRight:NO
  //                                                                              resultRows:M
  //                                                                           resultColumns:N
  //                                                                        interiorColumns:K
  //                                                                                  alpha:alpha
  //                                                                                   beta:1.0]; // beta=1.0 to add to existing output

  //       [matmul encodeToCommandBuffer:commandBuffer
  //                          leftMatrix:mat1Matrix
  //                         rightMatrix:mat2Matrix
  //                        resultMatrix:outMatrix];
  //     }

  //     // Execute the Metal operations
  //     [commandBuffer commit];
  //     [commandBuffer waitUntilCompleted];

  //     if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
  //       ET_LOG(Error, "aoti_torch_mps_addmm_out: Metal command buffer failed with status %ld", (long)commandBuffer.status);
  //       return Error::Internal;
  //     }

  //     ET_LOG(Debug, "aoti_torch_mps_addmm_out: Metal operations completed successfully");
  //     return Error::Ok;

  //   } catch (const std::exception& e) {
  //     ET_LOG(Error, "aoti_torch_mps_addmm_out exception: %s", e.what());
  //     return Error::Internal;
  //   } catch (...) {
  //     ET_LOG(Error, "aoti_torch_mps_addmm_out: unknown exception");
  //     return Error::Internal;
  //   }
  // }
}

} // extern "C"

} // namespace aoti
} // namespace backends
} // namespace executorch
