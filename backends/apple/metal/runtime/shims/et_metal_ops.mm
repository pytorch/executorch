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
#include "et_metal_ops.h"
#include "et_metal.h"
#include "utils.h"
#include "memory.h"
#include <functional>
#include <unordered_map>

namespace executorch {
namespace backends {
namespace aoti {

// Forward declaration of dispatch_sync_with_rethrow from et_metal.mm
void dispatch_sync_with_rethrow(dispatch_queue_t queue, void (^block)());

// Declare the global mapping from et_metal.mm
extern std::unordered_map<void*, id<MTLBuffer>> ptr_to_mtl_buffer;

extern "C" {

AOTITorchError aoti_torch_mps_mm_out(
    AOTITensorHandle out,
    AOTITensorHandle self,
    AOTITensorHandle mat2) {
  ET_LOG(Debug, "aoti_torch_mps_mm_out: Starting with out=%p, self=%p, mat2=%p",
         out, self, mat2);

  if (!out || !self || !mat2) {
    ET_LOG(Error, "aoti_torch_mps_mm_out: null tensor handles");
    return Error::InvalidArgument;
  }

  // Use the same dispatch pattern as aoti_torch_mps_run_command_block for consistent synchronization
  ETMetalStream* stream = getCurrentMetalStream();
  if (!stream) {
    ET_LOG(Error, "aoti_torch_mps_mm_out: Failed to get current Metal stream");
    return Error::Internal;
  }

  try {
    // Use dispatch_sync_with_rethrow to match custom kernel synchronization behavior
    dispatch_sync_with_rethrow(stream->queue(), ^() {
      @autoreleasepool {
        // Convert AOTITensorHandle to ExecutorTorch tensors
        auto out_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(out);
        auto self_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(self);
        auto mat2_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(mat2);

        ET_LOG(Debug, "aoti_torch_mps_mm_out: Converted tensor handles to ET tensors");

        // Validate tensor dimensions
        if (self_tensor->dim() != 2 || mat2_tensor->dim() != 2) {
          std::string error_msg = "aoti_torch_mps_mm_out: tensors must be 2-D, got " +
                                 std::to_string(self_tensor->dim()) + " and " +
                                 std::to_string(mat2_tensor->dim());
          ET_LOG(Error, "%s", error_msg.c_str());
          throw std::runtime_error(error_msg);
        }

        int64_t M = self_tensor->sizes()[0];  // rows of self
        int64_t K = self_tensor->sizes()[1];  // cols of self / rows of mat2
        int64_t N = mat2_tensor->sizes()[1];  // cols of mat2

        // Check matrix multiplication compatibility
        if (self_tensor->sizes()[1] != mat2_tensor->sizes()[0]) {
          std::string error_msg = "aoti_torch_mps_mm_out: incompatible matrix sizes for mm (" +
                                 std::to_string(M) + "x" + std::to_string(K) + " and " +
                                 std::to_string(mat2_tensor->sizes()[0]) + "x" + std::to_string(N) + ")";
          ET_LOG(Error, "%s", error_msg.c_str());
          throw std::runtime_error(error_msg);
        }

        // Log tensor shapes for debugging
        ET_LOG(Debug, "aoti_torch_mps_mm_out: self shape: [%d, %d], mat2 shape: [%d, %d], out shape: [%d, %d]",
               (int)M, (int)K, (int)mat2_tensor->sizes()[0], (int)N,
               out_tensor->dim() > 0 ? (int)out_tensor->sizes()[0] : 0,
               out_tensor->dim() > 1 ? (int)out_tensor->sizes()[1] : 0);

        // Get Metal device
        id<MTLDevice> device = get_metal_device();
        if (!device) {
          ET_LOG(Error, "aoti_torch_mps_mm_out: Failed to get Metal device");
          throw std::runtime_error("Failed to get Metal device");
        }

        // Get Metal buffers from tensors using the global mapping
        void* self_data_ptr = self_tensor->mutable_data_ptr();
        void* mat2_data_ptr = mat2_tensor->mutable_data_ptr();
        void* out_data_ptr = out_tensor->mutable_data_ptr();

        id<MTLBuffer> self_buffer = nullptr;
        id<MTLBuffer> mat2_buffer = nullptr;
        id<MTLBuffer> out_buffer = nullptr;

        // Look up Metal buffers from the global mapping
        auto self_it = ptr_to_mtl_buffer.find(self_data_ptr);
        auto mat2_it = ptr_to_mtl_buffer.find(mat2_data_ptr);
        auto out_it = ptr_to_mtl_buffer.find(out_data_ptr);

        if (self_it != ptr_to_mtl_buffer.end()) {
          self_buffer = self_it->second;
        }
        if (mat2_it != ptr_to_mtl_buffer.end()) {
          mat2_buffer = mat2_it->second;
        }
        if (out_it != ptr_to_mtl_buffer.end()) {
          out_buffer = out_it->second;
        }

        // If buffers are not in Metal memory, create temporary Metal buffers
        if (!self_buffer) {
          size_t self_size = self_tensor->numel() * sizeof(float);
          self_buffer = [device newBufferWithBytes:self_data_ptr
                                            length:self_size
                                           options:MTLResourceStorageModeShared];
          if (!self_buffer) {
            ET_LOG(Error, "aoti_torch_mps_mm_out: Failed to create Metal buffer for self tensor");
            throw std::runtime_error("Failed to create Metal buffer for self tensor");
          }
        }

        if (!mat2_buffer) {
          size_t mat2_size = mat2_tensor->numel() * sizeof(float);
          mat2_buffer = [device newBufferWithBytes:mat2_data_ptr
                                            length:mat2_size
                                           options:MTLResourceStorageModeShared];
          if (!mat2_buffer) {
            ET_LOG(Error, "aoti_torch_mps_mm_out: Failed to create Metal buffer for mat2 tensor");
            throw std::runtime_error("Failed to create Metal buffer for mat2 tensor");
          }
        }

        if (!out_buffer) {
          size_t out_size = out_tensor->numel() * sizeof(float);
          out_buffer = [device newBufferWithBytes:out_data_ptr
                                           length:out_size
                                          options:MTLResourceStorageModeShared];
          if (!out_buffer) {
            ET_LOG(Error, "aoti_torch_mps_mm_out: Failed to create Metal buffer for out tensor");
            throw std::runtime_error("Failed to create Metal buffer for out tensor");
          }
        }

        // End any existing kernel coalescing to ensure a clean state for MPS
        stream->endKernelCoalescing();

        // Get command buffer from stream (stream manages lifecycle)
        id<MTLCommandBuffer> commandBuffer = stream->commandBuffer();
        if (!commandBuffer) {
          ET_LOG(Error, "aoti_torch_mps_mm_out: Failed to get command buffer from stream");
          throw std::runtime_error("Failed to get command buffer from stream");
        }

        // Create matrix descriptors for the multiplication
        MPSMatrixDescriptor* selfDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                              columns:K
                                                                             matrices:1
                                                                             rowBytes:K * sizeof(float)
                                                                          matrixBytes:M * K * sizeof(float)
                                                                             dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor* mat2Desc = [MPSMatrixDescriptor matrixDescriptorWithRows:K
                                                                              columns:N
                                                                             matrices:1
                                                                             rowBytes:N * sizeof(float)
                                                                          matrixBytes:K * N * sizeof(float)
                                                                             dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor* outDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                             columns:N
                                                                            matrices:1
                                                                            rowBytes:N * sizeof(float)
                                                                         matrixBytes:M * N * sizeof(float)
                                                                            dataType:MPSDataTypeFloat32];

        MPSMatrix* selfMatrix = [[MPSMatrix alloc] initWithBuffer:self_buffer
                                                           offset:0
                                                       descriptor:selfDesc];

        MPSMatrix* mat2Matrix = [[MPSMatrix alloc] initWithBuffer:mat2_buffer
                                                           offset:0
                                                       descriptor:mat2Desc];

        MPSMatrix* outMatrix = [[MPSMatrix alloc] initWithBuffer:out_buffer
                                                          offset:0
                                                      descriptor:outDesc];

        // Create matrix multiplication kernel
        MPSMatrixMultiplication* matmul = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                                           transposeLeft:NO
                                                                          transposeRight:NO
                                                                              resultRows:M
                                                                           resultColumns:N
                                                                        interiorColumns:K
                                                                                   alpha:1.0
                                                                                    beta:0.0];

        // Encode the matrix multiplication
        [matmul encodeToCommandBuffer:commandBuffer
                           leftMatrix:selfMatrix
                          rightMatrix:mat2Matrix
                         resultMatrix:outMatrix];

        // Clean up MPS objects
        [selfMatrix release];
        [mat2Matrix release];
        [outMatrix release];
        [matmul release];

        ET_LOG(Debug, "aoti_torch_mps_mm_out: Matrix multiplication completed successfully with synchronization");
      }
    });

    return Error::Ok;

  } catch (const std::exception& e) {
    ET_LOG(Error, "aoti_torch_mps_mm_out exception: %s", e.what());
    return Error::Internal;
  } catch (...) {
    ET_LOG(Error, "aoti_torch_mps_mm_out: unknown exception");
    return Error::Internal;
  }
}

AOTITorchError aoti_torch_mps_addmm_out(
    AOTITensorHandle out,
    AOTITensorHandle self,
    AOTITensorHandle mat1,
    AOTITensorHandle mat2,
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
      // Convert AOTITensorHandle to ExecutorTorch tensors
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

AOTITorchError aoti_torch_mps_convolution(
    AOTITensorHandle input,
    AOTITensorHandle weight,
    AOTITensorHandle* bias,
    const int64_t* stride,
    int64_t stride_len_,
    const int64_t* padding,
    int64_t padding_len_,
    const int64_t* dilation,
    int64_t dilation_len_,
    int32_t transposed,
    const int64_t* output_padding,
    int64_t output_padding_len_,
    int64_t groups,
    AOTITensorHandle* ret0) {
  ET_LOG(Debug, "aoti_torch_mps_convolution: Starting with input=%p, weight=%p, bias=%p, groups=%lld, transposed=%d",
         input, weight, bias, groups, transposed);

  if (!input || !weight || !ret0) {
    ET_LOG(Error, "aoti_torch_mps_convolution: null required handles (input, weight, or ret0)");
    return Error::InvalidArgument;
  }

  @autoreleasepool {
    try {
      // Convert AOTITensorHandle to ExecutorTorch tensors
      auto input_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(input);
      auto weight_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(weight);

      // bias can be null for convolutions without bias
      executorch::runtime::etensor::Tensor* bias_tensor = nullptr;
      if (bias && *bias) {
        bias_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(*bias);
        ET_LOG(Debug, "aoti_torch_mps_convolution: Has bias tensor");
      } else {
        ET_LOG(Debug, "aoti_torch_mps_convolution: No bias tensor");
      }

      ET_LOG(Debug, "aoti_torch_mps_convolution: Converted tensor handles to ET tensors");

      // Log tensor shapes for debugging
      ET_LOG(Debug, "aoti_torch_mps_convolution: input shape: [%d, %d, %d, %d]",
             input_tensor->dim() > 0 ? (int)input_tensor->sizes()[0] : 0,
             input_tensor->dim() > 1 ? (int)input_tensor->sizes()[1] : 0,
             input_tensor->dim() > 2 ? (int)input_tensor->sizes()[2] : 0,
             input_tensor->dim() > 3 ? (int)input_tensor->sizes()[3] : 0);

      ET_LOG(Debug, "aoti_torch_mps_convolution: weight shape: [%d, %d, %d, %d]",
             weight_tensor->dim() > 0 ? (int)weight_tensor->sizes()[0] : 0,
             weight_tensor->dim() > 1 ? (int)weight_tensor->sizes()[1] : 0,
             weight_tensor->dim() > 2 ? (int)weight_tensor->sizes()[2] : 0,
             weight_tensor->dim() > 3 ? (int)weight_tensor->sizes()[3] : 0);

      // Log convolution parameters
      if (stride && stride_len_ >= 2) {
        ET_LOG(Debug, "aoti_torch_mps_convolution: stride: [%lld, %lld]", stride[0], stride[1]);
      }
      if (padding && padding_len_ >= 2) {
        ET_LOG(Debug, "aoti_torch_mps_convolution: padding: [%lld, %lld]", padding[0], padding[1]);
      }
      if (dilation && dilation_len_ >= 2) {
        ET_LOG(Debug, "aoti_torch_mps_convolution: dilation: [%lld, %lld]", dilation[0], dilation[1]);
      }
      if (output_padding && output_padding_len_ >= 2) {
        ET_LOG(Debug, "aoti_torch_mps_convolution: output_padding: [%lld, %lld]", output_padding[0], output_padding[1]);
      }

      // Calculate output dimensions
      // For now, we'll create a zero-filled tensor with the expected output shape
      // TODO: Implement actual 2D convolution using MetalPerformanceShaders or custom Metal kernels

      // Get input dimensions (assuming NCHW format)
      int64_t N = input_tensor->sizes()[0];  // batch size
      int64_t C_in = input_tensor->sizes()[1];  // input channels
      int64_t H_in = input_tensor->sizes()[2];  // input height
      int64_t W_in = input_tensor->sizes()[3];  // input width

      // Get weight dimensions (assuming OIHW format for weight)
      int64_t C_out = weight_tensor->sizes()[0];  // output channels
      int64_t kernel_h = weight_tensor->sizes()[2];  // kernel height
      int64_t kernel_w = weight_tensor->sizes()[3];  // kernel width

      // Calculate output dimensions
      int64_t stride_h = stride && stride_len_ > 0 ? stride[0] : 1;
      int64_t stride_w = stride && stride_len_ > 1 ? stride[1] : 1;
      int64_t pad_h = padding && padding_len_ > 0 ? padding[0] : 0;
      int64_t pad_w = padding && padding_len_ > 1 ? padding[1] : 0;
      int64_t dil_h = dilation && dilation_len_ > 0 ? dilation[0] : 1;
      int64_t dil_w = dilation && dilation_len_ > 1 ? dilation[1] : 1;

      int64_t H_out = (H_in + 2 * pad_h - dil_h * (kernel_h - 1) - 1) / stride_h + 1;
      int64_t W_out = (W_in + 2 * pad_w - dil_w * (kernel_w - 1) - 1) / stride_w + 1;

      ET_LOG(Debug, "aoti_torch_mps_convolution: Calculated output shape: [%lld, %lld, %lld, %lld]", N, C_out, H_out, W_out);

      // Validate output dimensions are positive
      if (N <= 0 || C_out <= 0 || H_out <= 0 || W_out <= 0) {
        ET_LOG(Error, "aoti_torch_mps_convolution: Invalid output dimensions N=%lld, C_out=%lld, H_out=%lld, W_out=%lld",
               N, C_out, H_out, W_out);
        return Error::InvalidArgument;
      }

      // Create output tensor with calculated dimensions
      std::vector<int32_t> output_sizes = {(int32_t)N, (int32_t)C_out, (int32_t)H_out, (int32_t)W_out};

      // Calculate expected number of elements
      size_t expected_numel = N * C_out * H_out * W_out;
      ET_LOG(Debug, "aoti_torch_mps_convolution: Expected output tensor numel = %zu", expected_numel);

      // Log the sizes vector for debugging
      ET_LOG(Debug, "aoti_torch_mps_convolution: output_sizes vector: [%d, %d, %d, %d]",
             output_sizes[0], output_sizes[1], output_sizes[2], output_sizes[3]);

      // Allocate memory for the tensor data
      size_t tensor_size_bytes = expected_numel * sizeof(float);
      void* tensor_data = std::malloc(tensor_size_bytes);
      if (!tensor_data) {
        ET_LOG(Error, "aoti_torch_mps_convolution: Failed to allocate %zu bytes for tensor", tensor_size_bytes);
        return Error::Internal;
      }

      // Zero out the allocated memory
      std::memset(tensor_data, 0, tensor_size_bytes);

      // Create tensor using aoti_torch_create_tensor_from_blob_v2 to ensure we have control over the memory
      // Convert sizes vector to int64_t array
      std::vector<int64_t> output_sizes_int64 = {N, C_out, H_out, W_out};

      // Calculate default strides for a contiguous tensor (NCHW format)
      std::vector<int64_t> output_strides = {
          C_out * H_out * W_out,  // Stride for N dimension
          H_out * W_out,          // Stride for C dimension
          W_out,                  // Stride for H dimension
          1                       // Stride for W dimension
      };

      AOTITensorHandle output_tensor_handle = nullptr;

      AOTITorchError create_result = aoti_torch_create_tensor_from_blob_v2(
          tensor_data,
          4,  // ndim
          output_sizes_int64.data(),
          output_strides.data(),
          0,  // storage_offset
          static_cast<int32_t>(SupportedDTypes::FLOAT32),  // dtype
          0,  // device_type (CPU)
          0,  // device_index
          &output_tensor_handle,
          0,  // layout (strided)
          nullptr,  // opaque_metadata
          0   // opaque_metadata_size
      );

      if (create_result != Error::Ok || !output_tensor_handle) {
        ET_LOG(Error, "aoti_torch_mps_convolution: Failed to create output tensor, error code: %d", static_cast<int>(create_result));
        std::free(tensor_data);  // Free the allocated memory on failure
        return Error::Internal;
      }

      // Verify the tensor was created with the correct size
      auto* et_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(output_tensor_handle);
      size_t actual_numel = et_tensor->numel();
      ET_LOG(Debug, "aoti_torch_mps_convolution: Created tensor with actual numel = %zu", actual_numel);

      if (actual_numel != expected_numel) {
        ET_LOG(Error, "aoti_torch_mps_convolution: Tensor size mismatch. Expected %zu, got %zu", expected_numel, actual_numel);
        std::free(tensor_data);  // Free the allocated memory on failure
        return Error::Internal;
      }

      // Store the tensor handle - mark that we own the memory since we manually allocated it with malloc
      *ret0 = output_tensor_handle;
      is_tensor_own_memory[et_tensor] = true;  // We allocated the memory manually

      ET_LOG(Debug, "aoti_torch_mps_convolution: Created zero-filled output tensor with %zu elements", actual_numel);
      return Error::Ok;

    } catch (const std::exception& e) {
      ET_LOG(Error, "aoti_torch_mps_convolution exception: %s", e.what());
      return Error::Internal;
    } catch (...) {
      ET_LOG(Error, "aoti_torch_mps_convolution: unknown exception");
      return Error::Internal;
    }
  }
}

} // extern "C"

} // namespace aoti
} // namespace backends
} // namespace executorch
