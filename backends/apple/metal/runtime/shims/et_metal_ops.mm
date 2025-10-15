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
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/backends/apple/metal/runtime/shims/et_metal_ops.h>
#include <executorch/backends/apple/metal/runtime/shims/et_metal.h>
#include <executorch/backends/apple/metal/runtime/shims/shim_mps.h>
#include <executorch/backends/apple/metal/runtime/shims/utils.h>
#include <executorch/backends/apple/metal/runtime/shims/memory.h>
#include <functional>
#include <unordered_map>

namespace executorch {
namespace backends {
namespace metal {

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

  @autoreleasepool {
    try {
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

      // Use the same dispatch pattern as other MPS operations for consistent synchronization
      ETMetalStream* stream = getCurrentMetalStream();
      if (!stream) {
        ET_LOG(Error, "aoti_torch_mps_mm_out: Failed to get current Metal stream");
        return Error::Internal;
      }

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

      // Look up Metal buffers from the global mapping
      auto self_it = ptr_to_mtl_buffer.find(self_data_ptr);
      auto mat2_it = ptr_to_mtl_buffer.find(mat2_data_ptr);
      auto out_it = ptr_to_mtl_buffer.find(out_data_ptr);

      if (self_it == ptr_to_mtl_buffer.end()) {
        ET_LOG(Error, "aoti_torch_mps_mm_out: self tensor not found in Metal buffer mapping");
        throw std::runtime_error("self tensor not found in Metal buffer mapping");
      }
      if (mat2_it == ptr_to_mtl_buffer.end()) {
        ET_LOG(Error, "aoti_torch_mps_mm_out: mat2 tensor not found in Metal buffer mapping");
        throw std::runtime_error("mat2 tensor not found in Metal buffer mapping");
      }
      if (out_it == ptr_to_mtl_buffer.end()) {
        ET_LOG(Error, "aoti_torch_mps_mm_out: out tensor not found in Metal buffer mapping");
        throw std::runtime_error("out tensor not found in Metal buffer mapping");
      }

      id<MTLBuffer> self_buffer = self_it->second;
      id<MTLBuffer> mat2_buffer = mat2_it->second;
      id<MTLBuffer> out_buffer = out_it->second;

      ET_LOG(Debug, "aoti_torch_mps_mm_out: Using existing Metal buffers - self=%p, mat2=%p, out=%p",
             self_buffer, mat2_buffer, out_buffer);

      // End any existing kernel coalescing to ensure a clean state for MPS
      stream->endKernelCoalescing();

      // Determine data type and element size
      int32_t dtype = static_cast<int32_t>(self_tensor->scalar_type());
      MPSDataType mps_dtype;
      size_t element_size;

      ET_LOG(Debug, "aoti_torch_mps_mm_out: self_tensor scalar_type=%d, SupportedDTypes::FLOAT32=%d, SupportedDTypes::BFLOAT16=%d",
             dtype, static_cast<int32_t>(SupportedDTypes::FLOAT32), static_cast<int32_t>(SupportedDTypes::BFLOAT16));

      if (dtype == static_cast<int32_t>(SupportedDTypes::FLOAT32)) {
        mps_dtype = MPSDataTypeFloat32;
        element_size = sizeof(float);
      } else if (dtype == static_cast<int32_t>(SupportedDTypes::BFLOAT16)) {
        mps_dtype = MPSDataTypeBFloat16;
        element_size = sizeof(uint16_t);  // bfloat16 is 16 bits
      } else {
        ET_LOG(Error, "aoti_torch_mps_mm_out: Unsupported data type: %d", dtype);
        throw std::runtime_error("Unsupported data type for matrix multiplication");
      }

      ET_LOG(Debug, "aoti_torch_mps_mm_out: dtype=%d, element_size=%zu", dtype, element_size);
      ET_LOG(Debug, "aoti_torch_mps_mm_out: M=%lld, K=%lld, N=%lld", M, K, N);

      // Create MPSGraph for matrix multiplication
      MPSGraph* mpsGraph = [MPSGraph new];
      ET_LOG(Debug, "aoti_torch_mps_mm_out: Created MPSGraph instance");

      // Define tensor shapes for placeholders
      NSArray<NSNumber*>* selfShape = @[@(M), @(K)];
      NSArray<NSNumber*>* mat2Shape = @[@(K), @(N)];
      NSArray<NSNumber*>* outShape = @[@(M), @(N)];

      ET_LOG(Debug, "aoti_torch_mps_mm_out: Creating placeholders with shapes self:[%d,%d] mat2:[%d,%d]",
             (int)M, (int)K, (int)K, (int)N);

      // Create placeholders for input tensors
      MPSGraphTensor* selfPlaceholder = [mpsGraph placeholderWithShape:selfShape
                                                              dataType:mps_dtype
                                                                  name:@"self"];
      MPSGraphTensor* mat2Placeholder = [mpsGraph placeholderWithShape:mat2Shape
                                                              dataType:mps_dtype
                                                                  name:@"mat2"];

      ET_LOG(Debug, "aoti_torch_mps_mm_out: Created input placeholders");

      // Perform matrix multiplication using MPSGraph
      MPSGraphTensor* mmOutput = [mpsGraph matrixMultiplicationWithPrimaryTensor:selfPlaceholder
                                                                 secondaryTensor:mat2Placeholder
                                                                            name:@"matrix_multiplication"];

      ET_LOG(Debug, "aoti_torch_mps_mm_out: Successfully created matrix multiplication tensor");

      // Create feeds dictionary for graph execution
      NSMutableDictionary* feeds = [NSMutableDictionary dictionary];

      // Create MPSGraphTensorData objects for input tensors
      MPSGraphTensorData* selfData = [[MPSGraphTensorData alloc] initWithMTLBuffer:self_buffer
                                                                              shape:selfShape
                                                                           dataType:mps_dtype];
      MPSGraphTensorData* mat2Data = [[MPSGraphTensorData alloc] initWithMTLBuffer:mat2_buffer
                                                                              shape:mat2Shape
                                                                           dataType:mps_dtype];

      feeds[selfPlaceholder] = selfData;
      feeds[mat2Placeholder] = mat2Data;

      ET_LOG(Debug, "aoti_torch_mps_mm_out: Created feeds dictionary");

      // Create results dictionary
      MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:out_buffer
                                                                               shape:outShape
                                                                            dataType:mps_dtype];

      NSDictionary* results = @{mmOutput: outputData};
      ET_LOG(Debug, "aoti_torch_mps_mm_out: Created results dictionary");

      // Execute the MPSGraph
      ET_LOG(Debug, "aoti_torch_mps_mm_out: Executing MPSGraph");

      @try {
        // Use stream helper to encode and synchronize correctly
        stream->executeMPSGraph(mpsGraph, feeds, results, SyncType::COMMIT_AND_CONTINUE);
      } @catch (NSException *exception) {
        ET_LOG(Error, "aoti_torch_mps_mm_out: NSException caught during executeMPSGraph: %s - %s",
              [[exception name] UTF8String], [[exception reason] UTF8String]);
        throw std::runtime_error("MPSGraph execution failed with NSException");
      }

      ET_LOG(Debug, "aoti_torch_mps_mm_out: MPSGraph execution completed successfully");

      return Error::Ok;

    } catch (const std::exception& e) {
      ET_LOG(Error, "aoti_torch_mps_mm_out exception: %s", e.what());
      return Error::Internal;
    } catch (...) {
      ET_LOG(Error, "aoti_torch_mps_mm_out: unknown exception");
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

      // Support conv1d and conv2d by inspecting weight rank.
      // conv1d: weight dims = [C_out, C_in, K]
      // conv2d: weight dims = [C_out, C_in, Kh, Kw]
      bool is_conv1d = (weight_tensor->dim() == 3);

      // Accept input ranks:
      // conv1d: 2D (C,W) or 3D (N,C,W)
      // conv2d: 3D (C,H,W) or 4D (N,C,H,W)
      bool has_batch_dim = false;
      bool is_input_4d = false;
      int64_t N = 1, C_in = 0, H_in = 1, W_in = 0;
      if (is_conv1d) {
        if (input_tensor->dim() == 2) {
          // (C, W)
          has_batch_dim = false;
          C_in = input_tensor->sizes()[0];
          W_in = input_tensor->sizes()[1];
          H_in = 1;
        } else if (input_tensor->dim() == 3) {
          // (N, C, W)
          has_batch_dim = true;
          N = input_tensor->sizes()[0];
          C_in = input_tensor->sizes()[1];
          W_in = input_tensor->sizes()[2];
          H_in = 1;
        } else {
          ET_LOG(Error, "aoti_torch_mps_convolution: conv1d expects 2D or 3D input, got %d", (int)input_tensor->dim());
          return Error::InvalidArgument;
        }
      } else {
        is_input_4d = (input_tensor->dim() == 4);
        if (is_input_4d) {
          // (N, C, H, W)
          has_batch_dim = true;
          N = input_tensor->sizes()[0];
          C_in = input_tensor->sizes()[1];
          H_in = input_tensor->sizes()[2];
          W_in = input_tensor->sizes()[3];
        } else if (input_tensor->dim() == 3) {
          // (C, H, W)
          has_batch_dim = false;
          N = 1;
          C_in = input_tensor->sizes()[0];
          H_in = input_tensor->sizes()[1];
          W_in = input_tensor->sizes()[2];
        } else {
          ET_LOG(Error, "aoti_torch_mps_convolution: conv2d expects 3D or 4D input, got %d", (int)input_tensor->dim());
          return Error::InvalidArgument;
        }
      }

      // Get weight dimensions
      int64_t C_out = weight_tensor->sizes()[0];  // output channels
      int64_t kernel_h = is_conv1d ? 1 : weight_tensor->sizes()[2];  // kernel height
      int64_t kernel_w = is_conv1d ? weight_tensor->sizes()[2] : weight_tensor->sizes()[3];  // kernel width

      // Calculate output spatial dimensions
      int64_t stride_h = is_conv1d ? 1 : (stride && stride_len_ > 0 ? stride[0] : 1);
      int64_t stride_w = is_conv1d ? (stride && stride_len_ > 0 ? stride[0] : 1)
                                   : (stride && stride_len_ > 1 ? stride[1] : 1);
      int64_t pad_h = is_conv1d ? 0 : (padding && padding_len_ > 0 ? padding[0] : 0);
      int64_t pad_w = is_conv1d ? (padding && padding_len_ > 0 ? padding[0] : 0)
                                : (padding && padding_len_ > 1 ? padding[1] : 0);
      int64_t dil_h = is_conv1d ? 1 : (dilation && dilation_len_ > 0 ? dilation[0] : 1);
      int64_t dil_w = is_conv1d ? (dilation && dilation_len_ > 0 ? dilation[0] : 1)
                                : (dilation && dilation_len_ > 1 ? dilation[1] : 1);

      int64_t H_out, W_out;
      if (transposed) {
        // For transposed convolution, output size calculation is different
        int64_t output_pad_h = is_conv1d ? 0 : (output_padding && output_padding_len_ > 0 ? output_padding[0] : 0);
        int64_t output_pad_w = is_conv1d ? (output_padding && output_padding_len_ > 0 ? output_padding[0] : 0)
                                         : (output_padding && output_padding_len_ > 1 ? output_padding[1] : 0);
        H_out = is_conv1d ? 1 : ((H_in - 1) * stride_h - 2 * pad_h + dil_h * (kernel_h - 1) + output_pad_h + 1);
        W_out = (W_in - 1) * stride_w - 2 * pad_w + dil_w * (kernel_w - 1) + output_pad_w + 1;
      } else {
        // Regular convolution output size calculation
        H_out = is_conv1d ? 1 : ((H_in + 2 * pad_h - dil_h * (kernel_h - 1) - 1) / stride_h + 1);
        W_out = (W_in + 2 * pad_w - dil_w * (kernel_w - 1) - 1) / stride_w + 1;
      }

      if (!is_conv1d && is_input_4d) {
        ET_LOG(Debug, "aoti_torch_mps_convolution: Calculated 4D output shape: [%lld, %lld, %lld, %lld]", N, C_out, H_out, W_out);
      } else if (!is_conv1d) {
        ET_LOG(Debug, "aoti_torch_mps_convolution: Calculated 3D output shape: [%lld, %lld, %lld]", C_out, H_out, W_out);
      } else if (is_conv1d && has_batch_dim) {
        ET_LOG(Debug, "aoti_torch_mps_convolution: Calculated 3D (1D conv) output shape: [%lld, %lld, %lld]", N, C_out, W_out);
      } else {
        ET_LOG(Debug, "aoti_torch_mps_convolution: Calculated 2D (1D conv) output shape: [%lld, %lld]", C_out, W_out);
      }

      // Validate output dimensions are positive
      if (N <= 0 || C_out <= 0 || H_out <= 0 || W_out <= 0) {
        ET_LOG(Error, "aoti_torch_mps_convolution: Invalid output dimensions N=%lld, C_out=%lld, H_out=%lld, W_out=%lld",
               N, C_out, H_out, W_out);
        return Error::InvalidArgument;
      }

      // Use the same dispatch pattern as other MPS operations for consistent synchronization
      ETMetalStream* stream = getCurrentMetalStream();
      if (!stream) {
        ET_LOG(Error, "aoti_torch_mps_convolution: Failed to get current Metal stream");
        return Error::Internal;
      }

      // Get Metal device
      id<MTLDevice> device = get_metal_device();
      if (!device) {
        ET_LOG(Error, "aoti_torch_mps_convolution: Failed to get Metal device");
        throw std::runtime_error("Failed to get Metal device");
      }

      // Get Metal buffers from tensors using the global mapping
      void* input_data_ptr = input_tensor->mutable_data_ptr();
      void* weight_data_ptr = weight_tensor->mutable_data_ptr();

      // Look up Metal buffers from the global mapping
      auto input_it = ptr_to_mtl_buffer.find(input_data_ptr);
      auto weight_it = ptr_to_mtl_buffer.find(weight_data_ptr);

      if (input_it == ptr_to_mtl_buffer.end()) {
        ET_LOG(Error, "aoti_torch_mps_convolution: input tensor not found in Metal buffer mapping");
        throw std::runtime_error("input tensor not found in Metal buffer mapping");
      }
      if (weight_it == ptr_to_mtl_buffer.end()) {
        ET_LOG(Error, "aoti_torch_mps_convolution: weight tensor not found in Metal buffer mapping");
        throw std::runtime_error("weight tensor not found in Metal buffer mapping");
      }

      id<MTLBuffer> input_buffer = input_it->second;
      id<MTLBuffer> weight_buffer = weight_it->second;

      ET_LOG(Debug, "aoti_torch_mps_convolution: Using existing Metal buffers - input=%p, weight=%p",
              input_buffer, weight_buffer);

      // End any existing kernel coalescing to ensure a clean state for MPS
      stream->endKernelCoalescing();

      // Ensure stream is ready; command buffer handled internally by stream helpers

      // Determine data type and element size
      int32_t dtype = static_cast<int32_t>(input_tensor->scalar_type());
      MPSDataType mps_dtype;
      size_t element_size;

      if (dtype == static_cast<int32_t>(SupportedDTypes::FLOAT32)) {
        mps_dtype = MPSDataTypeFloat32;
        element_size = sizeof(float);
      } else if (dtype == static_cast<int32_t>(SupportedDTypes::BFLOAT16)) {
        mps_dtype = MPSDataTypeBFloat16;
        element_size = sizeof(uint16_t);  // bfloat16 is 16 bits
      } else {
        ET_LOG(Error, "aoti_torch_mps_convolution: Unsupported data type: %d", dtype);
        throw std::runtime_error("Unsupported data type for convolution");
      }

      ET_LOG(Debug, "aoti_torch_mps_convolution: mps_dtype=%d, element_size=%zu", mps_dtype, element_size);

      // Create MPSGraph for convolution
      MPSGraph* mpsGraph = [MPSGraph new];
      ET_LOG(Debug, "aoti_torch_mps_convolution: Created MPSGraph instance");

      // Define tensor shapes for placeholders (always 4D NCHW for MPSGraph)
      NSArray<NSNumber*>* inputShape = @[@(N), @(C_in), @(H_in), @(W_in)];
      NSArray<NSNumber*>* weightShape = @[@(C_out), @(C_in), @(kernel_h), @(kernel_w)];

      ET_LOG(Debug, "aoti_torch_mps_convolution: Creating placeholders with shapes input:[%d,%d,%d,%d] weight:[%d,%d,%d,%d]",
              (int)N, (int)C_in, (int)H_in, (int)W_in,
              (int)C_out, (int)C_in, (int)kernel_h, (int)kernel_w);

      // Create placeholders for input tensors
      MPSGraphTensor* inputPlaceholder = [mpsGraph placeholderWithShape:inputShape
                                                                dataType:mps_dtype
                                                                    name:@"input"];
      MPSGraphTensor* weightPlaceholder = [mpsGraph placeholderWithShape:weightShape
                                                                  dataType:mps_dtype
                                                                      name:@"weight"];

      ET_LOG(Debug, "aoti_torch_mps_convolution: Created input and weight placeholders");

      // Create convolution descriptor
      MPSGraphConvolution2DOpDescriptor* convDesc = [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:stride_w
                                                                                                      strideInY:stride_h
                                                                                                    dilationRateInX:dil_w
                                                                                                    dilationRateInY:dil_h
                                                                                                        groups:groups
                                                                                                        paddingLeft:pad_w
                                                                                                      paddingRight:pad_w
                                                                                                        paddingTop:pad_h
                                                                                                      paddingBottom:pad_h
                                                                                                        paddingStyle:MPSGraphPaddingStyleExplicit
                                                                                                        dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                                                                                                    weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

      ET_LOG(Debug, "aoti_torch_mps_convolution: Created convolution descriptor with stride=[%lld,%lld], padding=[%lld,%lld], dilation=[%lld,%lld], groups=%lld",
              stride_w, stride_h, pad_w, pad_h, dil_w, dil_h, groups);

      // Perform convolution using MPSGraph
      MPSGraphTensor* convOutput = nil;
      if (transposed) {
        ET_LOG(Debug, "aoti_torch_mps_convolution: Using transposed convolution");
        // For transposed convolution, we need to handle output padding
        int64_t output_pad_h = output_padding && output_padding_len_ > 0 ? output_padding[0] : 0;
        int64_t output_pad_w = output_padding && output_padding_len_ > 1 ? output_padding[1] : 0;

        // For transposed convolution, we need to adjust the padding calculation
        // In transposed convolution, the effective padding is typically negative
        // and we use output_padding to control the final output size
        int64_t transposed_pad_h = pad_h - output_pad_h;
        int64_t transposed_pad_w = pad_w - output_pad_w;

        // Create transposed convolution descriptor with adjusted padding
        MPSGraphConvolution2DOpDescriptor* transposedConvDesc = [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:stride_w
                                                                                                                  strideInY:stride_h
                                                                                                            dilationRateInX:dil_w
                                                                                                            dilationRateInY:dil_h
                                                                                                                    groups:groups
                                                                                                              paddingLeft:transposed_pad_w
                                                                                                              paddingRight:transposed_pad_w
                                                                                                                paddingTop:transposed_pad_h
                                                                                                            paddingBottom:transposed_pad_h
                                                                                                              paddingStyle:MPSGraphPaddingStyleExplicit
                                                                                                              dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                                                                                                          weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

        convOutput = [mpsGraph convolution2DWithSourceTensor:inputPlaceholder
                                                  weightsTensor:weightPlaceholder
                                                      descriptor:transposedConvDesc
                                                            name:@"transposed_convolution"];
      } else {
        ET_LOG(Debug, "aoti_torch_mps_convolution: Using regular convolution");
        convOutput = [mpsGraph convolution2DWithSourceTensor:inputPlaceholder
                                                  weightsTensor:weightPlaceholder
                                                      descriptor:convDesc
                                                            name:@"convolution"];
      }

      ET_LOG(Debug, "aoti_torch_mps_convolution: Successfully created convolution tensor");

      // Handle bias if provided
      MPSGraphTensor* finalOutput = convOutput;
      MPSGraphTensor* biasPlaceholder = nil;
      if (bias_tensor) {
        ET_LOG(Debug, "aoti_torch_mps_convolution: Adding bias to convolution output");

        // Get bias tensor data
        void* bias_data_ptr = bias_tensor->mutable_data_ptr();
        auto bias_it = ptr_to_mtl_buffer.find(bias_data_ptr);

        if (bias_it != ptr_to_mtl_buffer.end()) {
          id<MTLBuffer> bias_buffer = bias_it->second;

          // Create bias placeholder
          NSArray<NSNumber*>* biasShape = @[@(C_out)];
          biasPlaceholder = [mpsGraph placeholderWithShape:biasShape
                                                    dataType:mps_dtype
                                                        name:@"bias"];

          // Add bias to convolution output
          finalOutput = [mpsGraph additionWithPrimaryTensor:convOutput
                                            secondaryTensor:biasPlaceholder
                                                        name:@"add_bias"];

          ET_LOG(Debug, "aoti_torch_mps_convolution: Added bias placeholder to graph");
        } else {
          ET_LOG(Debug, "aoti_torch_mps_convolution: Bias tensor not found in Metal buffer mapping, skipping bias");
        }
      }

      // Create feeds dictionary for graph execution
      NSMutableDictionary* feeds = [NSMutableDictionary dictionary];

      // Create MPSGraphTensorData objects for input tensors
      MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:input_buffer
                                                                                shape:inputShape
                                                                            dataType:mps_dtype];
      MPSGraphTensorData* weightData = [[MPSGraphTensorData alloc] initWithMTLBuffer:weight_buffer
                                                                                shape:weightShape
                                                                            dataType:mps_dtype];

      feeds[inputPlaceholder] = inputData;
      feeds[weightPlaceholder] = weightData;

      // Add bias data to feeds if provided
      if (bias_tensor && biasPlaceholder) {
        void* bias_data_ptr = bias_tensor->mutable_data_ptr();
        auto bias_it = ptr_to_mtl_buffer.find(bias_data_ptr);

        if (bias_it != ptr_to_mtl_buffer.end()) {
          id<MTLBuffer> bias_buffer = bias_it->second;
          NSArray<NSNumber*>* biasShape = @[@(C_out)];
          MPSGraphTensorData* biasData = [[MPSGraphTensorData alloc] initWithMTLBuffer:bias_buffer
                                                                                    shape:biasShape
                                                                                dataType:mps_dtype];

          feeds[biasPlaceholder] = biasData;
          ET_LOG(Debug, "aoti_torch_mps_convolution: Added bias tensor to feeds");
        }
      }

      ET_LOG(Debug, "aoti_torch_mps_convolution: Created feeds dictionary");

      // Create or reuse output Metal buffer via AOTI API; keeps GPU residency
      size_t output_size_bytes = N * C_out * H_out * W_out * element_size;
      void* output_contents_ptr = nullptr;
      AOTITorchError malloc_err = aoti_torch_mps_malloc(&output_contents_ptr, output_size_bytes);
      if (malloc_err != Error::Ok || !output_contents_ptr) {
        ET_LOG(Error, "aoti_torch_mps_convolution: Failed to allocate Metal buffer via aoti_torch_mps_malloc");
        throw std::runtime_error("Failed to allocate output Metal buffer");
      }

      auto out_it = ptr_to_mtl_buffer.find(output_contents_ptr);
      if (out_it == ptr_to_mtl_buffer.end()) {
        ET_LOG(Error, "aoti_torch_mps_convolution: aoti_torch_mps_malloc did not register buffer in map");
        throw std::runtime_error("Failed to look up allocated Metal buffer");
      }
      id<MTLBuffer> output_buffer = out_it->second;

      // Create results dictionary (MPSGraph output is 4D)
      NSArray<NSNumber*>* outputShape = @[@(N), @(C_out), @(H_out), @(W_out)];
      MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:output_buffer
                                                                                shape:outputShape
                                                                              dataType:mps_dtype];

      NSDictionary* results = @{finalOutput: outputData};
      ET_LOG(Debug, "aoti_torch_mps_convolution: Created results dictionary");

      // Execute the MPSGraph
      ET_LOG(Debug, "aoti_torch_mps_convolution: Executing MPSGraph");

      @try {
        // Use stream helper to encode and synchronize correctly
        stream->executeMPSGraph(mpsGraph, feeds, results, SyncType::COMMIT_AND_CONTINUE);
      } @catch (NSException *exception) {
        ET_LOG(Error, "aoti_torch_mps_convolution: NSException caught during executeMPSGraph: %s - %s",
              [[exception name] UTF8String], [[exception reason] UTF8String]);
        throw std::runtime_error("MPSGraph execution failed with NSException");
      }
      // } @catch (const std::exception& e) {
      //   ET_LOG(Error, "aoti_torch_mps_convolution exception: %s", e.what());
      //   throw std::runtime_error("MPSGraph execution failed");
      // }

      ET_LOG(Debug, "aoti_torch_mps_convolution: MPSGraph execution completed successfully");

      // Create output tensor handle on device (MPS) that points to GPU buffer
      std::vector<int64_t> output_sizes_int64;
      std::vector<int64_t> output_strides;
      if (!is_conv1d && is_input_4d) {
        output_sizes_int64 = {N, C_out, H_out, W_out};
        // Contiguous NCHW strides
        output_strides = {
            C_out * H_out * W_out,
            H_out * W_out,
            W_out,
            1
        };
      } else if (!is_conv1d) {
        output_sizes_int64 = {C_out, H_out, W_out};
        // Contiguous CHW strides
        output_strides = {
            H_out * W_out,
            W_out,
            1
        };
      } else if (is_conv1d && has_batch_dim) {
        output_sizes_int64 = {N, C_out, W_out};
        // Contiguous NCW strides
        output_strides = {
            C_out * W_out,
            W_out,
            1
        };
      } else {
        output_sizes_int64 = {C_out, W_out};
        // Contiguous CW strides
        output_strides = {
            W_out,
            1
        };
      }

      // Use the GPU buffer contents pointer directly for the tensor storage
      void* tensor_data = output_contents_ptr;

      AOTITensorHandle output_tensor_handle = nullptr;

      AOTITorchError create_result = aoti_torch_create_tensor_from_blob_v2(
          tensor_data,
          static_cast<int64_t>(output_sizes_int64.size()),  // ndim
          output_sizes_int64.data(),
          output_strides.data(),
          0,  // storage_offset
          dtype,  // dtype
          13,  // device_type (MPS)
          0,  // device_index
          &output_tensor_handle,
          0,  // layout (strided)
          nullptr,  // opaque_metadata
          0   // opaque_metadata_size
      );

      if (create_result != Error::Ok || !output_tensor_handle) {
        ET_LOG(Error, "aoti_torch_mps_convolution: Failed to create output tensor, error code: %d", static_cast<int>(create_result));
        aoti_torch_mps_free(tensor_data);  // Free the allocated GPU memory on failure
        throw std::runtime_error("Failed to create output tensor");
      }

      // Verify the tensor was created with the correct size
      auto* et_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(output_tensor_handle);
      size_t actual_numel = et_tensor->numel();
      size_t expected_numel = static_cast<size_t>(N * C_out * H_out * W_out);

      if (actual_numel != expected_numel) {
        ET_LOG(Error, "aoti_torch_mps_convolution: Tensor size mismatch. Expected %zu, got %zu", expected_numel, actual_numel);
        aoti_torch_mps_free(tensor_data);  // Free the allocated GPU memory on failure
        throw std::runtime_error("Tensor size mismatch");
      }

      // Store the tensor handle - mark that we own the memory since we manually allocated it with malloc
      *ret0 = output_tensor_handle;
      is_tensor_own_memory[et_tensor] = true;  // We allocated the GPU memory

      ET_LOG(Debug, "aoti_torch_mps_convolution: Created output tensor with %zu elements using MPSGraph", actual_numel);

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

AOTITorchError aoti_torch_mps__scaled_dot_product_attention_math_for_mps(
    AOTITensorHandle query,
    AOTITensorHandle key,
    AOTITensorHandle value,
    AOTITensorHandle* attn_mask,
    double dropout_p,
    int32_t is_causal,
    AOTITensorHandle* dropout_mask,
    double* scale,
    AOTITensorHandle* ret0,
    AOTITensorHandle* ret1) {

  ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Starting with MPSGraph implementation");

  if (!query || !key || !value || !ret0 || !ret1) {
    ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: null required tensor handles");
    return Error::InvalidArgument;
  }

  // Use the same dispatch pattern as other MPS operations for consistent synchronization
  ETMetalStream* stream = getCurrentMetalStream();
  if (!stream) {
    ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Failed to get current Metal stream");
    return Error::Internal;
  }

  try {
    @autoreleasepool {
      // Convert AOTITensorHandle to ExecutorTorch tensors
      auto* query_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(query);
      auto* key_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(key);
      auto* value_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(value);

      ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Converted tensor handles to ET tensors");

        // Validate tensor dimensions
        if (query_tensor->dim() < 3 || key_tensor->dim() < 3 || value_tensor->dim() < 3) {
          std::string error_msg = "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: tensors must be at least 3-D, got " +
                                 std::to_string(query_tensor->dim()) + ", " +
                                 std::to_string(key_tensor->dim()) + ", " +
                                 std::to_string(value_tensor->dim());
          ET_LOG(Error, "%s", error_msg.c_str());
          throw std::runtime_error(error_msg);
        }

        // Get tensor dimensions (assuming [batch, num_heads, seq_len, head_dim] format)
        int64_t batchSize = query_tensor->sizes()[0];
        int64_t num_heads = query_tensor->sizes()[1];
        int64_t qSize = query_tensor->sizes()[2];
        int64_t headSize = query_tensor->sizes()[3];
        int64_t kvSeqLength = key_tensor->sizes()[2];

        ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: batchSize=%lld, num_heads=%lld, qSize=%lld, headSize=%lld, kvSeqLength=%lld",
               batchSize, num_heads, qSize, headSize, kvSeqLength);

        // Determine data type and element size
        int32_t dtype = static_cast<int32_t>(query_tensor->scalar_type());
        MPSDataType mps_dtype;
        size_t element_size;

        if (dtype == static_cast<int32_t>(SupportedDTypes::FLOAT32)) {
          mps_dtype = MPSDataTypeFloat32;
          element_size = sizeof(float);
        } else if (dtype == static_cast<int32_t>(SupportedDTypes::BFLOAT16)) {
          mps_dtype = MPSDataTypeBFloat16;
          element_size = sizeof(uint16_t);  // bfloat16 is 16 bits
        } else {
          ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Unsupported data type: %d", dtype);
          throw std::runtime_error("Unsupported data type for scaled dot product attention");
        }

        ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: mps_dtype=%d, element_size=%zu", mps_dtype, element_size);

        // Check that headSize is not zero to avoid division by zero
        if (headSize == 0) {
          ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: headSize is zero");
          throw std::runtime_error("headSize must be non-zero for scaled dot product attention");
        }

        // Calculate scale factor
        double scale_factor = scale ? *scale : (1.0 / sqrt(static_cast<double>(headSize)));
        ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: scale_factor=%f", scale_factor);

        // Get Metal device
        id<MTLDevice> device = get_metal_device();
        if (!device) {
          ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Failed to get Metal device");
          throw std::runtime_error("Failed to get Metal device");
        }

        // Get Metal buffers for input tensors
        void* query_data_ptr = query_tensor->mutable_data_ptr();
        void* key_data_ptr = key_tensor->mutable_data_ptr();
        void* value_data_ptr = value_tensor->mutable_data_ptr();

        id<MTLBuffer> query_buffer = nullptr;
        id<MTLBuffer> key_buffer = nullptr;
        id<MTLBuffer> value_buffer = nullptr;

        // Look up Metal buffers from the global mapping
        auto query_it = ptr_to_mtl_buffer.find(query_data_ptr);
        auto key_it = ptr_to_mtl_buffer.find(key_data_ptr);
        auto value_it = ptr_to_mtl_buffer.find(value_data_ptr);

        if (query_it != ptr_to_mtl_buffer.end()) {
          query_buffer = query_it->second;
        }
        if (key_it != ptr_to_mtl_buffer.end()) {
          key_buffer = key_it->second;
        }
        if (value_it != ptr_to_mtl_buffer.end()) {
          value_buffer = value_it->second;
        }

        // Create temporary Metal buffers if not found in mapping
        if (!query_buffer) {
          size_t query_size = query_tensor->numel() * element_size;
          query_buffer = [device newBufferWithBytes:query_data_ptr
                                             length:query_size
                                            options:MTLResourceStorageModeShared];
          if (!query_buffer) {
            ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Failed to create Metal buffer for query tensor");
            throw std::runtime_error("Failed to create Metal buffer for query tensor");
          }
        }

        if (!key_buffer) {
          size_t key_size = key_tensor->numel() * element_size;
          key_buffer = [device newBufferWithBytes:key_data_ptr
                                           length:key_size
                                          options:MTLResourceStorageModeShared];
          if (!key_buffer) {
            ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Failed to create Metal buffer for key tensor");
            throw std::runtime_error("Failed to create Metal buffer for key tensor");
          }
        }

        if (!value_buffer) {
          size_t value_size = value_tensor->numel() * element_size;
          value_buffer = [device newBufferWithBytes:value_data_ptr
                                             length:value_size
                                            options:MTLResourceStorageModeShared];
          if (!value_buffer) {
            ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Failed to create Metal buffer for value tensor");
            throw std::runtime_error("Failed to create Metal buffer for value tensor");
          }
        }

        // Calculate output tensor dimensions
        std::vector<int64_t> output_sizes = {batchSize, num_heads, qSize, headSize};
        std::vector<int64_t> attn_sizes = {batchSize, num_heads, qSize, kvSeqLength};

        // Calculate strides for contiguous tensors
        std::vector<int64_t> out_strides = {
            num_heads * qSize * headSize,
            qSize * headSize,
            headSize,
            1
        };

        std::vector<int64_t> attn_strides = {
            num_heads * qSize * kvSeqLength,
            qSize * kvSeqLength,
            kvSeqLength,
            1
        };

        // Allocate output Metal buffers via AOTI API to keep GPU residency and reuse
        size_t out_size_bytes = batchSize * num_heads * qSize * headSize * element_size;
        size_t attn_size_bytes = batchSize * num_heads * qSize * kvSeqLength * element_size;

        void* out_contents_ptr = nullptr;
        AOTITorchError out_malloc_err = aoti_torch_mps_malloc(&out_contents_ptr, out_size_bytes);
        if (out_malloc_err != Error::Ok || !out_contents_ptr) {
          ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Failed to allocate out buffer via aoti_torch_mps_malloc");
          throw std::runtime_error("Failed to allocate output buffer");
        }
        auto out_map_it = ptr_to_mtl_buffer.find(out_contents_ptr);
        if (out_map_it == ptr_to_mtl_buffer.end()) {
          ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: out buffer not found in mapping after malloc");
          aoti_torch_mps_free(out_contents_ptr);
          throw std::runtime_error("Mapping for out buffer missing");
        }
        id<MTLBuffer> out_buffer = out_map_it->second;

        void* attn_contents_ptr = nullptr;
        AOTITorchError attn_malloc_err = aoti_torch_mps_malloc(&attn_contents_ptr, attn_size_bytes);
        if (attn_malloc_err != Error::Ok || !attn_contents_ptr) {
          ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Failed to allocate attn buffer via aoti_torch_mps_malloc");
          aoti_torch_mps_free(out_contents_ptr);
          throw std::runtime_error("Failed to allocate attn buffer");
        }
        auto attn_map_it = ptr_to_mtl_buffer.find(attn_contents_ptr);
        if (attn_map_it == ptr_to_mtl_buffer.end()) {
          ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: attn buffer not found in mapping after malloc");
          aoti_torch_mps_free(out_contents_ptr);
          aoti_torch_mps_free(attn_contents_ptr);
          throw std::runtime_error("Mapping for attn buffer missing");
        }
        id<MTLBuffer> attn_weights_buffer = attn_map_it->second;

        // End any existing kernel coalescing to ensure a clean state for MPS
        stream->endKernelCoalescing();

        // Method 1: Using MPSGraph scaledDotProductAttention API - with detailed error handling
        ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Implementing using MPSGraph scaledDotProductAttention");

        @try {
          // Check if scaledDotProductAttentionWithQueryTensor is available
          MPSGraph* testGraph = [MPSGraph new];
          if (![testGraph respondsToSelector:@selector(scaledDotProductAttentionWithQueryTensor:keyTensor:valueTensor:maskTensor:scale:name:)]) {
            ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: scaledDotProductAttentionWithQueryTensor API not available on this system");
            throw std::runtime_error("scaledDotProductAttentionWithQueryTensor API not available on this system");
          }
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: scaledDotProductAttentionWithQueryTensor API is available");

          // Create MPSGraph for scaled dot product attention
          MPSGraph* mpsGraph = [MPSGraph new];
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Created MPSGraph instance");

          // Define tensor shapes for placeholders
          NSArray<NSNumber*>* queryShape = @[@(batchSize), @(num_heads), @(qSize), @(headSize)];
          NSArray<NSNumber*>* keyShape = @[@(batchSize), @(num_heads), @(kvSeqLength), @(headSize)];
          NSArray<NSNumber*>* valueShape = @[@(batchSize), @(num_heads), @(kvSeqLength), @(headSize)];

          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Creating placeholders with shapes Q:[%d,%d,%d,%d] K:[%d,%d,%d,%d] V:[%d,%d,%d,%d]",
                 (int)batchSize, (int)num_heads, (int)qSize, (int)headSize,
                 (int)batchSize, (int)num_heads, (int)kvSeqLength, (int)headSize,
                 (int)batchSize, (int)num_heads, (int)kvSeqLength, (int)headSize);

          // Create placeholders for input tensors
          MPSGraphTensor* queryPlaceholder = [mpsGraph placeholderWithShape:queryShape
                                                                   dataType:mps_dtype
                                                                       name:@"query"];
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Created query placeholder");

          MPSGraphTensor* keyPlaceholder = [mpsGraph placeholderWithShape:keyShape
                                                                 dataType:mps_dtype
                                                                     name:@"key"];
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Created key placeholder");

          MPSGraphTensor* valuePlaceholder = [mpsGraph placeholderWithShape:valueShape
                                                                   dataType:mps_dtype
                                                                       name:@"value"];
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Created value placeholder");

          MPSGraphTensor* maskTensor = nil;

          // Handle causal mask
          if (is_causal) {
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Creating causal mask");

            // Create a causal mask: lower triangular matrix filled with 0s, upper triangle with -inf
            // Shape should be [qSize, kvSeqLength]
            NSArray<NSNumber*>* maskShape = @[@(qSize), @(kvSeqLength)];

            // Create ones tensor
            MPSGraphTensor* onesTensor = [mpsGraph constantWithScalar:1.0f
                                                                shape:maskShape
                                                             dataType:mps_dtype];
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Created ones tensor for causal mask");

            // Create lower triangular mask (including diagonal)
            MPSGraphTensor* causalMask = [mpsGraph bandPartWithTensor:onesTensor
                                                            numLower:-1
                                                            numUpper:0
                                                                name:@"causal_mask"];
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Created causal mask using bandPartWithTensor");

            // Convert mask to attention weights format: 0 for allowed positions, -inf for masked
            MPSGraphTensor* zerosTensor = [mpsGraph constantWithScalar:0.0f
                                                                 shape:maskShape
                                                              dataType:mps_dtype];

            MPSGraphTensor* negInfTensor = [mpsGraph constantWithScalar:-1e9f
                                                                  shape:maskShape
                                                               dataType:mps_dtype];

            // Select: where causal_mask == 1, use 0.0, else use -inf
            maskTensor = [mpsGraph selectWithPredicateTensor:causalMask
                                         truePredicateTensor:zerosTensor
                                        falsePredicateTensor:negInfTensor
                                                        name:@"causal_mask_final"];
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Created final causal mask using selectWithPredicateTensor");
          }

          // Handle explicit attention mask if provided
          MPSGraphTensor* explicitMaskPlaceholder = nil;
          if (attn_mask && *attn_mask) {
            auto* mask_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(*attn_mask);

            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Adding explicit attention mask");

            // Create mask placeholder
            NSMutableArray<NSNumber*>* maskShapeArray = [NSMutableArray array];
            for (int i = 0; i < mask_tensor->dim(); i++) {
              [maskShapeArray addObject:@(mask_tensor->sizes()[i])];
            }

            explicitMaskPlaceholder = [mpsGraph placeholderWithShape:maskShapeArray
                                                            dataType:mps_dtype
                                                                name:@"attention_mask"];

            if (maskTensor) {
              // Combine causal and explicit masks
              maskTensor = [mpsGraph additionWithPrimaryTensor:maskTensor
                                               secondaryTensor:explicitMaskPlaceholder
                                                          name:@"combined_mask"];
            } else {
              maskTensor = explicitMaskPlaceholder;
            }
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Created explicit mask placeholder");
          }

          // Perform scaled dot product attention using MPSGraph
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Calling scaledDotProductAttentionWithQueryTensor with scale=%f", scale_factor);

          MPSGraphTensor* outputTensor = [mpsGraph scaledDotProductAttentionWithQueryTensor:queryPlaceholder
                                                                                 keyTensor:keyPlaceholder
                                                                               valueTensor:valuePlaceholder
                                                                                maskTensor:maskTensor
                                                                                     scale:scale_factor
                                                                                      name:@"scaled_dot_product_attention"];
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Successfully created SDPA tensor");

          // Create feeds dictionary for graph execution
          NSMutableDictionary* feeds = [NSMutableDictionary dictionary];
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Created feeds dictionary");

          // Create MPSGraphTensorData objects for input tensors
          MPSGraphTensorData* queryData = [[MPSGraphTensorData alloc] initWithMTLBuffer:query_buffer
                                                                                  shape:queryShape
                                                                               dataType:mps_dtype];
          MPSGraphTensorData* keyData = [[MPSGraphTensorData alloc] initWithMTLBuffer:key_buffer
                                                                                shape:keyShape
                                                                             dataType:mps_dtype];
          MPSGraphTensorData* valueData = [[MPSGraphTensorData alloc] initWithMTLBuffer:value_buffer
                                                                                  shape:valueShape
                                                                               dataType:mps_dtype];
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Created MPSGraphTensorData objects for inputs");

          feeds[queryPlaceholder] = queryData;
          feeds[keyPlaceholder] = keyData;
          feeds[valuePlaceholder] = valueData;
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Added input tensors to feeds");

          // Add explicit mask data to feeds if provided
          if (explicitMaskPlaceholder && attn_mask && *attn_mask) {
            auto* mask_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(*attn_mask);
            void* mask_data_ptr = mask_tensor->mutable_data_ptr();

            // Get or create Metal buffer for mask
            id<MTLBuffer> mask_buffer = nullptr;
            auto mask_it = ptr_to_mtl_buffer.find(mask_data_ptr);
            if (mask_it != ptr_to_mtl_buffer.end()) {
              mask_buffer = mask_it->second;
            } else {
              size_t mask_size = mask_tensor->numel() * element_size;
              mask_buffer = [device newBufferWithBytes:mask_data_ptr
                                                length:mask_size
                                               options:MTLResourceStorageModeShared];
              if (!mask_buffer) {
                ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Failed to create Metal buffer for attention mask");
                throw std::runtime_error("Failed to create Metal buffer for attention mask");
              }
            }

            NSMutableArray<NSNumber*>* maskShapeArray = [NSMutableArray array];
            for (int i = 0; i < mask_tensor->dim(); i++) {
              [maskShapeArray addObject:@(mask_tensor->sizes()[i])];
            }

            MPSGraphTensorData* maskData = [[MPSGraphTensorData alloc] initWithMTLBuffer:mask_buffer
                                                                                   shape:maskShapeArray
                                                                                dataType:mps_dtype];
            feeds[explicitMaskPlaceholder] = maskData;
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Added explicit mask tensor to feeds");
          }

          // Create results dictionary
          NSArray<NSNumber*>* outputShape = @[@(batchSize), @(num_heads), @(qSize), @(headSize)];
          MPSGraphTensorData* outputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:out_buffer
                                                                                    shape:outputShape
                                                                                 dataType:mps_dtype];

          NSDictionary* results = @{outputTensor: outputData};
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Created results dictionary");

          // Execute via shared stream and keep results on GPU
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Executing MPSGraph using stream");
          stream->executeMPSGraph(mpsGraph, feeds, results, SyncType::COMMIT_AND_CONTINUE);
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: MPSGraph execution completed successfully");

        } @catch (NSException *exception) {
          ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: NSException caught: %s - %s",
                 [[exception name] UTF8String], [[exception reason] UTF8String]);
          throw std::runtime_error("MPSGraph operation failed with NSException");
        }

        // For attention weights, zero-fill the GPU buffer (shared memory allows CPU memset)
        std::memset(attn_contents_ptr, 0, attn_size_bytes);

        // Create output tensor handles
        AOTITensorHandle out_tensor_handle = nullptr;
        AOTITensorHandle attn_tensor_handle = nullptr;

        AOTITorchError create_out_result = aoti_torch_create_tensor_from_blob_v2(
            out_contents_ptr,
            4,  // ndim
            output_sizes.data(),
            out_strides.data(),
            0,  // storage_offset
            dtype,
            13,  // device_type (MPS)
            0,  // device_index
            &out_tensor_handle,
            0,  // layout (strided)
            nullptr,  // opaque_metadata
            0   // opaque_metadata_size
        );

        AOTITorchError create_attn_result = aoti_torch_create_tensor_from_blob_v2(
            attn_contents_ptr,
            4,  // ndim
            attn_sizes.data(),
            attn_strides.data(),
            0,  // storage_offset
            dtype,
            13,  // device_type (MPS)
            0,  // device_index
            &attn_tensor_handle,
            0,  // layout (strided)
            nullptr,  // opaque_metadata
            0   // opaque_metadata_size
        );

        if (create_out_result != Error::Ok || create_attn_result != Error::Ok ||
            !out_tensor_handle || !attn_tensor_handle) {
          ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Failed to create output tensors");
          aoti_torch_mps_free(out_contents_ptr);
          aoti_torch_mps_free(attn_contents_ptr);
          throw std::runtime_error("Failed to create output tensors");
        }

        // Mark that we own the memory for these tensors
        auto* out_et_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(out_tensor_handle);
        auto* attn_et_tensor = reinterpret_cast<executorch::runtime::etensor::Tensor*>(attn_tensor_handle);
        is_tensor_own_memory[out_et_tensor] = true;
        is_tensor_own_memory[attn_et_tensor] = true;

        // Set output tensor handles
        *ret0 = out_tensor_handle;
        *ret1 = attn_tensor_handle;

      ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: MPSGraph implementation completed successfully");
    }

    return Error::Ok;

  } catch (const std::exception& e) {
    ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps exception: %s", e.what());
    return Error::Internal;
  } catch (...) {
    ET_LOG(Error, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: unknown exception");
    return Error::Internal;
  }
}

} // extern "C"

} // namespace metal
} // namespace backends
} // namespace executorch
