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

using executorch::runtime::etensor::Tensor;

// Forward declaration of dispatch_sync_with_rethrow from et_metal.mm
void dispatch_sync_with_rethrow(dispatch_queue_t queue, void (^block)());

// Declare the global mapping from et_metal.mm
extern std::unordered_map<void*, id<MTLBuffer>> ptr_to_mtl_buffer;

namespace {

// Helper function to get Metal buffer from the global mapping
static id<MTLBuffer> get_mtl_buffer(Tensor* tensor, const char* op_name, const char* tensor_name) {
  void* data_ptr = tensor->mutable_data_ptr();
  auto it = ptr_to_mtl_buffer.find(data_ptr);
  if (it == ptr_to_mtl_buffer.end()) {
    ET_LOG(Error, "%s: %s tensor not found in Metal buffer mapping", op_name, tensor_name);
    throw std::runtime_error(std::string(tensor_name) + " tensor not found in Metal buffer mapping");
  }
  return it->second;
}

// Helper function to allocate a Metal buffer and register it in the global mapping.
static id<MTLBuffer> allocate_mtl_buffer(void** data_ptr, size_t size_bytes) {
  AOTITorchError malloc_err = aoti_torch_mps_malloc(data_ptr, size_bytes);
  if (malloc_err != Error::Ok) {
    ET_LOG(Error, "allocate_and_register_mtl_buffer: Failed to allocate Metal buffer via aoti_torch_mps_malloc");
    throw std::runtime_error("Failed to allocate output Metal buffer");
  }

  auto it = ptr_to_mtl_buffer.find(*data_ptr);
  if (it == ptr_to_mtl_buffer.end()) {
    ET_LOG(Error, "allocate_and_register_mtl_buffer: aoti_torch_mps_malloc did not register buffer in map");
    throw std::runtime_error("Failed to look up allocated Metal buffer");
  }
  return it->second;
}

} // namespace

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
      auto out_tensor = reinterpret_cast<Tensor*>(out);
      auto self_tensor = reinterpret_cast<Tensor*>(self);
      auto mat2_tensor = reinterpret_cast<Tensor*>(mat2);

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

      // Check if mat2 is transposed (non-contiguous due to transpose)
      // A transposed matrix will have stride(-2) == 1 (column-major instead of row-major)
      // For a 2D tensor with shape [K, N]:
      //   - Contiguous (row-major): strides = [N, 1]
      //   - Transposed (column-major): strides = [1, K]
      bool mat2_is_transposed = false;
      int64_t mat2_stride_0 = mat2_tensor->strides()[0];  // stride for dimension 0
      int64_t mat2_stride_1 = mat2_tensor->strides()[1];  // stride for dimension 1

      // Detect transposed layout: stride(-2) == 1 indicates column-major layout
      if (mat2_stride_0 == 1 && mat2_stride_1 != 1) {
        mat2_is_transposed = true;
        ET_LOG(Debug, "aoti_torch_mps_mm_out: mat2 is transposed (strides=[%lld, %lld])",
               mat2_stride_0, mat2_stride_1);
      } else {
        ET_LOG(Debug, "aoti_torch_mps_mm_out: mat2 is contiguous (strides=[%lld, %lld])",
               mat2_stride_0, mat2_stride_1);
      }

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

      // Get Metal buffers for input and output tensors
      id<MTLBuffer> self_buffer = get_mtl_buffer(self_tensor, "aoti_torch_mps_mm_out", "self");
      id<MTLBuffer> mat2_buffer = get_mtl_buffer(mat2_tensor, "aoti_torch_mps_mm_out", "mat2");
      id<MTLBuffer> out_buffer = get_mtl_buffer(out_tensor, "aoti_torch_mps_mm_out", "out");

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
      NSArray<NSNumber*>* outShape = @[@(M), @(N)];

      // For mat2, we need to handle both contiguous and transposed cases
      // If mat2 is transposed, its physical layout in memory is [N, K] (column-major)
      // but logically we need [K, N] for the matrix multiplication
      NSArray<NSNumber*>* mat2PhysicalShape;
      if (mat2_is_transposed) {
        // Physical shape reflects the actual memory layout (transposed)
        mat2PhysicalShape = @[@(N), @(K)];
        ET_LOG(Debug, "aoti_torch_mps_mm_out: mat2 physical shape (transposed): [%d,%d]", (int)N, (int)K);
      } else {
        // Physical shape is the logical shape (contiguous)
        mat2PhysicalShape = @[@(K), @(N)];
        ET_LOG(Debug, "aoti_torch_mps_mm_out: mat2 physical shape (contiguous): [%d,%d]", (int)K, (int)N);
      }

      ET_LOG(Debug, "aoti_torch_mps_mm_out: Creating placeholders with shapes self:[%d,%d] mat2:[%d,%d]",
             (int)M, (int)K,
             mat2_is_transposed ? (int)N : (int)K,
             mat2_is_transposed ? (int)K : (int)N);

      // Create placeholders for input tensors
      MPSGraphTensor* selfPlaceholder = [mpsGraph placeholderWithShape:selfShape
                                                              dataType:mps_dtype
                                                                  name:@"self"];
      MPSGraphTensor* mat2Placeholder = [mpsGraph placeholderWithShape:mat2PhysicalShape
                                                              dataType:mps_dtype
                                                                  name:@"mat2_physical"];

      ET_LOG(Debug, "aoti_torch_mps_mm_out: Created input placeholders");

      // If mat2 is transposed, apply transpose operation in the graph to get the logical shape
      MPSGraphTensor* mat2Logical;
      if (mat2_is_transposed) {
        // Transpose from physical [N, K] to logical [K, N]
        // MPSGraph transposeTensor swaps the last two dimensions for 2D tensors
        mat2Logical = [mpsGraph transposeTensor:mat2Placeholder
                                      dimension:-2
                                  withDimension:-1
                                           name:@"mat2_transposed"];
        ET_LOG(Debug, "aoti_torch_mps_mm_out: Applied transpose operation to mat2 in graph");
      } else {
        // No transpose needed, use placeholder directly
        mat2Logical = mat2Placeholder;
        ET_LOG(Debug, "aoti_torch_mps_mm_out: Using mat2 placeholder directly (no transpose needed)");
      }

      // Perform matrix multiplication using MPSGraph with the logical mat2 tensor
      MPSGraphTensor* mmOutput = [mpsGraph matrixMultiplicationWithPrimaryTensor:selfPlaceholder
                                                                 secondaryTensor:mat2Logical
                                                                            name:@"matrix_multiplication"];

      ET_LOG(Debug, "aoti_torch_mps_mm_out: Successfully created matrix multiplication tensor");

      // Create feeds dictionary for graph execution
      NSMutableDictionary* feeds = [NSMutableDictionary dictionary];

      // Create MPSGraphTensorData objects for input tensors
      // Use physical shapes to match how data is actually laid out in memory
      MPSGraphTensorData* selfData = [[MPSGraphTensorData alloc] initWithMTLBuffer:self_buffer
                                                                              shape:selfShape
                                                                           dataType:mps_dtype];
      MPSGraphTensorData* mat2Data = [[MPSGraphTensorData alloc] initWithMTLBuffer:mat2_buffer
                                                                              shape:mat2PhysicalShape
                                                                           dataType:mps_dtype];

      feeds[selfPlaceholder] = selfData;
      feeds[mat2Placeholder] = mat2Data;

      ET_LOG(Debug, "aoti_torch_mps_mm_out: Created feeds dictionary with physical shapes");

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
        stream->executeMPSGraph(mpsGraph, feeds, results, SyncType::COMMIT);
      } @catch (NSException *exception) {
        ET_LOG(Error, "aoti_torch_mps_mm_out: NSException caught during executeMPSGraph: %s - %s",
              [[exception name] UTF8String], [[exception reason] UTF8String]);
        throw std::runtime_error("MPSGraph execution failed with NSException");
      }

      ET_LOG(Debug, "aoti_torch_mps_mm_out: MPSGraph execution completed successfully");

      // Release MPSGraph to prevent memory leak
      [mpsGraph release];
      mpsGraph = nil;

      [selfData release];
      [mat2Data release];
      [outputData release];

      ET_LOG(Debug, "aoti_torch_mps_mm_out: Executed successfully");
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
      auto input_tensor = reinterpret_cast<Tensor*>(input);
      auto weight_tensor = reinterpret_cast<Tensor*>(weight);

      // bias can be null for convolutions without bias
      Tensor* bias_tensor = nullptr;
      if (bias && *bias) {
        bias_tensor = reinterpret_cast<Tensor*>(*bias);
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
      }

      // Create feeds dictionary for graph execution
      NSMutableDictionary* feeds = [NSMutableDictionary dictionary];

      // Get Metal buffers from tensors
      id<MTLBuffer> input_buffer = get_mtl_buffer(input_tensor, "aoti_torch_mps_convolution", "input");
      id<MTLBuffer> weight_buffer = get_mtl_buffer(weight_tensor, "aoti_torch_mps_convolution", "weight");

      ET_LOG(Debug, "aoti_torch_mps_convolution: Using existing Metal buffers - input=%p, weight=%p",
              input_buffer, weight_buffer);

      // Create MPSGraphTensorData objects for input tensors
      MPSGraphTensorData* inputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:input_buffer
                                                                                shape:inputShape
                                                                            dataType:mps_dtype];
      MPSGraphTensorData* weightData = [[MPSGraphTensorData alloc] initWithMTLBuffer:weight_buffer
                                                                                shape:weightShape
                                                                            dataType:mps_dtype];

      feeds[inputPlaceholder] = inputData;
      feeds[weightPlaceholder] = weightData;

      MPSGraphTensorData* biasData = nil;

      // Add bias data to feeds if provided
      if (bias_tensor && biasPlaceholder) {
        id<MTLBuffer> bias_buffer = get_mtl_buffer(bias_tensor, "aoti_torch_mps_convolution", "bias");

        NSArray<NSNumber*>* biasShape = @[@(C_out)];
        biasData = [[MPSGraphTensorData alloc] initWithMTLBuffer:bias_buffer
                                                           shape:biasShape
                                                        dataType:mps_dtype];

        feeds[biasPlaceholder] = biasData;
        ET_LOG(Debug, "aoti_torch_mps_convolution: Added bias tensor to feeds");
      }

      ET_LOG(Debug, "aoti_torch_mps_convolution: Created feeds dictionary");

      // Create Metal buffer for output tensor
      size_t output_size_bytes = N * C_out * H_out * W_out * element_size;
      void* output_contents_ptr = nullptr;
      id<MTLBuffer> output_buffer = allocate_mtl_buffer(&output_contents_ptr, output_size_bytes);

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
        stream->executeMPSGraph(mpsGraph, feeds, results, SyncType::COMMIT);
      } @catch (NSException *exception) {
        ET_LOG(Error, "aoti_torch_mps_convolution: NSException caught during executeMPSGraph: %s - %s",
              [[exception name] UTF8String], [[exception reason] UTF8String]);
        throw std::runtime_error("MPSGraph execution failed with NSException");
      } @catch (...) {
        ET_LOG(Error, "aoti_torch_mps_convolution: MPSGraph execution failed");
        throw std::runtime_error("MPSGraph execution failed");
      }

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
      auto* et_tensor = reinterpret_cast<Tensor*>(output_tensor_handle);
      size_t actual_numel = et_tensor->numel();
      size_t expected_numel = static_cast<size_t>(N * C_out * H_out * W_out);

      if (actual_numel != expected_numel) {
        ET_LOG(Error, "aoti_torch_mps_convolution: Tensor size mismatch. Expected %zu, got %zu", expected_numel, actual_numel);
        aoti_torch_mps_free(tensor_data);  // Free the allocated GPU memory on failure
        throw std::runtime_error("Tensor size mismatch");
      }

      // Store the tensor handle - mark that we own the memory since we manually allocated it
      *ret0 = output_tensor_handle;
      // Note: memory_to_n_tensor is managed automatically in aoti_torch_create_tensor_from_blob_v2
      // The function sets it to NOT_OWN, but we need to change it to 1 since we allocated it
      extern std::unordered_map<void*, int32_t> memory_to_n_tensor;
      memory_to_n_tensor[tensor_data] = 1;

      // Release MPSGraph to prevent memory leak
      [mpsGraph release];
      mpsGraph = nil;

      [inputData release];
      [weightData release];
      if (biasData) [biasData release];
      [outputData release];

      ET_LOG(Debug, "aoti_torch_mps_convolution: Created output tensor with %zu elements using MPSGraph", actual_numel);

      ET_LOG(Debug, "aoti_torch_mps_convolution: Executed successfully");
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
      auto* query_tensor = reinterpret_cast<Tensor*>(query);
      auto* key_tensor = reinterpret_cast<Tensor*>(key);
      auto* value_tensor = reinterpret_cast<Tensor*>(value);

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

        // Detect non-contiguous layouts for query, key, and value tensors
        // For a 4D tensor [batch, num_heads, seq_len, head_dim], common non-contiguous patterns:
        // - Transposed last 2 dims (dims 2,3): strides[2] == 1 && strides[3] == seq_len (seq_len and head_dim swapped)
        // - Transposed internal dims (dims 1,2): strides[1] == head_dim && strides[2] == num_heads*head_dim (num_heads and seq_len swapped)
        // - Other permutations may exist depending on upstream operations

        bool query_is_transposed_last2 = false;   // transpose of dims -2 and -1
        bool query_is_transposed_internal = false; // transpose of dims 1 and 2
        bool key_is_transposed_last2 = false;
        bool key_is_transposed_internal = false;
        bool value_is_transposed_last2 = false;
        bool value_is_transposed_internal = false;

        // Expected contiguous strides for query [batch, num_heads, qSize, headSize]
        int64_t expected_q_stride_3 = 1;
        int64_t expected_q_stride_2 = headSize;
        int64_t expected_q_stride_1 = qSize * headSize;
        int64_t expected_q_stride_0 = num_heads * qSize * headSize;

        // Check query tensor layout
        auto q_strides = query_tensor->strides();
        if (q_strides[3] != expected_q_stride_3 || q_strides[2] != expected_q_stride_2 ||
            q_strides[1] != expected_q_stride_1) {
          // Check if it's a transpose of the last two dimensions (dims 2 and 3)
          if (q_strides[2] == 1 && q_strides[3] == qSize && q_strides[1] == qSize * headSize) {
            query_is_transposed_last2 = true;
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Query tensor has transposed last 2 dims (dims 2,3) (strides=[%lld,%lld,%lld,%lld])",
                   (int64_t)q_strides[0], (int64_t)q_strides[1], (int64_t)q_strides[2], (int64_t)q_strides[3]);
          }
          // Check if it's a transpose of the internal dimensions (dims 1 and 2)
          else if (q_strides[1] == headSize && q_strides[2] == num_heads * headSize && q_strides[3] == 1) {
            query_is_transposed_internal = true;
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Query tensor has transposed internal dims (dims 1,2) (strides=[%lld,%lld,%lld,%lld])",
                   (int64_t)q_strides[0], (int64_t)q_strides[1], (int64_t)q_strides[2], (int64_t)q_strides[3]);
          } else {
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Query tensor is non-contiguous with unusual layout (strides=[%lld,%lld,%lld,%lld])",
                   (int64_t)q_strides[0], (int64_t)q_strides[1], (int64_t)q_strides[2], (int64_t)q_strides[3]);
          }
        } else {
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Query tensor is contiguous (strides=[%lld,%lld,%lld,%lld])",
                 (int64_t)q_strides[0], (int64_t)q_strides[1], (int64_t)q_strides[2], (int64_t)q_strides[3]);
        }

        // Expected contiguous strides for key [batch, num_heads, kvSeqLength, headSize]
        int64_t expected_k_stride_3 = 1;
        int64_t expected_k_stride_2 = headSize;
        int64_t expected_k_stride_1 = kvSeqLength * headSize;
        int64_t expected_k_stride_0 = num_heads * kvSeqLength * headSize;

        // Check key tensor layout
        auto k_strides = key_tensor->strides();
        if (k_strides[3] != expected_k_stride_3 || k_strides[2] != expected_k_stride_2 ||
            k_strides[1] != expected_k_stride_1) {
          // Check if it's a transpose of the last two dimensions (dims 2 and 3)
          if (k_strides[2] == 1 && k_strides[3] == kvSeqLength && k_strides[1] == kvSeqLength * headSize) {
            key_is_transposed_last2 = true;
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Key tensor has transposed last 2 dims (dims 2,3) (strides=[%lld,%lld,%lld,%lld])",
                   (int64_t)k_strides[0], (int64_t)k_strides[1], (int64_t)k_strides[2], (int64_t)k_strides[3]);
          }
          // Check if it's a transpose of the internal dimensions (dims 1 and 2)
          else if (k_strides[1] == headSize && k_strides[2] == num_heads * headSize && k_strides[3] == 1) {
            key_is_transposed_internal = true;
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Key tensor has transposed internal dims (dims 1,2) (strides=[%lld,%lld,%lld,%lld])",
                   (int64_t)k_strides[0], (int64_t)k_strides[1], (int64_t)k_strides[2], (int64_t)k_strides[3]);
          } else {
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Key tensor is non-contiguous with unusual layout (strides=[%lld,%lld,%lld,%lld])",
                   (int64_t)k_strides[0], (int64_t)k_strides[1], (int64_t)k_strides[2], (int64_t)k_strides[3]);
          }
        } else {
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Key tensor is contiguous (strides=[%lld,%lld,%lld,%lld])",
                 (int64_t)k_strides[0], (int64_t)k_strides[1], (int64_t)k_strides[2], (int64_t)k_strides[3]);
        }

        // Expected contiguous strides for value [batch, num_heads, kvSeqLength, headSize]
        int64_t expected_v_stride_3 = 1;
        int64_t expected_v_stride_2 = headSize;
        int64_t expected_v_stride_1 = kvSeqLength * headSize;
        int64_t expected_v_stride_0 = num_heads * kvSeqLength * headSize;

        // Check value tensor layout
        auto v_strides = value_tensor->strides();
        if (v_strides[3] != expected_v_stride_3 || v_strides[2] != expected_v_stride_2 ||
            v_strides[1] != expected_v_stride_1) {
          // Check if it's a transpose of the last two dimensions (dims 2 and 3)
          if (v_strides[2] == 1 && v_strides[3] == kvSeqLength && v_strides[1] == kvSeqLength * headSize) {
            value_is_transposed_last2 = true;
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Value tensor has transposed last 2 dims (dims 2,3) (strides=[%lld,%lld,%lld,%lld])",
                   (int64_t)v_strides[0], (int64_t)v_strides[1], (int64_t)v_strides[2], (int64_t)v_strides[3]);
          }
          // Check if it's a transpose of the internal dimensions (dims 1 and 2)
          else if (v_strides[1] == headSize && v_strides[2] == num_heads * headSize && v_strides[3] == 1) {
            value_is_transposed_internal = true;
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Value tensor has transposed internal dims (dims 1,2) (strides=[%lld,%lld,%lld,%lld])",
                   (int64_t)v_strides[0], (int64_t)v_strides[1], (int64_t)v_strides[2], (int64_t)v_strides[3]);
          } else {
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Value tensor is non-contiguous with unusual layout (strides=[%lld,%lld,%lld,%lld])",
                   (int64_t)v_strides[0], (int64_t)v_strides[1], (int64_t)v_strides[2], (int64_t)v_strides[3]);
          }
        } else {
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Value tensor is contiguous (strides=[%lld,%lld,%lld,%lld])",
                 (int64_t)v_strides[0], (int64_t)v_strides[1], (int64_t)v_strides[2], (int64_t)v_strides[3]);
        }

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

        // Get Metal buffers for query, key and value tensors
        id<MTLBuffer> query_buffer = get_mtl_buffer(query_tensor, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps", "query");
        id<MTLBuffer> key_buffer = get_mtl_buffer(key_tensor, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps", "key");
        id<MTLBuffer> value_buffer = get_mtl_buffer(value_tensor, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps", "value");

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
        id<MTLBuffer> out_buffer = allocate_mtl_buffer(&out_contents_ptr, out_size_bytes);

        void* attn_contents_ptr = nullptr;
        id<MTLBuffer> attn_weights_buffer = allocate_mtl_buffer(&attn_contents_ptr, attn_size_bytes);

        // End any existing kernel coalescing to ensure a clean state for MPS
        stream->endKernelCoalescing();

        // Method 1: Using MPSGraph scaledDotProductAttention API - with detailed error handling
        ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Implementing using MPSGraph scaledDotProductAttention");

        @try {
          // Create MPSGraph for scaled dot product attention
          MPSGraph* mpsGraph = [MPSGraph new];
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Created MPSGraph instance");

          // Define physical tensor shapes for placeholders (matching actual memory layout)
          // Two transpose patterns supported:
          // 1. Last 2 dims transposed (dims 2,3): [batch, num_heads, head_dim, seq_len]
          // 2. Internal dims transposed (dims 1,2): [batch, seq_len, num_heads, head_dim]
          NSArray<NSNumber*>* queryPhysicalShape;
          NSArray<NSNumber*>* keyPhysicalShape;
          NSArray<NSNumber*>* valuePhysicalShape;

          if (query_is_transposed_last2) {
            // Physical layout: [batch, num_heads, headSize, qSize] (dims 2,3 swapped)
            queryPhysicalShape = @[@(batchSize), @(num_heads), @(headSize), @(qSize)];
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Query physical shape (transposed dims 2,3): [%d,%d,%d,%d]",
                   (int)batchSize, (int)num_heads, (int)headSize, (int)qSize);
          } else if (query_is_transposed_internal) {
            // Physical layout: [batch, qSize, num_heads, headSize] (dims 1,2 swapped)
            queryPhysicalShape = @[@(batchSize), @(qSize), @(num_heads), @(headSize)];
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Query physical shape (transposed dims 1,2): [%d,%d,%d,%d]",
                   (int)batchSize, (int)qSize, (int)num_heads, (int)headSize);
          } else {
            // Physical layout matches logical layout: [batch, num_heads, qSize, headSize]
            queryPhysicalShape = @[@(batchSize), @(num_heads), @(qSize), @(headSize)];
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Query physical shape (contiguous): [%d,%d,%d,%d]",
                   (int)batchSize, (int)num_heads, (int)qSize, (int)headSize);
          }

          if (key_is_transposed_last2) {
            // Physical layout: [batch, num_heads, headSize, kvSeqLength] (dims 2,3 swapped)
            keyPhysicalShape = @[@(batchSize), @(num_heads), @(headSize), @(kvSeqLength)];
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Key physical shape (transposed dims 2,3): [%d,%d,%d,%d]",
                   (int)batchSize, (int)num_heads, (int)headSize, (int)kvSeqLength);
          } else if (key_is_transposed_internal) {
            // Physical layout: [batch, kvSeqLength, num_heads, headSize] (dims 1,2 swapped)
            keyPhysicalShape = @[@(batchSize), @(kvSeqLength), @(num_heads), @(headSize)];
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Key physical shape (transposed dims 1,2): [%d,%d,%d,%d]",
                   (int)batchSize, (int)kvSeqLength, (int)num_heads, (int)headSize);
          } else {
            // Physical layout matches logical layout: [batch, num_heads, kvSeqLength, headSize]
            keyPhysicalShape = @[@(batchSize), @(num_heads), @(kvSeqLength), @(headSize)];
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Key physical shape (contiguous): [%d,%d,%d,%d]",
                   (int)batchSize, (int)num_heads, (int)kvSeqLength, (int)headSize);
          }

          if (value_is_transposed_last2) {
            // Physical layout: [batch, num_heads, headSize, kvSeqLength] (dims 2,3 swapped)
            valuePhysicalShape = @[@(batchSize), @(num_heads), @(headSize), @(kvSeqLength)];
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Value physical shape (transposed dims 2,3): [%d,%d,%d,%d]",
                   (int)batchSize, (int)num_heads, (int)headSize, (int)kvSeqLength);
          } else if (value_is_transposed_internal) {
            // Physical layout: [batch, kvSeqLength, num_heads, headSize] (dims 1,2 swapped)
            valuePhysicalShape = @[@(batchSize), @(kvSeqLength), @(num_heads), @(headSize)];
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Value physical shape (transposed dims 1,2): [%d,%d,%d,%d]",
                   (int)batchSize, (int)kvSeqLength, (int)num_heads, (int)headSize);
          } else {
            // Physical layout matches logical layout: [batch, num_heads, kvSeqLength, headSize]
            valuePhysicalShape = @[@(batchSize), @(num_heads), @(kvSeqLength), @(headSize)];
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Value physical shape (contiguous): [%d,%d,%d,%d]",
                   (int)batchSize, (int)num_heads, (int)kvSeqLength, (int)headSize);
          }

          // Create placeholders for input tensors with physical shapes
          MPSGraphTensor* queryPlaceholder = [mpsGraph placeholderWithShape:queryPhysicalShape
                                                                   dataType:mps_dtype
                                                                       name:@"query_physical"];
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Created query placeholder");

          MPSGraphTensor* keyPlaceholder = [mpsGraph placeholderWithShape:keyPhysicalShape
                                                                 dataType:mps_dtype
                                                                     name:@"key_physical"];
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Created key placeholder");

          MPSGraphTensor* valuePlaceholder = [mpsGraph placeholderWithShape:valuePhysicalShape
                                                                   dataType:mps_dtype
                                                                       name:@"value_physical"];
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Created value placeholder");

          // Apply transpose operations in the graph to convert physical to logical layout
          // Logical shapes needed for SDPA: Q[batch, num_heads, qSize, headSize],
          //                                 K[batch, num_heads, kvSeqLength, headSize],
          //                                 V[batch, num_heads, kvSeqLength, headSize]
          MPSGraphTensor* queryLogical;
          MPSGraphTensor* keyLogical;
          MPSGraphTensor* valueLogical;

          if (query_is_transposed_last2) {
            // Transpose dims 2,3: [batch, num_heads, headSize, qSize] → [batch, num_heads, qSize, headSize]
            queryLogical = [mpsGraph transposeTensor:queryPlaceholder
                                           dimension:-2
                                       withDimension:-1
                                                name:@"query_transposed_last2"];
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Applied transpose (dims 2,3) to query tensor in graph");
          } else if (query_is_transposed_internal) {
            // Transpose dims 1,2: [batch, qSize, num_heads, headSize] → [batch, num_heads, qSize, headSize]
            queryLogical = [mpsGraph transposeTensor:queryPlaceholder
                                           dimension:1
                                       withDimension:2
                                                name:@"query_transposed_internal"];
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Applied transpose (dims 1,2) to query tensor in graph");
          } else {
            queryLogical = queryPlaceholder;
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Using query placeholder directly (no transpose needed)");
          }

          if (key_is_transposed_last2) {
            // Transpose dims 2,3: [batch, num_heads, headSize, kvSeqLength] → [batch, num_heads, kvSeqLength, headSize]
            keyLogical = [mpsGraph transposeTensor:keyPlaceholder
                                         dimension:-2
                                     withDimension:-1
                                              name:@"key_transposed_last2"];
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Applied transpose (dims 2,3) to key tensor in graph");
          } else if (key_is_transposed_internal) {
            // Transpose dims 1,2: [batch, kvSeqLength, num_heads, headSize] → [batch, num_heads, kvSeqLength, headSize]
            keyLogical = [mpsGraph transposeTensor:keyPlaceholder
                                         dimension:1
                                     withDimension:2
                                              name:@"key_transposed_internal"];
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Applied transpose (dims 1,2) to key tensor in graph");
          } else {
            keyLogical = keyPlaceholder;
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Using key placeholder directly (no transpose needed)");
          }

          if (value_is_transposed_last2) {
            // Transpose dims 2,3: [batch, num_heads, headSize, kvSeqLength] → [batch, num_heads, kvSeqLength, headSize]
            valueLogical = [mpsGraph transposeTensor:valuePlaceholder
                                           dimension:-2
                                       withDimension:-1
                                                name:@"value_transposed_last2"];
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Applied transpose (dims 2,3) to value tensor in graph");
          } else if (value_is_transposed_internal) {
            // Transpose dims 1,2: [batch, kvSeqLength, num_heads, headSize] → [batch, num_heads, kvSeqLength, headSize]
            valueLogical = [mpsGraph transposeTensor:valuePlaceholder
                                           dimension:1
                                       withDimension:2
                                                name:@"value_transposed_internal"];
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Applied transpose (dims 1,2) to value tensor in graph");
          } else {
            valueLogical = valuePlaceholder;
            ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Using value placeholder directly (no transpose needed)");
          }

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
            auto* mask_tensor = reinterpret_cast<Tensor*>(*attn_mask);

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

          // Perform scaled dot product attention using MPSGraph with logical (possibly transposed) tensors
          // The logical tensors have the correct shapes for attention computation regardless of input memory layout
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Calling scaledDotProductAttentionWithQueryTensor with scale=%f", scale_factor);

          MPSGraphTensor* outputTensor = [mpsGraph scaledDotProductAttentionWithQueryTensor:queryLogical
                                                                                 keyTensor:keyLogical
                                                                               valueTensor:valueLogical
                                                                                maskTensor:maskTensor
                                                                                     scale:scale_factor
                                                                                      name:@"scaled_dot_product_attention"];
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Successfully created SDPA tensor");

          // Create feeds dictionary for graph execution
          NSMutableDictionary* feeds = [NSMutableDictionary dictionary];
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Created feeds dictionary");

          // Create MPSGraphTensorData objects for input tensors using physical shapes
          // Physical shapes match the actual memory layout of the tensors
          MPSGraphTensorData* queryData = [[MPSGraphTensorData alloc] initWithMTLBuffer:query_buffer
                                                                                  shape:queryPhysicalShape
                                                                               dataType:mps_dtype];
          MPSGraphTensorData* keyData = [[MPSGraphTensorData alloc] initWithMTLBuffer:key_buffer
                                                                                shape:keyPhysicalShape
                                                                             dataType:mps_dtype];
          MPSGraphTensorData* valueData = [[MPSGraphTensorData alloc] initWithMTLBuffer:value_buffer
                                                                                  shape:valuePhysicalShape
                                                                               dataType:mps_dtype];
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Created MPSGraphTensorData objects with physical shapes");

          feeds[queryPlaceholder] = queryData;
          feeds[keyPlaceholder] = keyData;
          feeds[valuePlaceholder] = valueData;
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Added input tensors to feeds");

          MPSGraphTensorData* maskData = nil;

          // Add explicit mask data to feeds if provided
          if (explicitMaskPlaceholder && attn_mask && *attn_mask) {
            auto* mask_tensor = reinterpret_cast<Tensor*>(*attn_mask);
            // Get Metal buffer for mask
            id<MTLBuffer> mask_buffer = get_mtl_buffer(mask_tensor, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps", "mask");

            NSMutableArray<NSNumber*>* maskShapeArray = [NSMutableArray array];
            for (int i = 0; i < mask_tensor->dim(); i++) {
              [maskShapeArray addObject:@(mask_tensor->sizes()[i])];
            }

            maskData = [[MPSGraphTensorData alloc] initWithMTLBuffer:mask_buffer
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
          stream->executeMPSGraph(mpsGraph, feeds, results, SyncType::COMMIT);
          ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: MPSGraph execution completed successfully");

          // Release MPSGraph to prevent memory leak
          [mpsGraph release];
          mpsGraph = nil;

          [queryData release];
          [keyData release];
          [valueData release];
          if (maskData) [maskData release];
          [outputData release];

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
        // Note: memory_to_n_tensor is managed automatically in aoti_torch_create_tensor_from_blob_v2
        // The function sets it to NOT_OWN, but we need to change it to 1 since we allocated it
        extern std::unordered_map<void*, int32_t> memory_to_n_tensor;
        memory_to_n_tensor[out_contents_ptr] = 1;
        memory_to_n_tensor[attn_contents_ptr] = 1;

        // Set output tensor handles
        *ret0 = out_tensor_handle;
        *ret1 = attn_tensor_handle;

      ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: MPSGraph implementation completed successfully");
    }

    ET_LOG(Debug, "aoti_torch_mps__scaled_dot_product_attention_math_for_mps: Executed successfully");
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
