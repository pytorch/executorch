/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/apple/metal/runtime/ops/common.h>

namespace executorch {
namespace backends {
namespace metal {

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
      // Convert AOTITensorHandle to ExecuTorch tensors
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

      // Define tensor shapes for placeholders (needed for both cache hit and miss)
      NSArray<NSNumber*>* selfShape = @[@(M), @(K)];

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

      // Create cache key for this matrix multiplication
      GraphCacheKey cache_key;
      cache_key.op_name = "mm";
      cache_key.shape_params = {M, K, N};
      cache_key.dtype = dtype;
      cache_key.transpose_flag = mat2_is_transposed;

      // Check if we have a cached graph
      MPSGraph* mpsGraph = nullptr;
      MPSGraphTensor* mmOutput = nil;
      MPSGraphTensor* selfPlaceholder = nil;
      MPSGraphTensor* mat2Placeholder = nil;

      auto cache_it = graph_cache.find(cache_key);
      if (cache_it != graph_cache.end()) {
        // Cache hit - reuse compiled graph and tensor references
        CachedGraph& cached = cache_it->second;
        mpsGraph = cached.graph;
        selfPlaceholder = cached.input1;
        mat2Placeholder = cached.input2;
        mmOutput = cached.output;

        cache_stats.hits++;
        cache_stats.logStats();
        ET_LOG(Debug, "aoti_torch_mps_mm_out: Using cached MPSGraph (cache hit, %zu total hits)", cache_stats.hits);

      } else {
        // Cache miss - create and compile new graph
        mpsGraph = [MPSGraph new];
        cache_stats.misses++;
        cache_stats.logStats();
        ET_LOG(Debug, "aoti_torch_mps_mm_out: Created new MPSGraph instance (cache miss, %zu total misses)", cache_stats.misses);

        ET_LOG(Debug, "aoti_torch_mps_mm_out: Creating placeholders with shapes self:[%d,%d] mat2:[%d,%d]",
                (int)M, (int)K,
                mat2_is_transposed ? (int)N : (int)K,
                mat2_is_transposed ? (int)K : (int)N);

        // Create placeholders for input tensors
        selfPlaceholder = [mpsGraph placeholderWithShape:selfShape
                                                dataType:mps_dtype
                                                    name:@"self"];
        mat2Placeholder = [mpsGraph placeholderWithShape:mat2PhysicalShape
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
        mmOutput = [mpsGraph matrixMultiplicationWithPrimaryTensor:selfPlaceholder
                                                                    secondaryTensor:mat2Logical
                                                                              name:@"matrix_multiplication"];

        ET_LOG(Debug, "aoti_torch_mps_mm_out: Successfully created matrix multiplication tensor");

        // Cache the compiled graph and tensor references for reuse
        CachedGraph cached_graph;
        cached_graph.graph = mpsGraph;
        cached_graph.input1 = selfPlaceholder;
        cached_graph.input2 = mat2Placeholder;
        cached_graph.input3 = nil;
        cached_graph.output = mmOutput;
        graph_cache[cache_key] = cached_graph;

        ET_LOG(Debug, "aoti_torch_mps_mm_out: Cached compiled MPSGraph for future reuse");
      }  // End of cache miss/hit block

      // Define output shape
      NSArray<NSNumber*>* outShape = @[@(M), @(N)];

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

} // extern "C"

} // namespace metal
} // namespace backends
} // namespace executorch
