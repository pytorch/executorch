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

AOTITorchError aoti_torch_mps_bmm_out(
    AOTITensorHandle out,
    AOTITensorHandle self,
    AOTITensorHandle mat2) {

  // Validate non-null handles
  if (!out || !self || !mat2) {
    ET_LOG(Error, "aoti_torch_mps_bmm_out: null tensor handles");
    return Error::InvalidArgument;
  }

  @autoreleasepool {
    try {
      // Convert AOTITensorHandle to ExecuTorch tensors
      auto out_tensor = reinterpret_cast<Tensor*>(out);
      auto self_tensor = reinterpret_cast<Tensor*>(self);
      auto mat2_tensor = reinterpret_cast<Tensor*>(mat2);

      // Validate tensor dimensions - bmm requires 3-D tensors
      if (self_tensor->dim() != 3 || mat2_tensor->dim() != 3 || out_tensor->dim() != 3) {
        ET_LOG(Error, "aoti_torch_mps_bmm_out: tensors must be 3-D. "
               "Got self.dim=%zd, mat2.dim=%zd, out.dim=%zd",
               self_tensor->dim(),
               mat2_tensor->dim(),
               out_tensor->dim());
        return Error::InvalidArgument;
      }

      int64_t B = self_tensor->sizes()[0];  // batch size
      int64_t M = self_tensor->sizes()[1];  // rows of self
      int64_t K = self_tensor->sizes()[2];  // cols of self / rows of mat2
      int64_t N = mat2_tensor->sizes()[2];  // cols of mat2

      // Validate shape constraints
      // self: [B, M, K], mat2: [B, K, N], out: [B, M, N]
      if (mat2_tensor->sizes()[0] != B) {
        ET_LOG(Error, "aoti_torch_mps_bmm_out: batch size mismatch. "
               "Expected mat2[0]=%d to match self[0]=%lld. "
               "self.shape=[%lld,%lld,%lld], mat2.shape=[%d,%d,%d]",
               (int)mat2_tensor->sizes()[0], (long long)B,
               (long long)B, (long long)M, (long long)K,
               (int)mat2_tensor->sizes()[0], (int)mat2_tensor->sizes()[1], (int)mat2_tensor->sizes()[2]);
        return Error::InvalidArgument;
      }

      if (mat2_tensor->sizes()[1] != K) {
        ET_LOG(Error, "aoti_torch_mps_bmm_out: incompatible matrix dimensions for bmm. "
               "Expected mat2[1]=%d to match self[2]=%lld. "
               "Cannot multiply [%lld,%lld,%lld] @ [%d,%d,%d]",
               (int)mat2_tensor->sizes()[1], (long long)K,
               (long long)B, (long long)M, (long long)K,
               (int)mat2_tensor->sizes()[0], (int)mat2_tensor->sizes()[1], (int)mat2_tensor->sizes()[2]);
        return Error::InvalidArgument;
      }

      if (out_tensor->sizes()[0] != B || out_tensor->sizes()[1] != M || out_tensor->sizes()[2] != N) {
        ET_LOG(Error, "aoti_torch_mps_bmm_out: output shape mismatch. "
               "Expected out.shape=[%lld,%lld,%lld], got [%d,%d,%d]",
               (long long)B, (long long)M, (long long)N,
               (int)out_tensor->sizes()[0], (int)out_tensor->sizes()[1], (int)out_tensor->sizes()[2]);
        return Error::InvalidArgument;
      }

      // Validate dtype consistency
      int32_t self_dtype = static_cast<int32_t>(self_tensor->scalar_type());
      int32_t mat2_dtype = static_cast<int32_t>(mat2_tensor->scalar_type());
      int32_t out_dtype = static_cast<int32_t>(out_tensor->scalar_type());

      if (self_dtype != mat2_dtype || self_dtype != out_dtype) {
        ET_LOG(Error, "aoti_torch_mps_bmm_out: dtype mismatch. "
               "All tensors must have same dtype. Got self.dtype=%d, mat2.dtype=%d, out.dtype=%d",
               self_dtype, mat2_dtype, out_dtype);
        return Error::InvalidArgument;
      }

      int32_t dtype = self_dtype;

      // Validate layout: BMM requires strictly contiguous 3D tensors
      // For shape [B, M, K], contiguous strides MUST be [M*K, K, 1]
      //
      // Why strict contiguity is required:
      // - MPSGraphTensorData initWithMTLBuffer:shape:dataType: interprets the MTLBuffer
      //   as containing dense row-major data for the given shape
      // - Non-contiguous layouts (transposed, views with strides, etc.) have different
      //   memory layouts that don't match what MPS expects
      // - This would result in SILENT WRONG RESULTS
      // - This is an _out op: we must NOT create implicit copies
      // - Policy: Reject non-contiguous inputs explicitly (transposed/view tensors unsupported)
      //
      // Limitation: This implementation does not explicitly check storage offset (no API available).
      // Tensors with non-zero storage offsets are not explicitly rejected but may work if they
      // happen to have contiguous strides. Users should ensure tensors are base tensors without offsets.
      auto self_strides = self_tensor->strides();
      auto mat2_strides = mat2_tensor->strides();
      auto out_strides = out_tensor->strides();

      // Check self tensor is contiguous [B, M, K] with strides [M*K, K, 1]
      if (self_strides[2] != 1 || self_strides[1] != K || self_strides[0] != M * K) {
        ET_LOG(Error, "aoti_torch_mps_bmm_out: self tensor must be contiguous. "
               "Only dense row-major layout supported; transposed/view tensors are unsupported. "
               "Expected strides=[%lld,%lld,1] for shape=[%lld,%lld,%lld], got strides=[%lld,%lld,%lld].",
               (long long)(M * K), (long long)K, (long long)B, (long long)M, (long long)K,
               self_strides[0], self_strides[1], self_strides[2]);
        return Error::InvalidArgument;
      }

      // Check mat2 tensor is contiguous [B, K, N] with strides [K*N, N, 1]
      if (mat2_strides[2] != 1 || mat2_strides[1] != N || mat2_strides[0] != K * N) {
        ET_LOG(Error, "aoti_torch_mps_bmm_out: mat2 tensor must be contiguous. "
               "Only dense row-major layout supported; transposed/view tensors are unsupported. "
               "Expected strides=[%lld,%lld,1] for shape=[%lld,%lld,%lld], got strides=[%lld,%lld,%lld].",
               (long long)(K * N), (long long)N, (long long)B, (long long)K, (long long)N,
               mat2_strides[0], mat2_strides[1], mat2_strides[2]);
        return Error::InvalidArgument;
      }

      // Check out tensor is contiguous [B, M, N] with strides [M*N, N, 1]
      if (out_strides[2] != 1 || out_strides[1] != N || out_strides[0] != M * N) {
        ET_LOG(Error, "aoti_torch_mps_bmm_out: out tensor must be contiguous. "
               "Only dense row-major layout supported; transposed/view tensors are unsupported. "
               "Expected strides=[%lld,%lld,1] for shape=[%lld,%lld,%lld], got strides=[%lld,%lld,%lld].",
               (long long)(M * N), (long long)N, (long long)B, (long long)M, (long long)N,
               out_strides[0], out_strides[1], out_strides[2]);
        return Error::InvalidArgument;
      }

      // Get Metal stream and device
      ETMetalStream* stream = getCurrentMetalStream();
      if (!stream) {
        ET_LOG(Error, "aoti_torch_mps_bmm_out: Failed to get current Metal stream");
        return Error::Internal;
      }

      // Get Metal buffers for input and output tensors
      id<MTLBuffer> self_buffer = get_mtl_buffer(self_tensor, "aoti_torch_mps_bmm_out", "self");
      id<MTLBuffer> mat2_buffer = get_mtl_buffer(mat2_tensor, "aoti_torch_mps_bmm_out", "mat2");
      id<MTLBuffer> out_buffer = get_mtl_buffer(out_tensor, "aoti_torch_mps_bmm_out", "out");

      // Validate buffers are non-null
      if (!self_buffer || !mat2_buffer || !out_buffer) {
        ET_LOG(Error, "aoti_torch_mps_bmm_out: Failed to get Metal buffers. "
               "self_buffer=%p, mat2_buffer=%p, out_buffer=%p",
               self_buffer, mat2_buffer, out_buffer);
        return Error::Internal;
      }

      // End any existing kernel coalescing to ensure clean state
      // (consistent with mm_out and conv pattern)
      stream->endKernelCoalescing();

      // Map dtype to MPS type and validate support
      // Note: Only FLOAT32 and BFLOAT16 are supported in Metal backend (see utils.h)
      // FLOAT16 is not in SupportedDTypes enum and is not supported
      MPSDataType mps_dtype;

      if (dtype == static_cast<int32_t>(SupportedDTypes::FLOAT32)) {
        mps_dtype = MPSDataTypeFloat32;
      } else if (dtype == static_cast<int32_t>(SupportedDTypes::BFLOAT16)) {
        mps_dtype = MPSDataTypeBFloat16;
      } else {
        ET_LOG(Error, "aoti_torch_mps_bmm_out: Unsupported data type: %d. "
               "Supported types: FLOAT32 (%d), BFLOAT16 (%d)",
               dtype,
               static_cast<int32_t>(SupportedDTypes::FLOAT32),
               static_cast<int32_t>(SupportedDTypes::BFLOAT16));
        return Error::InvalidArgument;
      }

      // Define shapes for graph placeholders and tensor data
      NSArray<NSNumber*>* selfShape = @[@(B), @(M), @(K)];
      NSArray<NSNumber*>* mat2Shape = @[@(B), @(K), @(N)];
      NSArray<NSNumber*>* outShape = @[@(B), @(M), @(N)];

      // Create cache key for this batched matrix multiplication
      // Cache key includes: op_name, shape params {B, M, K, N}, dtype, transpose_flag
      GraphCacheKey cache_key;
      cache_key.op_name = "bmm";
      cache_key.shape_params = {B, M, K, N};
      cache_key.dtype = dtype;
      cache_key.transpose_flag = false;  // BMM has no transpose handling

      // Check if we have a cached graph
      MPSGraph* mpsGraph = nullptr;
      MPSGraphTensor* outputTensor = nil;
      MPSGraphTensor* selfPlaceholder = nil;
      MPSGraphTensor* mat2Placeholder = nil;

      auto cache_it = graph_cache.find(cache_key);
      if (cache_it != graph_cache.end()) {
        // Cache hit - reuse compiled graph and tensor references
        CachedGraph& cached = cache_it->second;
        mpsGraph = cached.graph;
        selfPlaceholder = cached.input1;
        mat2Placeholder = cached.input2;
        outputTensor = cached.output;

        cache_stats.hits++;
        cache_stats.logStats();

      } else {
        // Cache miss - create and compile new graph
        mpsGraph = [MPSGraph new];
        cache_stats.misses++;
        cache_stats.logStats();

        // Create 3D placeholders for batched matrices
        // These represent the logical shapes for the batched matrix multiplication
        selfPlaceholder = [mpsGraph placeholderWithShape:selfShape
                                                dataType:mps_dtype
                                                    name:@"self"];
        mat2Placeholder = [mpsGraph placeholderWithShape:mat2Shape
                                                dataType:mps_dtype
                                                    name:@"mat2"];

        // MPSGraph matrixMultiplication handles batched case natively when given 3D tensors
        // For 3D inputs [B,M,K] @ [B,K,N] -> [B,M,N]
        outputTensor = [mpsGraph matrixMultiplicationWithPrimaryTensor:selfPlaceholder
                                                       secondaryTensor:mat2Placeholder
                                                                  name:@"bmm_result"];

        // Cache the compiled graph and tensor references for reuse
        CachedGraph cached_graph;
        cached_graph.graph = mpsGraph;
        cached_graph.input1 = selfPlaceholder;
        cached_graph.input2 = mat2Placeholder;
        cached_graph.input3 = nil;  // No third input for BMM
        cached_graph.output = outputTensor;
        graph_cache[cache_key] = cached_graph;

      }  // End of cache miss/hit block

      // Create feeds dictionary for graph execution
      NSMutableDictionary* feeds = [NSMutableDictionary dictionary];

      // Create MPSGraphTensorData objects for input tensors
      // These wrap the MTLBuffers with the shape information
      // Initialize to nil for safe cleanup in exception path
      MPSGraphTensorData* selfData = nil;
      MPSGraphTensorData* mat2Data = nil;
      MPSGraphTensorData* outputData = nil;

      selfData = [[MPSGraphTensorData alloc] initWithMTLBuffer:self_buffer
                                                          shape:selfShape
                                                       dataType:mps_dtype];
      mat2Data = [[MPSGraphTensorData alloc] initWithMTLBuffer:mat2_buffer
                                                          shape:mat2Shape
                                                       dataType:mps_dtype];

      feeds[selfPlaceholder] = selfData;
      feeds[mat2Placeholder] = mat2Data;

      // Create output tensor data
      outputData = [[MPSGraphTensorData alloc] initWithMTLBuffer:out_buffer
                                                           shape:outShape
                                                        dataType:mps_dtype];

      // Build results dictionary
      NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
        outputTensor: outputData
      };

      // Execute the batched matrix multiplication
      @try {
        stream->executeMPSGraph(mpsGraph, feeds, results, SyncType::COMMIT);
      } @catch (NSException *exception) {
        ET_LOG(Error, "aoti_torch_mps_bmm_out: NSException caught during executeMPSGraph: %s - %s",
              [[exception name] UTF8String], [[exception reason] UTF8String]);
        // Guard releases against nil
        if (selfData) [selfData release];
        if (mat2Data) [mat2Data release];
        if (outputData) [outputData release];
        return Error::Internal;
      }

      // Release MPSGraphTensorData objects
      [selfData release];
      [mat2Data release];
      [outputData release];

      return Error::Ok;

    } catch (const std::exception& e) {
      ET_LOG(Error, "aoti_torch_mps_bmm_out exception: %s", e.what());
      return Error::Internal;
    } catch (...) {
      ET_LOG(Error, "aoti_torch_mps_bmm_out: unknown exception");
      return Error::Internal;
    }
  }
}


} // extern "C"

} // namespace metal
} // namespace backends
} // namespace executorch
