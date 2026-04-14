/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Top-k operator using MPSGraph.
// Used by MoE routing (torch.topk in SparseMoE.forward).

#include <executorch/backends/apple/metal/runtime/ops/common.h>

namespace executorch {
namespace backends {
namespace metal {

extern "C" {

AOTITorchError aoti_torch_mps_topk(
    AOTITensorHandle self,
    int64_t k,
    int64_t dim,
    int32_t largest,
    int32_t sorted,
    AOTITensorHandle* ret0,   // values
    AOTITensorHandle* ret1) { // indices

  ET_LOG(Debug, "aoti_torch_mps_topk: k=%lld, dim=%lld, largest=%d, sorted=%d",
         k, dim, largest, sorted);

  if (!self || !ret0 || !ret1) {
    ET_LOG(Error, "aoti_torch_mps_topk: null tensor handles");
    return Error::InvalidArgument;
  }

  ETMetalStream* stream = getCurrentMetalStream();
  if (!stream) {
    ET_LOG(Error, "aoti_torch_mps_topk: Failed to get Metal stream");
    return Error::Internal;
  }

  try {
    @autoreleasepool {
      auto* self_tensor = reinterpret_cast<Tensor*>(self);

      int64_t ndim = self_tensor->dim();
      if (dim < 0) {
        dim += ndim;
      }
      if (dim < 0 || dim >= ndim) {
        ET_LOG(Error, "aoti_torch_mps_topk: invalid dim");
        return Error::InvalidArgument;
      }

      int64_t dim_size = self_tensor->sizes()[dim];
      if (k > dim_size) {
        ET_LOG(Error, "aoti_torch_mps_topk: k=%lld > dim_size=%lld\n", k, dim_size);
        return Error::InvalidArgument;
      }

      // Determine dtype
      int32_t dtype = static_cast<int32_t>(self_tensor->scalar_type());
      size_t element_size;
      MPSDataType mps_dtype;

      if (dtype == static_cast<int32_t>(SupportedDTypes::FLOAT32)) {
        element_size = sizeof(float);
        mps_dtype = MPSDataTypeFloat32;
      } else if (dtype == static_cast<int32_t>(SupportedDTypes::BFLOAT16)) {
        element_size = sizeof(uint16_t);
        mps_dtype = MPSDataTypeBFloat16;
      } else {
        ET_LOG(Error, "aoti_torch_mps_topk: Unsupported dtype %d", dtype);
        return Error::InvalidArgument;
      }

      // Build output shape: same as input but with dim replaced by k
      std::vector<int64_t> out_sizes;
      for (int64_t i = 0; i < ndim; i++) {
        out_sizes.push_back(i == dim ? k : self_tensor->sizes()[i]);
      }

      // Compute strides (contiguous)
      std::vector<int64_t> out_strides(ndim);
      out_strides[ndim - 1] = 1;
      for (int64_t i = ndim - 2; i >= 0; i--) {
        out_strides[i] = out_strides[i + 1] * out_sizes[i + 1];
      }

      // Total elements
      size_t num_elements = 1;
      for (auto s : out_sizes) num_elements *= s;

      // Allocate output buffers
      size_t values_bytes = num_elements * element_size;
      size_t indices_bytes = num_elements * sizeof(int32_t);

      void* values_ptr = nullptr;
      void* indices_ptr = nullptr;
      allocate_mtl_buffer(&values_ptr, values_bytes);
      allocate_mtl_buffer(&indices_ptr, indices_bytes);

      // Build MPSGraph
      // Convert input shape to NSArray<NSNumber*>
      NSMutableArray<NSNumber*>* input_shape = [NSMutableArray arrayWithCapacity:ndim];
      for (int64_t i = 0; i < ndim; i++) {
        [input_shape addObject:@(self_tensor->sizes()[i])];
      }

      // Check graph cache
      GraphCacheKey cache_key;
      cache_key.op_name = "topk";
      cache_key.shape_params.push_back(k);
      cache_key.shape_params.push_back(dim);
      cache_key.shape_params.push_back(largest);
      for (int64_t i = 0; i < ndim; i++) {
        cache_key.shape_params.push_back(self_tensor->sizes()[i]);
      }
      cache_key.dtype = dtype;
      cache_key.transpose_flag = false;

      auto cache_it = graph_cache.find(cache_key);
      if (cache_it != graph_cache.end()) {
        cache_stats.hits++;
        auto& cached = cache_it->second;

        id<MTLBuffer> self_buffer = get_mtl_buffer(self_tensor, "topk", "self");
        id<MTLBuffer> values_buffer = ptr_to_mtl_buffer[values_ptr];
        id<MTLBuffer> indices_buffer = ptr_to_mtl_buffer[indices_ptr];

        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
          cached.input1: [[MPSGraphTensorData alloc] initWithMTLBuffer:self_buffer shape:input_shape dataType:mps_dtype],
        };

        NSMutableArray<NSNumber*>* out_ns_shape = [NSMutableArray arrayWithCapacity:ndim];
        for (int64_t i = 0; i < ndim; i++) {
          [out_ns_shape addObject:@(out_sizes[i])];
        }

        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
          cached.output: [[MPSGraphTensorData alloc] initWithMTLBuffer:values_buffer shape:out_ns_shape dataType:mps_dtype],
          cached.input2: [[MPSGraphTensorData alloc] initWithMTLBuffer:indices_buffer shape:out_ns_shape dataType:MPSDataTypeInt32],
        };

        stream->executeMPSGraph(cached.graph, feeds, results, SyncType::COMMIT);
      } else {
        cache_stats.misses++;
        ET_LOG(Debug, "aoti_torch_mps_topk: cache miss, building graph");

        @try {
        MPSGraph* graph = [[MPSGraph alloc] init];
        MPSGraphTensor* input = [graph placeholderWithShape:input_shape
                                                   dataType:mps_dtype
                                                       name:@"self"];

        // MPSGraph topK: returns (values, indices) along the last dimension.
        // If dim != -1, we need to transpose dim to last, topk, then transpose back.
        MPSGraphTensor* work = input;
        bool need_transpose = (dim != ndim - 1);

        if (need_transpose) {
          work = [graph transposeTensor:work dimension:dim withDimension:ndim - 1 name:nil];
        }

        // MPSGraph topKWithTensor returns along the last axis
        NSArray<MPSGraphTensor*>* topk_results;
        if (largest) {
          topk_results = [graph topKWithSourceTensor:work k:(NSUInteger)k name:nil];
        } else {
          // For smallest: negate, topk, negate back
          MPSGraphTensor* neg = [graph negativeWithTensor:work name:nil];
          topk_results = [graph topKWithSourceTensor:neg k:(NSUInteger)k name:nil];
          topk_results = @[
            [graph negativeWithTensor:topk_results[0] name:nil],
            topk_results[1]
          ];
        }

        MPSGraphTensor* values_out = topk_results[0];
        MPSGraphTensor* indices_out = topk_results[1];

        if (need_transpose) {
          values_out = [graph transposeTensor:values_out dimension:dim withDimension:ndim - 1 name:nil];
          indices_out = [graph transposeTensor:indices_out dimension:dim withDimension:ndim - 1 name:nil];
        }

        // Cache the graph
        CachedGraph cached_graph;
        cached_graph.graph = graph;
        cached_graph.input1 = input;
        cached_graph.input2 = indices_out;  // reuse input2 slot for indices output
        cached_graph.output = values_out;
        graph_cache[cache_key] = cached_graph;

        // Execute
        id<MTLBuffer> self_buffer = get_mtl_buffer(self_tensor, "topk", "self");
        id<MTLBuffer> values_buffer = ptr_to_mtl_buffer[values_ptr];
        id<MTLBuffer> indices_buffer = ptr_to_mtl_buffer[indices_ptr];

        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
          input: [[MPSGraphTensorData alloc] initWithMTLBuffer:self_buffer shape:input_shape dataType:mps_dtype],
        };

        NSMutableArray<NSNumber*>* out_ns_shape = [NSMutableArray arrayWithCapacity:ndim];
        for (int64_t i = 0; i < ndim; i++) {
          [out_ns_shape addObject:@(out_sizes[i])];
        }

        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
          values_out: [[MPSGraphTensorData alloc] initWithMTLBuffer:values_buffer shape:out_ns_shape dataType:mps_dtype],
          indices_out: [[MPSGraphTensorData alloc] initWithMTLBuffer:indices_buffer shape:out_ns_shape dataType:MPSDataTypeInt32],
        };

        ET_LOG(Debug, "aoti_torch_mps_topk: executing MPSGraph");
        stream->executeMPSGraph(graph, feeds, results, SyncType::COMMIT);
        ET_LOG(Debug, "aoti_torch_mps_topk: MPSGraph done");
        } @catch (NSException* e) {
          ET_LOG(Error, "aoti_torch_mps_topk: ObjC exception: %s - %s",
                  e.name.UTF8String, e.reason.UTF8String);
          throw std::runtime_error(std::string("MPSGraph topk failed: ") + e.reason.UTF8String);
        }
      }

      // Create output tensor handles
      // Values tensor
      AOTITensorHandle values_handle = nullptr;
      aoti_torch_create_tensor_from_blob_v2(
          values_ptr, ndim, out_sizes.data(), out_strides.data(),
          0, dtype, 13, 0, &values_handle, 0, nullptr, 0);

      if (!values_handle) {
        ET_LOG(Error, "aoti_torch_mps_topk: failed to create values tensor");
        aoti_torch_mps_free(values_ptr);
        aoti_torch_mps_free(indices_ptr);
        return Error::Internal;
      }
      ET_LOG(Debug, "aoti_torch_mps_topk: values tensor created");

      extern std::unordered_map<void*, int32_t> memory_to_n_tensor;
      memory_to_n_tensor[values_ptr] = 1;

      // Indices tensor — MPSGraph outputs int32, AOTInductor expects int64.
      // Allocate a new int64 buffer and convert.
      size_t indices_i64_bytes = num_elements * sizeof(int64_t);
      void* indices_i64_ptr = nullptr;
      allocate_mtl_buffer(&indices_i64_ptr, indices_i64_bytes);

      // Copy int32 → int64 on CPU (small tensor, fast)
      {
        auto* stream_sync = getCurrentMetalStream();
        stream_sync->synchronize(SyncType::COMMIT_AND_WAIT);

        int32_t* src = reinterpret_cast<int32_t*>(indices_ptr);
        int64_t* dst = reinterpret_cast<int64_t*>(indices_i64_ptr);
        for (size_t i = 0; i < num_elements; i++) {
          dst[i] = static_cast<int64_t>(src[i]);
        }
      }
      aoti_torch_mps_free(indices_ptr);

      int32_t indices_dtype = static_cast<int32_t>(exec_aten::ScalarType::Long);
      std::vector<int64_t> indices_strides(ndim);
      indices_strides[ndim - 1] = 1;
      for (int64_t i = ndim - 2; i >= 0; i--) {
        indices_strides[i] = indices_strides[i + 1] * out_sizes[i + 1];
      }

      AOTITensorHandle indices_handle = nullptr;
      AOTITorchError idx_err = aoti_torch_create_tensor_from_blob_v2(
          indices_i64_ptr, ndim, out_sizes.data(), indices_strides.data(),
          0, indices_dtype, 13, 0, &indices_handle, 0, nullptr, 0);

      if (idx_err != Error::Ok || !indices_handle) {
        ET_LOG(Error, "aoti_torch_mps_topk: failed to create indices tensor, err=%d", idx_err);
        aoti_torch_mps_free(indices_i64_ptr);
        return Error::Internal;
      }
      memory_to_n_tensor[indices_i64_ptr] = 1;

      *ret0 = values_handle;
      *ret1 = indices_handle;

      ET_LOG(Debug, "aoti_torch_mps_topk: Completed successfully");

    } // @autoreleasepool

    return Error::Ok;

  } catch (const std::exception& e) {
    ET_LOG(Error, "aoti_torch_mps_topk exception: %s", e.what());
    return Error::Internal;
  } catch (...) {
    ET_LOG(Error, "aoti_torch_mps_topk: unknown exception");
    return Error::Internal;
  }
}

} // extern "C"

} // namespace metal
} // namespace backends
} // namespace executorch
