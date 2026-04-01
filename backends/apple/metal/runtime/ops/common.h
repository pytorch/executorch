/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef __OBJC__
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
typedef id<MTLBuffer> MTLBuffer_t;
typedef MPSGraph* MPSGraph_t;
typedef MPSGraphTensor* MPSGraphTensor_t;
typedef void (^dispatch_block_t)();
#else
typedef void* MTLBuffer_t;
typedef void* MPSGraph_t;
typedef void* MPSGraphTensor_t;
typedef void* dispatch_block_t;
#endif

#include <executorch/backends/apple/metal/runtime/shims/et_metal.h>
#include <executorch/backends/apple/metal/runtime/shims/memory.h>
#include <executorch/backends/apple/metal/runtime/shims/shim_mps.h>
#include <executorch/backends/apple/metal/runtime/shims/utils.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/platform/log.h>
#include <functional>
#include <memory>
#include <unordered_map>

namespace executorch {
namespace backends {
namespace metal {

using executorch::runtime::etensor::Tensor;

void dispatch_sync_with_rethrow(dispatch_queue_t queue, dispatch_block_t block);

extern std::unordered_map<void*, MTLBuffer_t> ptr_to_mtl_buffer;

struct GraphCacheKey {
  std::string op_name;
  std::vector<int64_t> shape_params;
  int32_t dtype;
  bool transpose_flag;

  bool operator==(const GraphCacheKey& other) const {
    return op_name == other.op_name && shape_params == other.shape_params &&
        dtype == other.dtype && transpose_flag == other.transpose_flag;
  }
};

struct GraphCacheKeyHash {
  std::size_t operator()(const GraphCacheKey& key) const {
    std::size_t hash = std::hash<std::string>{}(key.op_name);
    for (auto val : key.shape_params) {
      hash ^=
          std::hash<int64_t>{}(val) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    hash ^= std::hash<int32_t>{}(key.dtype) + 0x9e3779b9 + (hash << 6) +
        (hash >> 2);
    hash ^= std::hash<bool>{}(key.transpose_flag) + 0x9e3779b9 + (hash << 6) +
        (hash >> 2);
    return hash;
  }
};

struct CachedGraph {
  MPSGraph_t graph;
  MPSGraphTensor_t input1;
  MPSGraphTensor_t input2;
  MPSGraphTensor_t input3;
  MPSGraphTensor_t output;
};

struct CacheStats {
  size_t hits = 0;
  size_t misses = 0;

  void logStats() {
    if ((hits + misses) % 100 == 0 && (hits + misses) > 0) {
      double hit_rate = 100.0 * hits / (hits + misses);
      ET_LOG(
          Debug,
          "MPSGraph cache stats: %zu hits, %zu misses (%.1f%% hit rate)",
          hits,
          misses,
          hit_rate);
    }
  }
};

extern std::unordered_map<GraphCacheKey, CachedGraph, GraphCacheKeyHash>
    graph_cache;
extern CacheStats cache_stats;

MTLBuffer_t
get_mtl_buffer(Tensor* tensor, const char* op_name, const char* tensor_name);
MTLBuffer_t allocate_mtl_buffer(void** data_ptr, size_t size_bytes);

} // namespace metal
} // namespace backends
} // namespace executorch
