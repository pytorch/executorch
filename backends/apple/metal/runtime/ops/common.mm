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

std::unordered_map<GraphCacheKey, CachedGraph, GraphCacheKeyHash> graph_cache;
CacheStats cache_stats;

id<MTLBuffer> get_mtl_buffer(Tensor* tensor, const char* op_name, const char* tensor_name) {
  void* data_ptr = tensor->mutable_data_ptr();
  auto it = ptr_to_mtl_buffer.find(data_ptr);
  if (it == ptr_to_mtl_buffer.end()) {
    ET_LOG(Error, "%s: %s tensor not found in Metal buffer mapping", op_name, tensor_name);
    throw std::runtime_error(std::string(tensor_name) + " tensor not found in Metal buffer mapping");
  }
  return it->second;
}

id<MTLBuffer> allocate_mtl_buffer(void** data_ptr, size_t size_bytes) {
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

} // namespace metal
} // namespace backends
} // namespace executorch
