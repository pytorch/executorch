/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/apple/metal/runtime/ops/common.h>

#include <cstring>

namespace executorch {
namespace backends {
namespace metal {

std::unordered_map<GraphCacheKey, CachedGraph, GraphCacheKeyHash> graph_cache;
CacheStats cache_stats;

id<MTLBuffer> get_mtl_buffer(
    Tensor* tensor,
    const char* op_name,
    const char* tensor_name,
    bool* settle_aliases) {
  // MPSGraph reads its inputs as dense tensors. A strided view is handed over
  // as a packed copy, which is encoded on the stream ahead of the graph and so
  // needs no settling; a graph cannot write its result through one.
  if (metal_is_strided_view(tensor)) {
    if (std::strcmp(tensor_name, "out") == 0) {
      ET_LOG(Error, "%s: the out tensor is a view that is not densely packed, which is unsupported", op_name);
      throw std::runtime_error("out tensor is a non-packed view");
    }
    id<MTLBuffer> packed = metal_packed_copy_of_strided_view(*tensor);
    if (!packed) {
      ET_LOG(Error, "%s: failed to make a packed copy of the %s view", op_name, tensor_name);
      throw std::runtime_error(std::string(tensor_name) + " view could not be packed");
    }
    return packed;
  }

  void* data_ptr = tensor->mutable_data_ptr();
  id<MTLBuffer> buffer = nil;
  size_t offset = 0;
  if (!metal_resolve_buffer(data_ptr, &buffer, &offset)) {
    ET_LOG(Error, "%s: %s tensor not found in Metal buffer mapping", op_name, tensor_name);
    throw std::runtime_error(std::string(tensor_name) + " tensor not found in Metal buffer mapping");
  }
  // An empty tensor reads and writes nothing, so it needs no buffer of its
  // own at its offset, and Metal makes none of length 0.
  if (offset == 0 || tensor->nbytes() == 0) {
    return buffer;
  }

  // The tensor is a view that starts partway into `buffer`. MPSGraphTensorData
  // cannot address into a buffer, so the graph needs an MTLBuffer that begins at
  // the view, over the same memory. Metal does not relate that alias to
  // `buffer`, and work using one does not see pending work on the other, so the
  // graph has to run with the memory settled on both sides of it. That is asked
  // of the one graph this buffer is for, through executeMPSGraph.
  id<MTLBuffer> alias = [get_metal_device() newBufferWithBytesNoCopy:data_ptr
                                                              length:tensor->nbytes()
                                                             options:MTLResourceStorageModeShared
                                                         deallocator:nil];
  if (!alias) {
    ET_LOG(Error, "%s: failed to wrap the %s view in a Metal buffer", op_name, tensor_name);
    throw std::runtime_error(std::string(tensor_name) + " view could not be wrapped in a Metal buffer");
  }
  *settle_aliases = true;
  return [alias autorelease];
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
