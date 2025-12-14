/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/xnnpack/runtime/XNNWeightsCache.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <sys/stat.h>
#include <xnnpack.h>
#include <exception>
#include <memory>
#include <new>
#include <string>
#include <vector>

namespace executorch {
namespace backends {
namespace xnnpack {
namespace delegate {

using executorch::ET_RUNTIME_NAMESPACE::NamedDataMap;
using executorch::runtime::MemoryAllocator;

XNNWeightsCache::XNNWeightsCache() {
  weights_cache_.context = this;
  weights_cache_.look_up = (size_t(*)(
      void*, const xnn_weights_cache_look_up_key*))XNNWeightsCache::look_up;
  weights_cache_.reserve_space =
      (void* (*)(void*, size_t))XNNWeightsCache::reserve_space;
  weights_cache_.look_up_or_insert =
      (size_t(*)(void*, const xnn_weights_cache_look_up_key*, void*, size_t))
          XNNWeightsCache::look_up_or_insert;
  weights_cache_.is_finalized = (bool (*)(void*))XNNWeightsCache::is_finalized;
  weights_cache_.offset_to_addr =
      (void* (*)(void*, size_t))XNNWeightsCache::offset_to_addr;
  weights_cache_.delete_cache =
      (enum xnn_status(*)(void*))XNNWeightsCache::delete_cache;
}

Error XNNWeightsCache::initialize_for_runtime(
    MemoryAllocator* runtime_allocator,
    const NamedDataMap* named_data_map) {
  runtime_allocator_ = runtime_allocator;
  named_data_map_ = named_data_map;
  is_finalized_ = false;

  return Error::Ok;
}

Result<std::vector<std::string>> XNNWeightsCache::finalize_for_runtime() {
  is_finalized_ = true;

  // All data has been packed by create_runtime
  // so we clear the unpacked data as it is no longer needed
  for (FreeableBuffer& buffer : unpacked_data_) {
    buffer.Free();
  }
  unpacked_data_.clear();
  unpacked_data_to_name_.clear();

  std::vector<std::string> packed_data_names;
  // update the reference count of all the packed data
  // used by this runtime
  for (auto& entry : name_to_packed_data_metadata_) {
    if (entry.second.in_current_runtime) {
      entry.second.ref_count++;
      entry.second.in_current_runtime = false;
      packed_data_names.push_back(entry.first);
    }
  }

  return packed_data_names;
}

Result<const uint8_t*> XNNWeightsCache::load_unpacked_data(
    const std::string& name) {
  Result<FreeableBuffer> named_data = named_data_map_->get_data(name.c_str());
  if (!named_data.ok()) {
    ET_LOG(Error, "Failed to load constant data for key %s", name.c_str());
    return Error::InvalidExternalData;
  }
  const uint8_t* data_pointer =
      static_cast<const uint8_t*>(named_data.get().data());
  unpacked_data_.push_back(std::move(named_data.get()));
  unpacked_data_to_name_[data_pointer] = name;

  return data_pointer;
}

Error XNNWeightsCache::delete_packed_data(
    const std::vector<std::string>& packed_data_names) {
  if (!is_finalized_) {
    ET_LOG(
        Error,
        "Error, attempted to delete packed data from the cache but the cache is not finalized");
    return Error::InvalidArgument;
  }
  for (const std::string& name : packed_data_names) {
    auto entry = name_to_packed_data_metadata_.find(name);
    if (entry == name_to_packed_data_metadata_.end()) {
      ET_LOG(
          Error,
          "Error, attempted to deleted packed data: %s, from the cache but it wasn't found",
          name.c_str());
      return Error::InvalidArgument;
    } else {
      entry->second.ref_count--;
      if (entry->second.ref_count == 0) {
        void* packed_data_ptr = packed_data_ptrs_[entry->second.offset];
        // Erase the key/value from the map frees the pointer holding the packed
        // data
        packed_pointer_to_container_.erase(packed_data_ptr);
        // remove the pointer from the packed_data_ptrs_
        packed_data_ptrs_[entry->second.offset] = nullptr;
        // Erase the name to packed metadata entry
        name_to_packed_data_metadata_.erase(entry->first);
      }
    }
  }

  return Error::Ok;
}

size_t XNNWeightsCache::look_up(
    XNNWeightsCache* context,
    const xnn_weights_cache_look_up_key* cache_key) {
  const void* unpacked_weights_ptr = cache_key->kernel;
  const void* unpacked_bias_ptr = cache_key->bias;
  auto entry = context->unpacked_data_to_name_.find(unpacked_weights_ptr);

  // Check if weight_pointer has been cached
  if (entry == context->unpacked_data_to_name_.end()) {
    return SIZE_MAX;
  }

  std::string weight_bias_name = entry->second;

  // Check if bias_pointer has been cached
  if (unpacked_bias_ptr != nullptr) {
    auto bias_entry = context->unpacked_data_to_name_.find(unpacked_bias_ptr);
    if (bias_entry != context->unpacked_data_to_name_.end()) {
      weight_bias_name.append(bias_entry->second);
    }
  }

  // check if weight_bias_name has been packed already
  auto packed_weight_entry =
      context->name_to_packed_data_metadata_.find(weight_bias_name);
  if (packed_weight_entry == context->name_to_packed_data_metadata_.end()) {
    return SIZE_MAX;
  }
  packed_weight_entry->second.in_current_runtime = true;

  return packed_weight_entry->second.offset;
}

/**
 * Reserve space in the weight cache for n bytes of weight data, aligned to
 * context->kPackedAllocationAlignment. This function will return nullptr if
 * the allocation fails.
 */
void* XNNWeightsCache::reserve_space(XNNWeightsCache* context, size_t n) {
  // MemoryAllocator* allocator = context->runtime_allocator_;
  // void* reserved_pointer = allocator->allocate(n,
  // context->kPackedAllocationAlignment);

  // return reserved_pointer;
  try {
    std::string data_container;
    size_t raw_allocation_size = n + context->kPackedAllocationAlignment - 1;
    data_container.resize(raw_allocation_size);

    void* maybe_aligned_space = data_container.data();
    void* aligned_space = std::align(
        context->kPackedAllocationAlignment,
        n,
        maybe_aligned_space,
        raw_allocation_size // Note that std::align mutates this value.
    );
    ET_CHECK_MSG(aligned_space != nullptr, "Memory alignment failed.");

    context->packed_pointer_to_container_[aligned_space] =
        std::move(data_container);
    return aligned_space;
  } catch (std::bad_alloc& e) {
    // XNNPACK can gracefully handle allocation failures, so return nullptr.
    // We want to be able to recover from a failed attempt to load a large
    // model without a crash.
    ET_LOG(
        Error,
        "XNN weight cache failed to allocate %zu bytes: %s.",
        n,
        e.what());
    return nullptr;
  }
}

size_t XNNWeightsCache::look_up_or_insert(
    XNNWeightsCache* context,
    const xnn_weights_cache_look_up_key* cache_key,
    void* ptr,
    size_t size) {
  size_t offset = context->look_up(context, cache_key);

  if (offset != SIZE_MAX) {
    void* saved_ptr = context->offset_to_addr(context, offset);
    // Check for null pointers before calling memcmp
    if (ptr == nullptr || saved_ptr == nullptr) {
      // If either pointer is null, cache is invalid
      return SIZE_MAX;
    }
    if (0 == memcmp(ptr, saved_ptr, size)) {
      return offset;
    }
    // Failure, cache is out of date
    return SIZE_MAX;
  }

  // Add to Cache if it is not finalized
  size_t next_offset = context->packed_data_ptrs_.size();
  auto entry = context->unpacked_data_to_name_.find(cache_key->kernel);

  // Check if weight_pointer has been cached
  if (entry != context->unpacked_data_to_name_.end()) {
    std::string weight_bias_name = entry->second;
    if (cache_key->bias != nullptr) {
      auto bias_entry = context->unpacked_data_to_name_.find(cache_key->bias);
      if (bias_entry != context->unpacked_data_to_name_.end()) {
        weight_bias_name.append(bias_entry->second);
      }
    }
    PackedDataMeta packed_data_metadata;
    packed_data_metadata.offset = next_offset;
    packed_data_metadata.ref_count =
        0; // ref_count is only incremented after finalizing for runtime
    packed_data_metadata.in_current_runtime = true;
    context->name_to_packed_data_metadata_[weight_bias_name] =
        packed_data_metadata;
  } else {
    ET_LOG(
        Info,
        "Warning: Unpacked weight and bias were not registered with names, "
        "this will add new cache entries for packed data and may affect performance.");
  }
  context->packed_data_ptrs_.push_back(ptr);

  return next_offset;
}

bool XNNWeightsCache::is_finalized(XNNWeightsCache* context) {
  return context->is_finalized_;
}

void* XNNWeightsCache::offset_to_addr(XNNWeightsCache* context, size_t offset) {
  return context->packed_data_ptrs_[offset];
}

enum xnn_status XNNWeightsCache::delete_cache(XNNWeightsCache* context) {
  return xnn_status_success;
}

} // namespace delegate
} // namespace xnnpack
} // namespace backends
} // namespace executorch
