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
#ifndef _WIN32
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#endif
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

XNNWeightsCache::~XNNWeightsCache() {
#ifndef _WIN32
  for (auto& region : mmap_regions_) {
    if (region.addr != nullptr && region.addr != MAP_FAILED) {
      munmap(region.addr, region.size);
    }
  }
  mmap_regions_.clear();
  if (packed_file_fd_ >= 0) {
    close(packed_file_fd_);
    packed_file_fd_ = -1;
  }
#endif
}

Error XNNWeightsCache::initialize_for_runtime(
    MemoryAllocator* runtime_allocator,
    const NamedDataMap* named_data_map) {
  runtime_allocator_ = runtime_allocator;
  named_data_map_ = named_data_map;
  is_finalized_ = false;

#ifndef _WIN32
  // Open the file for packed weights. Each reserve_space() call
  // independently mmaps a region of the file. Once packed_file_disabled_
  // is set we never re-open — re-opening with O_TRUNC would corrupt any
  // still-live mappings into the same path and cause SIGBUS on access.
  if (!packed_cache_path_.empty() && packed_file_fd_ < 0 &&
      !packed_file_disabled_) {
    packed_file_fd_ =
        open(packed_cache_path_.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (packed_file_fd_ < 0) {
      ET_LOG(
          Error,
          "Failed to open packed weight file: %s (errno=%d)",
          packed_cache_path_.c_str(),
          errno);
    } else if (flock(packed_file_fd_, LOCK_EX | LOCK_NB) != 0) {
      // Another XNNWeightsCache instance (this process or another) is
      // already using this path. O_TRUNC above would corrupt its mappings.
      // Disable mmap for this instance to prevent collision; fall back to
      // heap allocation for the remainder of this cache's lifetime.
      ET_LOG(
          Error,
          "Another instance is using packed weight cache file %s (errno=%d); "
          "disabling mmap path",
          packed_cache_path_.c_str(),
          errno);
      close(packed_file_fd_);
      packed_file_fd_ = -1;
      packed_file_disabled_ = true;
    } else {
      ET_LOG(Info, "Opened packed weight file: %s", packed_cache_path_.c_str());
    }
  }
#endif

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

#ifndef _WIN32
  // Schedule async flush for newly added regions only.
  // MS_ASYNC returns immediately; OS flushes in the background.
  if (mmap_regions_.size() > mmap_regions_synced_) {
    size_t new_count = mmap_regions_.size() - mmap_regions_synced_;
    for (size_t i = mmap_regions_synced_; i < mmap_regions_.size(); ++i) {
      if (mmap_regions_[i].addr != nullptr) {
        msync(mmap_regions_[i].addr, mmap_regions_[i].size, MS_ASYNC);
      }
    }
    mmap_regions_synced_ = mmap_regions_.size();
    ET_LOG(
        Info,
        "Scheduled async flush: %zu new regions (%zu total), %zu MB packed weights",
        new_count,
        mmap_regions_.size(),
        packed_file_used_ / (1024 * 1024));
  }
#endif

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
        // Erase the key/value from the map frees the pointer holding the
        // packed data. No-op on the file-backed mmap path, where the
        // container is not populated.
        packed_pointer_to_container_.erase(packed_data_ptr);
#ifndef _WIN32
        // File-backed mmap path: munmap the region so VM and page-cache
        // usage is released, not just retained until cache destruction.
        // The vector slot is set to nullptr below so existing offsets remain
        // valid for any concurrent lookups.
        auto region_it = file_ptr_to_region_index_.find(packed_data_ptr);
        if (region_it != file_ptr_to_region_index_.end()) {
          size_t idx = region_it->second;
          MmapRegion& region = mmap_regions_[idx];
          if (region.addr != nullptr && region.addr != MAP_FAILED) {
            munmap(region.addr, region.size);
            region.addr = nullptr;
            region.size = 0;
          }
          file_ptr_to_region_index_.erase(region_it);
        }
#endif
        // Remove the pointer from packed_data_ptrs_.
        packed_data_ptrs_[entry->second.offset] = nullptr;
        // Erase the name to packed metadata entry.
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

void* XNNWeightsCache::reserve_space(XNNWeightsCache* context, size_t n) {
#ifndef _WIN32
  if (context->packed_file_fd_ >= 0) {
    size_t page_size = sysconf(_SC_PAGESIZE);
    size_t file_offset =
        (context->packed_file_used_ + page_size - 1) & ~(page_size - 1);
    size_t map_size = (n + page_size - 1) & ~(page_size - 1);

    if (ftruncate(context->packed_file_fd_, file_offset + map_size) != 0) {
      ET_LOG(
          Error,
          "ftruncate to %zu failed (errno=%d)",
          file_offset + map_size,
          errno);
      close(context->packed_file_fd_);
      context->packed_file_fd_ = -1;
      // Existing mmap_regions_ still reference this inode. Disable the
      // file-backed path permanently so a future initialize_for_runtime
      // doesn't re-open + O_TRUNC the same path and trigger SIGBUS on the
      // stale mappings.
      context->packed_file_disabled_ = true;
      return context->reserve_space_heap(n);
    }

    void* ptr = mmap(
        nullptr,
        map_size,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        context->packed_file_fd_,
        file_offset);
    if (ptr == MAP_FAILED) {
      ET_LOG(Error, "mmap %zu bytes failed (errno=%d)", map_size, errno);
      close(context->packed_file_fd_);
      context->packed_file_fd_ = -1;
      context->packed_file_disabled_ = true;
      return context->reserve_space_heap(n);
    }

    // mmap returns page-aligned (>= 4 KiB), which trivially satisfies the
    // 64-byte kPackedAllocationAlignment XNNPACK expects. Assert defensively.
    ET_DCHECK_MSG(
        (reinterpret_cast<uintptr_t>(ptr) % kPackedAllocationAlignment) == 0,
        "mmap returned ptr not aligned to %zu bytes",
        kPackedAllocationAlignment);

    context->packed_file_used_ = file_offset + map_size;
    context->file_ptr_to_region_index_[ptr] = context->mmap_regions_.size();
    context->mmap_regions_.push_back({ptr, map_size});
    return ptr;
  }
#endif

  return context->reserve_space_heap(n);
}

void* XNNWeightsCache::reserve_space_heap(size_t n) {
  try {
    std::string data_container;
    size_t raw_allocation_size = n + kPackedAllocationAlignment - 1;
    data_container.resize(raw_allocation_size);

    void* maybe_aligned_space = data_container.data();
    void* aligned_space = std::align(
        kPackedAllocationAlignment,
        n,
        maybe_aligned_space,
        raw_allocation_size // Note that std::align mutates this value.
    );
    ET_CHECK_MSG(aligned_space != nullptr, "Memory alignment failed.");

    packed_pointer_to_container_[aligned_space] = std::move(data_container);
    return aligned_space;
  } catch (std::bad_alloc& e) {
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

  // XNNPACK can call this with ptr==nullptr when it previously hit the cache
  // and skipped packing. We can't validate against the ptr contents in this
  // case, so just return the offset. This might actually be a bug in XNNPACK
  // since calling look_up_or_insert with ptr==nullptr doesn't really make
  // sense...
  if (ptr == nullptr) {
    return offset;
  }

  if (offset != SIZE_MAX) {
    void* saved_ptr = context->offset_to_addr(context, offset);
    if (saved_ptr != nullptr && 0 == memcmp(ptr, saved_ptr, size)) {
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

void XNNWeightsCache::set_packed_cache_path(const std::string& path) {
  packed_cache_path_ = path;
}

} // namespace delegate
} // namespace xnnpack
} // namespace backends
} // namespace executorch
