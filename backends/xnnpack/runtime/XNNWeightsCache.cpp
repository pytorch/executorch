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

// Trivial helpers for little-endian byte serialization of the trailer.
template <typename T>
static void append_le(std::vector<uint8_t>& buf, T value) {
  const auto* p = reinterpret_cast<const uint8_t*>(&value);
  buf.insert(buf.end(), p, p + sizeof(T));
}

template <typename T>
static T read_le(const uint8_t* src) {
  T value;
  memcpy(&value, src, sizeof(T));
  return value;
}

#ifndef _WIN32
// Open the cache file and take an advisory exclusive lock. Returns the
// fd, or -1 if open/flock failed (logs the failure). The caller decides
// how to recover (typically: skip the mmap path for this init).
static int open_locked(const std::string& path, int flags) {
  int fd = open(path.c_str(), flags, 0600);
  if (fd < 0) {
    ET_LOG(Error, "open(%s) failed (errno=%d)", path.c_str(), errno);
    return -1;
  }
  if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
    ET_LOG(Error, "flock(%s) failed (errno=%d)", path.c_str(), errno);
    close(fd);
    return -1;
  }
  return fd;
}

// Drop in-memory state that referenced a now-truncated cache file.
// Heap-backed entries (live in packed_pointer_to_container_) stay; their
// packed_data_ptrs_ slots remain valid so existing offsets don't shift.
void XNNWeightsCache::reset_for_fresh_write() {
  for (auto& region : mmap_regions_) {
    if (region.addr != nullptr && region.addr != MAP_FAILED) {
      munmap(region.addr, region.size);
    }
  }
  mmap_regions_.clear();
  mmap_regions_synced_ = 0;
  packed_file_used_ = 0;
  ptr_to_file_offset_.clear();
  file_ptr_to_region_index_.clear();
  for (auto it = name_to_packed_data_metadata_.begin();
       it != name_to_packed_data_metadata_.end();) {
    bool is_heap_backed = false;
    if (it->second.offset < packed_data_ptrs_.size()) {
      void* ptr = packed_data_ptrs_[it->second.offset];
      if (ptr != nullptr &&
          packed_pointer_to_container_.find(ptr) !=
              packed_pointer_to_container_.end()) {
        is_heap_backed = true;
      }
    }
    if (is_heap_backed) {
      ++it;
    } else {
      it = name_to_packed_data_metadata_.erase(it);
    }
  }
}
#endif

Error XNNWeightsCache::initialize_for_runtime(
    MemoryAllocator* runtime_allocator,
    const NamedDataMap* named_data_map) {
  runtime_allocator_ = runtime_allocator;
  named_data_map_ = named_data_map;
  is_finalized_ = false;

#ifndef _WIN32
  if (packed_cache_path_.empty() || packed_file_fd_ >= 0) {
    return Error::Ok;
  }

  // Entries already in memory (from a prior load_packed_cache or a prior
  // fresh-write session). Just reopen the write fd that save_packed_index
  // closed; subsequent reserve_space can extend the file. Using metadata
  // emptiness (not a separate flag) as the gate avoids a latent bug
  // where fresh-write→save→re-init re-enters load_packed_cache and
  // double-mmaps the same file.
  if (!name_to_packed_data_metadata_.empty()) {
    packed_file_fd_ = open_locked(packed_cache_path_, O_RDWR);
    return Error::Ok;
  }

  // No in-memory entries: try to load the saved trailer; on success open
  // a write fd for any new entries. If load fails, fall through to
  // fresh-write below.
  if (load_packed_cache()) {
    ET_LOG(
        Info,
        "Loaded packed weight cache: %s (%zu entries)",
        packed_cache_path_.c_str(),
        name_to_packed_data_metadata_.size());
    packed_file_fd_ = open_locked(packed_cache_path_, O_RDWR);
    return Error::Ok;
  }

  // Fresh write. Skip O_TRUNC in open_locked so a concurrent holder's
  // mmap stays valid; truncate explicitly only after we hold the lock.
  packed_file_fd_ = open_locked(packed_cache_path_, O_RDWR | O_CREAT);
  if (packed_file_fd_ < 0) {
    return Error::Ok;
  }
  if (ftruncate(packed_file_fd_, 0) != 0) {
    ET_LOG(
        Error,
        "ftruncate(0) failed for %s (errno=%d); heap fallback this init",
        packed_cache_path_.c_str(),
        errno);
    close(packed_file_fd_);
    packed_file_fd_ = -1;
    return Error::Ok;
  }
  reset_for_fresh_write();
  ET_LOG(
      Info,
      "Opened packed weight file for writing: %s",
      packed_cache_path_.c_str());
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
  // Synchronous flush for newly added regions. MS_SYNC blocks until the
  // dirty pages are written to disk and marked clean
  if (mmap_regions_.size() > mmap_regions_synced_) {
    size_t new_count = mmap_regions_.size() - mmap_regions_synced_;
    for (size_t i = mmap_regions_synced_; i < mmap_regions_.size(); ++i) {
      if (mmap_regions_[i].addr != nullptr) {
        msync(mmap_regions_[i].addr, mmap_regions_[i].size, MS_SYNC);
      }
    }
    mmap_regions_synced_ = mmap_regions_.size();
    ET_LOG(
        Info,
        "Synced %zu new regions (%zu total), %zu MB packed weights",
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

void XNNWeightsCache::release_entry(void* packed_data_ptr) {
  packed_pointer_to_container_.erase(packed_data_ptr);
#ifndef _WIN32
  // Per-entry file-backed mmap region: munmap to release VM. The
  // packed_data_ptrs_ slot is nulled by the caller so existing offsets
  // stay valid.
  auto region_it = file_ptr_to_region_index_.find(packed_data_ptr);
  if (region_it != file_ptr_to_region_index_.end()) {
    MmapRegion& region = mmap_regions_[region_it->second];
    if (region.addr != nullptr && region.addr != MAP_FAILED) {
      munmap(region.addr, region.size);
      region.addr = nullptr;
      region.size = 0;
    }
    file_ptr_to_region_index_.erase(region_it);
  }
#endif
}

void XNNWeightsCache::full_unload() {
#ifndef _WIN32
  for (auto& region : mmap_regions_) {
    if (region.addr != nullptr && region.addr != MAP_FAILED) {
      munmap(region.addr, region.size);
      region.addr = nullptr;
      region.size = 0;
    }
  }
  mmap_regions_.clear();
  mmap_regions_synced_ = 0;
  packed_data_ptrs_.clear();
  ptr_to_file_offset_.clear();
  file_ptr_to_region_index_.clear();
  if (packed_file_fd_ >= 0) {
    close(packed_file_fd_);
    packed_file_fd_ = -1;
  }
#endif
}

Error XNNWeightsCache::delete_packed_data(
    const std::vector<std::string>& packed_data_names) {
  if (!is_finalized_) {
    ET_LOG(Error, "delete_packed_data called before finalize_for_runtime");
    return Error::InvalidArgument;
  }
  for (const std::string& name : packed_data_names) {
    auto entry = name_to_packed_data_metadata_.find(name);
    if (entry == name_to_packed_data_metadata_.end()) {
      ET_LOG(Error, "delete_packed_data: '%s' not found", name.c_str());
      return Error::InvalidArgument;
    }
    if (--entry->second.ref_count > 0) {
      continue;
    }
    // Keep from_load entries: their packed bytes live in the cache file
    // and stay valid until full unload. Erasing them would force the
    // next init to re-pack and append ~450 MB to the file per cycle.
    if (entry->second.from_load) {
      entry->second.in_current_runtime = false;
      continue;
    }
    release_entry(packed_data_ptrs_[entry->second.offset]);
    packed_data_ptrs_[entry->second.offset] = nullptr;
    name_to_packed_data_metadata_.erase(entry);
  }

  // Last entry gone: drop all in-memory state. File on disk is preserved
  // so the next process can load_packed_cache and skip re-packing. If
  // reserve_space after the last save corrupted the trailer, load will
  // fall through to fresh-write — same outcome as truncating here.
  if (name_to_packed_data_metadata_.empty()) {
    full_unload();
  }
  return Error::Ok;
}

size_t XNNWeightsCache::look_up(
    XNNWeightsCache* context,
    const xnn_weights_cache_look_up_key* cache_key) {
  const void* unpacked_weights_ptr = cache_key->kernel;
  const void* unpacked_bias_ptr = cache_key->bias;
  auto entry = context->unpacked_data_to_name_.find(unpacked_weights_ptr);
  if (entry == context->unpacked_data_to_name_.end()) {
    return SIZE_MAX;
  }
  std::string weight_bias_name = entry->second;

  if (unpacked_bias_ptr != nullptr) {
    auto bias_entry = context->unpacked_data_to_name_.find(unpacked_bias_ptr);
    if (bias_entry != context->unpacked_data_to_name_.end()) {
      weight_bias_name.append(bias_entry->second);
    }
  }

  auto packed_weight_entry =
      context->name_to_packed_data_metadata_.find(weight_bias_name);
  if (packed_weight_entry == context->name_to_packed_data_metadata_.end()) {
    return SIZE_MAX;
  }
  // XNNPACK upgrade detection: a ukernel whose implementation changed
  // produces a different seed. Reject the cached entry so look_up_or_insert
  // falls through to re-pack with the current ukernel.
  if (packed_weight_entry->second.seed != cache_key->seed) {
    ET_LOG(
        Info,
        "look_up: seed mismatch for '%s' (cached=0x%08x, current=0x%08x); "
        "treating as miss for re-pack",
        weight_bias_name.c_str(),
        packed_weight_entry->second.seed,
        cache_key->seed);
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
          "reserve_space ftruncate to %zu failed (errno=%d)",
          file_offset + map_size,
          errno);
      close(context->packed_file_fd_);
      context->packed_file_fd_ = -1;
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
      ET_LOG(
          Error,
          "reserve_space mmap %zu bytes failed (errno=%d)",
          map_size,
          errno);
      close(context->packed_file_fd_);
      context->packed_file_fd_ = -1;
      return context->reserve_space_heap(n);
    }

    // mmap returns page-aligned (>= 4 KiB), which trivially satisfies the
    // 64-byte kPackedAllocationAlignment XNNPACK expects.
    ET_DCHECK_MSG(
        (reinterpret_cast<uintptr_t>(ptr) % kPackedAllocationAlignment) == 0,
        "mmap returned ptr not aligned to %zu bytes",
        kPackedAllocationAlignment);

    context->packed_file_used_ = file_offset + map_size;
    context->file_ptr_to_region_index_[ptr] = context->mmap_regions_.size();
    context->mmap_regions_.push_back({ptr, map_size});
    context->ptr_to_file_offset_[ptr] = file_offset;
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

  // XNNPACK calls with ptr==nullptr after a cache hit (no packing
  // happened, nothing to validate against). Return the offset as-is.
  if (ptr == nullptr) {
    return offset;
  }

  if (offset != SIZE_MAX) {
    void* saved_ptr = context->offset_to_addr(context, offset);
    if (saved_ptr != nullptr && 0 == memcmp(ptr, saved_ptr, size)) {
      return offset;
    }
    // Cache out of date: name hits but packed bytes differ.
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
    packed_data_metadata.data_size = size;
    packed_data_metadata.ref_count =
        0; // ref_count is only incremented after finalizing for runtime
    packed_data_metadata.in_current_runtime = true;
    packed_data_metadata.seed = cache_key->seed;
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

Error XNNWeightsCache::save_packed_index() {
#ifndef _WIN32
  if (packed_file_fd_ < 0) {
    return Error::Ok;
  }
  // Skip no-op saves: identical bytes would still bump mtime via
  // pwrite/fsync, making the cache file appear modified on every load.
  // The `mmap_regions_at_last_save_ > 0` guard is sufficient because a
  // successful save closes packed_file_fd_ before returning, so re-entry
  // past the `fd < 0` early-return above requires initialize_for_runtime
  // to reopen the fd, which only happens via load_packed_cache (or the
  // fresh-write path) that always populates at least one mmap region.
  if (mmap_regions_.size() == mmap_regions_at_last_save_ &&
      mmap_regions_at_last_save_ > 0) {
    return Error::Ok;
  }

  size_t index_start = packed_file_used_;
  std::vector<uint8_t> buf;
  uint32_t entry_count = 0;

  // Index entry: [name_len:u32][name][file_offset:u64][data_size:u64][seed:u32]
  for (const auto& [name, meta] : name_to_packed_data_metadata_) {
    void* ptr = packed_data_ptrs_[meta.offset];
    auto it = ptr_to_file_offset_.find(ptr);
    if (it == ptr_to_file_offset_.end()) {
      continue;
    }
    entry_count++;
    append_le(buf, static_cast<uint32_t>(name.size()));
    buf.insert(buf.end(), name.begin(), name.end());
    append_le(buf, static_cast<uint64_t>(it->second));
    append_le(buf, static_cast<uint64_t>(meta.data_size));
    append_le(buf, meta.seed);
  }

  // Footer: [index_start:u64][entry_count:u32][magic:u32][version:u32]
  append_le(buf, static_cast<uint64_t>(index_start));
  append_le(buf, entry_count);
  append_le(buf, kCacheMagic);
  append_le(buf, kCacheVersion);

  if (ftruncate(packed_file_fd_, index_start + buf.size()) != 0) {
    ET_LOG(Error, "Failed to extend file for index (errno=%d)", errno);
    return Error::Internal;
  }
  ssize_t written =
      pwrite(packed_file_fd_, buf.data(), buf.size(), index_start);
  if (written != static_cast<ssize_t>(buf.size())) {
    ET_LOG(Error, "Failed to write index (errno=%d)", errno);
    return Error::Internal;
  }
  // Ensure trailer is on disk before we declare success.
  if (fsync(packed_file_fd_) != 0) {
    ET_LOG(Error, "fsync of packed cache failed (errno=%d)", errno);
    // Continue — data is in page cache; durability is best-effort.
  }
  // Log the final file size (= index_start + trailer) so production
  // logs surface unbounded growth from orphan packs: a same-name
  // re-pack leaves the old packed bytes in the file even though the
  // trailer drops the old entry. Monitoring file_bytes over time tells
  // us when GC or a size cap is needed.
  const size_t file_bytes = index_start + buf.size();
  ET_LOG(
      Info,
      "Saved packed weight index: %u entries at offset %zu, file_bytes=%zu",
      entry_count,
      index_start,
      file_bytes);

  // Promote freshly-packed entries to from_load now that they're durable
  // on disk, so delete_packed_data preserves them across unload/reload.
  for (auto& [name, meta] : name_to_packed_data_metadata_) {
    if (!meta.from_load &&
        ptr_to_file_offset_.find(packed_data_ptrs_[meta.offset]) !=
            ptr_to_file_offset_.end()) {
      meta.from_load = true;
    }
  }

  mmap_regions_at_last_save_ = mmap_regions_.size();

  // Close the fd so the next init re-enters load_packed_cache and reads
  // the trailer we just wrote.
  if (close(packed_file_fd_) != 0) {
    ET_LOG(Error, "close of packed cache fd failed (errno=%d)", errno);
  }
  packed_file_fd_ = -1;
#endif
  return Error::Ok;
}

bool XNNWeightsCache::load_packed_cache() {
#ifndef _WIN32
  int fd = open(packed_cache_path_.c_str(), O_RDONLY);
  if (fd < 0) {
    return false;
  }
  // Prevent racing with a concurrent writer
  if (flock(fd, LOCK_SH | LOCK_NB) != 0) {
    close(fd);
    return false;
  }
  struct stat st {};
  if (fstat(fd, &st) != 0 || st.st_size < 20) {
    close(fd);
    return false;
  }
  size_t file_size = static_cast<size_t>(st.st_size);

  uint8_t footer[20];
  if (pread(fd, footer, 20, file_size - 20) != 20) {
    close(fd);
    return false;
  }
  uint64_t index_start = read_le<uint64_t>(footer);
  uint32_t entry_count = read_le<uint32_t>(footer + 8);
  uint32_t magic = read_le<uint32_t>(footer + 12);
  uint32_t version = read_le<uint32_t>(footer + 16);

  if (magic != kCacheMagic || version != kCacheVersion ||
      index_start >= file_size - 20) {
    close(fd);
    return false;
  }
  const size_t index_region_end = file_size - 20;

  void* map = mmap(nullptr, file_size, PROT_READ, MAP_SHARED, fd, 0);
  close(fd);
  if (map == MAP_FAILED) {
    return false;
  }
  mmap_regions_.push_back({map, file_size});

  const uint8_t* cursor = static_cast<const uint8_t*>(map) + index_start;
  const uint8_t* end = static_cast<const uint8_t*>(map) + index_region_end;

  for (uint32_t i = 0; i < entry_count && cursor + 4 <= end; ++i) {
    uint32_t name_len = read_le<uint32_t>(cursor);
    cursor += 4;
    // [file_offset:u64][data_size:u64][seed:u32] = 20 bytes
    if (cursor + name_len + 20 > end) {
      // Truncated entry header: trailer doesn't match the entry_count we
      // read from the footer, so the cache is corrupt. Apply the same
      // full rollback as the invalid-bounds branch below — otherwise the
      // entries inserted so far would be silently accepted as a partial
      // cache, and the next save_packed_index would rewrite a trailer
      // covering only that subset (permanently dropping the rest).
      ET_LOG(
          Error,
          "load_packed_cache: truncated entry header at index %u (entry_count=%u); aborting load",
          i,
          entry_count);
      munmap(map, file_size);
      mmap_regions_.pop_back();
      name_to_packed_data_metadata_.clear();
      packed_data_ptrs_.clear();
      ptr_to_file_offset_.clear();
      return false;
    }
    std::string name(reinterpret_cast<const char*>(cursor), name_len);
    cursor += name_len;
    uint64_t file_offset = read_le<uint64_t>(cursor);
    cursor += 8;
    uint64_t data_size = read_le<uint64_t>(cursor);
    cursor += 8;
    uint32_t seed = read_le<uint32_t>(cursor);
    cursor += 4;

    // Bounds check: the entry's bytes must lie entirely inside the
    // packed-data region.
    if (file_offset >= index_start || data_size > index_start - file_offset) {
      ET_LOG(
          Error,
          "load_packed_cache: entry '%s' has invalid bounds (file_offset=%llu, data_size=%llu, index_start=%llu); aborting load",
          name.c_str(),
          static_cast<unsigned long long>(file_offset),
          static_cast<unsigned long long>(data_size),
          static_cast<unsigned long long>(index_start));
      // Roll back any partial state.
      munmap(map, file_size);
      mmap_regions_.pop_back();
      name_to_packed_data_metadata_.clear();
      packed_data_ptrs_.clear();
      ptr_to_file_offset_.clear();
      return false;
    }

    size_t ptr_index = packed_data_ptrs_.size();
    void* entry_ptr = static_cast<char*>(map) + file_offset;
    packed_data_ptrs_.push_back(entry_ptr);
    // Tracked so a subsequent save_packed_index can rewrite the trailer
    // covering both loaded and newly-packed entries.
    ptr_to_file_offset_[entry_ptr] = file_offset;
    PackedDataMeta meta;
    meta.offset = ptr_index;
    meta.data_size = data_size;
    meta.ref_count = 0;
    meta.in_current_runtime = false;
    meta.from_load = true;
    meta.seed = seed;
    name_to_packed_data_metadata_[name] = meta;
  }

  packed_file_used_ = index_start;
  // In-memory state matches the on-disk trailer; the next save would be
  // a no-op. Initialize watermark so save_packed_index short-circuits.
  mmap_regions_at_last_save_ = mmap_regions_.size();
  return true;
#else
  return false;
#endif
}

} // namespace delegate
} // namespace xnnpack
} // namespace backends
} // namespace executorch
