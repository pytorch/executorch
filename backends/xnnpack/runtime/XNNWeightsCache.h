/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <xnnpack.h>

#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/executor/pte_data_map.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace executorch {
namespace backends {
namespace xnnpack {
namespace delegate {

using executorch::ET_RUNTIME_NAMESPACE::NamedDataMap;
using executorch::runtime::Error;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::Result;

struct PackedDataMeta {
  size_t offset{};
  size_t data_size{0};
  // Count number of xnn_runtime_t this packed data is used in
  size_t ref_count{};
  // true if this packed data was inserted or looked up for the
  // current runtime being created
  bool in_current_runtime{};
  // True if this entry's bytes are persisted in the on-disk cache file
  // (either originally loaded via load_packed_cache, or freshly packed
  // and then save_packed_index-ed). Used by delete_packed_data to
  // detect when all persistent entries are gone, at which point
  // cache_loaded_ is auto-invalidated so the next init re-enters
  // load_packed_cache and reuses the saved file instead of re-packing.
  bool from_load{false};
  // Per-ukernel seed from xnn_weights_cache_look_up_key.seed. XNNPACK
  // guarantees this is consistent across runs of the same ukernel; when
  // XNNPACK upgrades and a ukernel implementation changes, the seed
  // changes. look_up rejects entries whose stored seed doesn't match
  // the caller's seed so that stale cache entries don't deliver wrongly
  // packed weights to a newer ukernel.
  uint32_t seed{0};
};

class XNNWeightsCache {
 public:
  XNNWeightsCache();
  ~XNNWeightsCache();

  // Owns OS resources (file descriptor, mmap regions). Non-copyable,
  // non-movable. cppcoreguidelines-special-member-functions.
  XNNWeightsCache(const XNNWeightsCache&) = delete;
  XNNWeightsCache& operator=(const XNNWeightsCache&) = delete;
  XNNWeightsCache(XNNWeightsCache&&) = delete;
  XNNWeightsCache& operator=(XNNWeightsCache&&) = delete;

  /**
   * Initializes the XNNWeightsCache for the next xnn_create_runtime
   */
  Error initialize_for_runtime(
      MemoryAllocator* runtime_allocator,
      const NamedDataMap* named_data_map);

  /**
   * Finalizes the weights cache after the weights have been packed
   * in xnn_create_runtime.
   *
   * This should only be called after creating the runtime. Returns
   * the name of all the packed weights used by this runtime
   */
  Result<std::vector<std::string>> finalize_for_runtime();

  // Taken from XNN_ALLOCATION_ALIGNMENT in xnnpack/common.h
  static const size_t kPackedAllocationAlignment = 64;

  /**
   * Returns XNNPACK's underlying weights_cache pointer
   */
  inline xnn_weights_cache_t get() {
    return (xnn_weights_cache_t)&weights_cache_;
  }

  /**
   * Returns the number of unpacked data
   */
  inline size_t get_num_unpacked_data() {
    return unpacked_data_.size();
  }

  /**
   * Returns the names of all unpacked data
   */
  inline std::vector<std::string> get_unpacked_data_names() {
    std::vector<std::string> names;
    names.reserve(unpacked_data_to_name_.size());
    for (const auto& pair : unpacked_data_to_name_) {
      names.push_back(pair.second);
    }
    return names;
  }

  /**
   * Returns the packed data names
   */
  inline std::vector<std::string> get_packed_data_names() {
    std::vector<std::string> names;
    names.reserve(name_to_packed_data_metadata_.size());
    for (const auto& pair : name_to_packed_data_metadata_) {
      names.push_back(pair.first);
    }
    return names;
  }

  /**
   * Loads unpacked named data from the NamedDataMap into this XNNWeightsCache
   * and returns a pointer to the unpacked data. This unpacked data is given
   * to XNNPACK's define_tensor APIs, and used as the cache key for
   * look_up_or_insert.
   * @param[in] name The name of the data to load
   * @param[out] out the pointer to the unpacked data that was loaded
   */
  Result<const uint8_t*> load_unpacked_data(const std::string& name);

  /**
   * Deletes the packed data associated with the names given.
   * Decrements the ref_count if the packed data is used by other
   * models
   *
   */
  Error delete_packed_data(const std::vector<std::string>& packed_names);

  /**
   * Set the path for the file-backed packed weight storage.
   * When set, reserve_space() allocates from a MAP_SHARED file instead
   * of heap, and finalize_for_runtime() calls msync to make pages clean.
   *
   * The path MUST be unique per XNNWeightsCache instance — sharing it
   * across instances (or processes) would mean O_TRUNC corrupts the other
   * holder's mappings (SIGBUS on access). initialize_for_runtime() takes
   * an advisory exclusive flock on the file; if the lock fails the mmap
   * path is disabled for this instance and allocations fall back to heap.
   */
  void set_packed_cache_path(const std::string& path);

  /** Save packed weight index so subsequent loads skip packing. */
  Error save_packed_index();

 private:
  static constexpr uint32_t kCacheMagic = 0x58505743; // "XPWC"
  // Bump when the on-disk layout (footer or per-entry record) changes.
  // v2: per-entry seed added — old v1 files don't carry seeds and would
  // load with seed=0, mismatching every fresh look_up with a non-zero
  // seed, causing a stampede of re-packs. Reject v1 outright.
  static constexpr uint32_t kCacheVersion = 2;
  bool load_packed_cache();
  void reset_for_fresh_write();
  void release_entry(void* packed_data_ptr);
  void full_unload();
  // Runtime Allocator used to reserve memory for packed weights
  MemoryAllocator* runtime_allocator_;

  // Named Data Map used to load named data
  const NamedDataMap* named_data_map_;

  // Map of unpacked pointers to the data name
  std::unordered_map<const void*, std::string> unpacked_data_to_name_;
  // Map of data names to offset into the packed data
  std::unordered_map<std::string, PackedDataMeta> name_to_packed_data_metadata_;
  // Vector holding list of pointers to the packed data
  std::vector<void*> packed_data_ptrs_;
  // vector holding list of strings which are containers for packed_data_ptrs
  std::unordered_map<void*, std::string> packed_pointer_to_container_;
  // Vector hodling list of unpacked freeable buffers
  std::vector<FreeableBuffer> unpacked_data_;
  // xnnpack's weight cache provider
  xnn_weights_cache_provider weights_cache_;
  // whether or not the weight cache is finalized
  bool is_finalized_;

  // File-backed mmap for packed weights. When packed_cache_path_ is set,
  // reserve_space() allocates from this mmap'd file instead of heap.
  // After msync, pages become clean file-backed → 0 phys_footprint.
  //
  std::string packed_cache_path_;
  int packed_file_fd_{-1};
  size_t packed_file_used_{0};
  // True once load_packed_cache() has populated metadata from a saved
  // index, OR once a fresh-write session has been persisted to disk via
  // save_packed_index() (so subsequent inits can load from it).
  bool cache_loaded_{false};
  // Tracks file offset of each file-backed allocation. Used by
  // save_packed_index() to serialize (name → offset, size) index.
  std::unordered_map<void*, size_t> ptr_to_file_offset_;
  struct MmapRegion {
    void* addr;
    size_t size;
  };
  std::vector<MmapRegion> mmap_regions_;
  size_t mmap_regions_synced_{0};
  // Number of regions present at the time of the most recent successful
  // save_packed_index. Used to skip no-op saves: identical bytes would
  // still bump mtime via pwrite/fsync, making the cache file appear
  // modified on every load when nothing has actually changed. A successful
  // save closes packed_file_fd_ before returning, so the no-op check is
  // unreachable except after a load_packed_cache (or fresh-write path)
  // re-opens the fd — both paths populate at least one mmap region, so
  // the "zero regions saved" edge case never lives long enough to matter.
  size_t mmap_regions_at_last_save_{0};
  // For file-backed packed allocations, maps the returned ptr to its index
  // in mmap_regions_, so delete_packed_data() can munmap when ref_count==0.
  std::unordered_map<void*, size_t> file_ptr_to_region_index_;

  // Function pointers to override XNNPACK's default xnn_weights_cache_provider
  // functions.
  static size_t look_up(
      XNNWeightsCache* context,
      const xnn_weights_cache_look_up_key* cache_key);

  static void* reserve_space(XNNWeightsCache* context, size_t n);

  // Heap-backed allocation path. Used when the mmap path is not configured
  // or has failed for this allocation.
  void* reserve_space_heap(size_t n);

  static size_t look_up_or_insert(
      XNNWeightsCache* context,
      const xnn_weights_cache_look_up_key* cache_key,
      void* ptr,
      size_t size);

  static bool is_finalized(XNNWeightsCache* context);

  static void* offset_to_addr(XNNWeightsCache* context, size_t offset);

  static enum xnn_status delete_cache(XNNWeightsCache* context);
};

} // namespace delegate
} // namespace xnnpack
} // namespace backends
} // namespace executorch
