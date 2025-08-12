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
  size_t offset;
  // Count number of xnn_runtime_t this packed data is used in
  size_t ref_count;
  // true if this packed data was inserted or looked up for the
  // current runtime being created
  bool in_current_runtime;
};

class XNNWeightsCache {
 public:
  XNNWeightsCache();

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
  };

  /**
   * Returns the names of all unpacked data
   */
  inline std::vector<std::string> get_unpacked_data_names() {
    std::vector<std::string> names;
    for (const auto& pair : unpacked_data_to_name_) {
      names.push_back(pair.second);
    }
    return names;
  };

  /**
   * Returns the packed data names
   */
  inline std::vector<std::string> get_packed_data_names() {
    std::vector<std::string> names;
    for (const auto& pair : name_to_packed_data_metadata_) {
      names.push_back(pair.first);
    }
    return names;
  };

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

 private:
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

  // Function pointers to override XNNPACK's default xnn_weights_cache_provider
  // functions.
  static size_t look_up(
      XNNWeightsCache* context,
      const xnn_weights_cache_look_up_key* cache_key);

  static void* reserve_space(XNNWeightsCache* context, size_t n);

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
