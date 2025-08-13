/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/imem_alloc.h>
#include <cstdint>
#include <memory>
#include <vector>

namespace example {

// Structure to hold key-value cache buffers
template <typename T>
struct KVCache {
  T* buffer;
  T* output_buffer;
};

// Enumeration for key-value manager modes
enum KVManagerMode { SMART_MASK = 0x0, SHIFT_POINTER = 0x1 };
/**
 * @class KVManager
 * @brief Class for kv cache update, rearrangement, and buffer allocatation.
 */
template <typename T>
class KVManager {
 public:
  struct Metadata {
    int32_t context_len;
    int64_t head_dim;
    int32_t max_ar_len;
    int32_t max_cache_len;
    int64_t num_heads;
    int64_t num_layers;
  };
  KVManager(KVManagerMode kv_updater, Metadata metadata);

  /**
   * @brief Allocate buffer for KV cache and set the cur_ar_len_.
   * @param buffer_manager Pointer to IMemAlloc instance which depends on
   * kv_updater.
   * @param ar_len Length of input tokens.
   */
  void init_cache(IMemAlloc* buffer_manager, int32_t ar_len);

  /**
   * @brief Switch key and value cache from AR-cur to AR-dst.
   * @param ar_len_dst Target length of input tokens.
   */
  void rearrange_cache(int32_t ar_len_dst);

  /**
   * @brief Initialize attention mask based on kv manager mode, and attention
   * map.
   * For example,
   * ar_len = 4, CL = 6, n_past = 0,
   * attention map: {-1, 0, 1, 2} and SMART_MASK.
   * Attention_mask will be:
   * [     0     0 65535     0     0     0 ]
   * [     0     0 65535 65535     0     0 ]
   * [     0     0 65535 65535 65535     0 ]
   * [     0     0 65535 65535 65535 65535 ]
   * @param attention_mask Pointer to the attention mask array to be
   * initialized.
   * @param attention_map Vector containing the attention map values. The shape
   * of attention map should be [ar_len].
   * @param ar_len Length of input tokens.
   * @param n_past Number of past elements in the cache.
   */
  void init_attention_mask(
      uint16_t* attention_mask,
      const std::vector<int32_t>& attention_map,
      int32_t ar_len,
      int32_t n_past);

  /**
   * @brief Update attention mask based on kv manager mode, and n_update.
   * @param attention_mask Pointer to the attention mask array to be
   * initialized.
   * @param ar_len Length of input tokens.
   * @param n_past Number of past elements in the cache.
   * @param n_update Number of elements to be updated.
   */
  void update_attention_mask(
      uint16_t* attention_mask,
      int32_t ar_len,
      int32_t n_past,
      int32_t n_update);

  /**
   * @brief Reset the data pointer of the I/O cache tensor based on number of
   * past cache, kv manager mode, current ar length and KV cache data pointer
   * for SHIFT_POINTER mode.
   * @param k_cache_in Reference to the input key cache TensorImpl vector.
   * @param k_cache_out Reference to the output key cache TensorImpl vector.
   * @param v_cache_in Reference to the input value cache TensorImpl vector.
   * @param v_cache_out Reference to the output value cache TensorImpl vector.
   * @param ar_len Length of input tokens.
   * @param n_past Number of past elements in the cache.
   * @return Returns true if the data pointer is updated; otherwise, returns
   * false.
   */
  bool update_cache_tensor(
      std::vector<std::vector<std::unique_ptr<executorch::aten::TensorImpl>>>&
          k_cache_in,
      std::vector<std::vector<std::unique_ptr<executorch::aten::TensorImpl>>>&
          k_cache_out,
      std::vector<std::vector<std::unique_ptr<executorch::aten::TensorImpl>>>&
          v_cache_in,
      std::vector<std::vector<std::unique_ptr<executorch::aten::TensorImpl>>>&
          v_cache_out,
      int32_t ar_len,
      int32_t n_past);

  /**
   * @brief Based on cur_ar_len_ to update cache
   * @param ar_len Length of input tokens.
   * @param n_past Number of past elements in the cache.
   * @param n_update Number of elements to be updated.
   * @param selected Indicate which position to be updated
   */
  void update_cache(
      int32_t ar_len,
      int32_t n_past,
      int32_t n_update,
      const std::vector<bool>& selected);

  const std::vector<std::vector<KVCache<T>>>& get_k_cache_() const {
    return k_cache_;
  }
  const std::vector<std::vector<KVCache<T>>>& get_v_cache_() const {
    return v_cache_;
  }

  inline const size_t total_cache_size_in_bytes() const {
    return total_cache_size_;
  }

 private:
  // Helper functions to rearrange and update key and value caches
  void rearrange_key(KVCache<T>& k_cache, int32_t ar_len_dst);
  void rearrange_value(KVCache<T>& v_cache, int32_t ar_len_dst);
  void update_key(
      KVCache<T>& k_cache,
      int32_t n_past,
      int32_t n_update,
      const std::vector<bool>& selected);
  void update_value(
      KVCache<T>& v_cache,
      int32_t n_past,
      int32_t n_update,
      const std::vector<bool>& selected);
  KVManagerMode kv_updater_;

  // metadata
  Metadata metadata_;
  size_t total_cache_size_;
  int32_t cur_ar_len_;
  // Store start pointer of k and v cache for input and output
  // input: layer -> head -> head_dim * max_cache_len
  // output: layer -> head -> head_dim * max_ar_len
  std::vector<std::vector<KVCache<T>>> k_cache_;
  std::vector<std::vector<KVCache<T>>> v_cache_;
};
} // namespace example
