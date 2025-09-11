/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/kv_manager.h>
#include <executorch/runtime/platform/assert.h>
namespace example {
template <typename T>
KVManager<T>::KVManager(KVManagerMode kv_updater, Metadata metadata)
    : kv_updater_(kv_updater), metadata_(metadata) {
  k_cache_.resize(
      metadata_.num_layers, std::vector<KVCache<T>>(metadata_.num_heads));
  v_cache_.resize(
      metadata_.num_layers, std::vector<KVCache<T>>(metadata_.num_heads));

  // Calculate cache size
  switch (kv_updater_) {
    case KVManagerMode::SMART_MASK: {
      size_t cache_in_bytes = metadata_.num_layers * metadata_.num_heads *
          metadata_.head_dim * metadata_.max_cache_len * sizeof(T);
      size_t cache_out_bytes = metadata_.num_layers * metadata_.num_heads *
          metadata_.head_dim * metadata_.max_ar_len * sizeof(T);
      total_cache_size_ = 2 * (cache_in_bytes + cache_out_bytes);
      break;
    }
    case KVManagerMode::SHIFT_POINTER: {
      size_t k_cache_in_bytes = metadata_.num_layers * metadata_.num_heads *
          (metadata_.head_dim + 1) * metadata_.max_cache_len * sizeof(T);
      size_t k_cache_out_bytes = metadata_.num_layers * metadata_.num_heads *
          metadata_.head_dim * metadata_.max_ar_len * sizeof(T);
      // Use the same memory for input and output of value cache in shift
      // pointer mode. Note that using context length to prevent exceeding the
      // range when the AR-N model updates the last block in shift pointer
      // mode.
      size_t v_cache_bytes = metadata_.num_layers * (metadata_.num_heads + 1) *
          metadata_.head_dim * metadata_.context_len * sizeof(T);
      total_cache_size_ = k_cache_in_bytes + k_cache_out_bytes + v_cache_bytes;
      break;
    }
    default:
      break;
  }
};

template <typename T>
void KVManager<T>::init_attention_mask(
    uint16_t* attention_mask,
    const std::vector<int32_t>& attention_map,
    int32_t ar_len,
    int32_t n_past) {
  ET_CHECK_MSG(
      attention_map.size() <= ar_len,
      "The size of attention_map (%zu) doesn't match with ar_len (%d)",
      attention_map.size(),
      ar_len);
  uint16_t neg_val = 0;
  uint16_t pos_val = 65535;
  // Clear the attention mask
  std::fill_n(attention_mask, ar_len * metadata_.context_len, neg_val);

  // SMART_MASK requires special handling of attention mask
  switch (kv_updater_) {
    case KVManagerMode::SMART_MASK: {
      uint16_t* past_ptr = attention_mask;
      uint16_t* new_ptr = attention_mask + (metadata_.context_len - ar_len);
      // All inputs will necessarily attend to n_past and itself
      for (int i = 0; i < ar_len; i++) {
        // Iterate across ar_len
        if (attention_map[i] < 0) {
          // If negative, attend to only past tokens
          std::fill_n(past_ptr, n_past, pos_val);
        } else {
          // If positive, copy attention map from (relative to 0th input) parent
          // Parent token index
          const int32_t pidx = attention_map[i];
          uint16_t* parent_ptr = attention_mask + pidx * metadata_.context_len;
          std::memcpy(
              past_ptr, parent_ptr, metadata_.context_len * sizeof(uint16_t));
        }
        // Attend to itself
        new_ptr[i] = pos_val;
        past_ptr += metadata_.context_len;
        new_ptr += metadata_.context_len;
      }
      break;
    }
    case KVManagerMode::SHIFT_POINTER: {
      // Only fill in ar_len. Rest will be padding
      const size_t attn_row_start = metadata_.context_len - n_past - ar_len;
      for (int i = 0; i < ar_len; i++) {
        uint16_t* cur_ptr =
            attention_mask + i * metadata_.context_len + attn_row_start;
        // Attend to itself
        cur_ptr[n_past + i] = pos_val;
        if (attention_map[i] < 0) {
          // If negative, attend to only past tokens
          std::fill_n(cur_ptr, n_past, pos_val);
        } else {
          // If positive, copy attention map from (relative to 0th input) parent
          // Parent token index
          const int32_t pidx = attention_map[i];
          uint16_t* parent_ptr =
              attention_mask + pidx * metadata_.context_len + attn_row_start;
          std::memcpy(
              cur_ptr, parent_ptr, (n_past + pidx + 1) * sizeof(uint16_t));
        }
      }
      break;
    }
    default:
      break;
  }
}

template <typename T>
void KVManager<T>::init_attention_mask(
    uint16_t* attention_mask,
    const std::vector<int32_t>& attention_map,
    int32_t ar_len,
    int32_t n_past,
    int32_t sliding_window) {
  ET_CHECK_MSG(
      attention_map.size() <= ar_len,
      "The size of attention_map (%zu) doesn't match with ar_len (%d)",
      attention_map.size(),
      ar_len);
  uint16_t neg_val = 0;
  uint16_t pos_val = 65535;
  // Clear the attention mask
  std::fill_n(attention_mask, ar_len * metadata_.context_len, neg_val);

  // SMART_MASK requires special handling of attention mask
  switch (kv_updater_) {
    case KVManagerMode::SMART_MASK: {
      uint16_t* past_ptr = attention_mask;
      uint16_t* new_ptr = attention_mask + (metadata_.context_len - ar_len);
      // All inputs will necessarily attend to n_past and itself
      for (int i = 0; i < ar_len; i++) {
        // Iterate across ar_len
        if (attention_map[i] < 0) {
          // If negative, attend to only past tokens
          std::fill_n(past_ptr, n_past, pos_val);
        } else {
          // If positive, copy attention map from (relative to 0th input) parent
          // Parent token index
          const int32_t pidx = attention_map[i];
          uint16_t* parent_ptr = attention_mask + pidx * metadata_.context_len;
          std::memcpy(
              past_ptr, parent_ptr, metadata_.context_len * sizeof(uint16_t));
        }
        // Attend to itself
        new_ptr[i] = pos_val;

        // mask by limitation of sliding_window
        int32_t avalible_context_len = sliding_window - (i + 1) - n_past;
        if (n_past > avalible_context_len) {
          std::fill_n(past_ptr, n_past - avalible_context_len, neg_val);
        }

        past_ptr += metadata_.context_len;
        new_ptr += metadata_.context_len;
      }
      break;
    }
    case KVManagerMode::SHIFT_POINTER: {
      // Only fill in ar_len. Rest will be padding
      const size_t attn_row_start = metadata_.context_len - n_past - ar_len;
      for (int i = 0; i < ar_len; i++) {
        uint16_t* cur_ptr =
            attention_mask + i * metadata_.context_len + attn_row_start;
        // Attend to itself
        cur_ptr[n_past + i] = pos_val;
        if (attention_map[i] < 0) {
          // If negative, attend to only past tokens
          std::fill_n(cur_ptr, n_past, pos_val);
        } else {
          // If positive, copy attention map from (relative to 0th input) parent
          // Parent token index
          const int32_t pidx = attention_map[i];
          uint16_t* parent_ptr =
              attention_mask + pidx * metadata_.context_len + attn_row_start;
          std::memcpy(
              cur_ptr, parent_ptr, (n_past + pidx + 1) * sizeof(uint16_t));
        }
      }
      break;
    }
    default:
      break;
  }
}

template <typename T>
void KVManager<T>::update_attention_mask(
    uint16_t* attention_mask,
    int32_t ar_len,
    int32_t n_past,
    int32_t n_update) {
  uint16_t pos_val = 65535;
  uint16_t* cur_ptr = attention_mask;
  if (kv_updater_ == KVManagerMode::SMART_MASK)
    cur_ptr += n_past;
  if (kv_updater_ == KVManagerMode::SHIFT_POINTER)
    cur_ptr += metadata_.context_len - n_past - ar_len - n_update;

  for (int i = 0; i < ar_len; i++) {
    std::fill_n(cur_ptr, n_update, pos_val);
    cur_ptr += metadata_.context_len;
  }
}

template <typename T>
void KVManager<T>::update_attention_mask(
    uint16_t* attention_mask,
    int32_t ar_len,
    int32_t n_past,
    int32_t n_update,
    int32_t sliding_window) {
  uint16_t pos_val = 65535;
  uint16_t neg_val = 0;
  uint16_t* cur_ptr = attention_mask;
  if (kv_updater_ == KVManagerMode::SMART_MASK)
    cur_ptr += n_past;
  if (kv_updater_ == KVManagerMode::SHIFT_POINTER)
    cur_ptr += metadata_.context_len - n_past - ar_len - n_update;

  for (int i = 0; i < ar_len; i++) {
    std::fill_n(cur_ptr, n_update, pos_val);
    int32_t avalible_cache_len = sliding_window - (i + 1);
    if (kv_updater_ == KVManagerMode::SMART_MASK) {
      if (n_past + n_update > avalible_cache_len) {
        std::fill_n(
            cur_ptr - n_past, n_past + n_update - avalible_cache_len, neg_val);
      }
    } else if (kv_updater_ == KVManagerMode::SHIFT_POINTER) {
      if (std::abs(n_past + ar_len) > avalible_cache_len) {
        int32_t n_invalid = n_past - avalible_cache_len;
        std::fill_n(
            cur_ptr, std::abs(n_past + ar_len) - avalible_cache_len, neg_val);
      }

      cur_ptr += metadata_.context_len;
    }
  }
}

template <typename T>
void KVManager<T>::init_cache(IMemAlloc* buffer_manager, int32_t ar_len) {
  cur_ar_len_ = ar_len;
  const size_t max_in_cache_block_in_bytes =
      metadata_.max_cache_len * sizeof(T);
  const size_t max_out_cache_block_in_bytes = metadata_.max_ar_len * sizeof(T);

  switch (kv_updater_) {
    case KVManagerMode::SMART_MASK: {
      const size_t cache_in_bytes =
          metadata_.head_dim * max_in_cache_block_in_bytes;
      const size_t cache_out_bytes =
          metadata_.head_dim * max_out_cache_block_in_bytes;
      for (int layer = 0; layer < metadata_.num_layers; ++layer) {
        for (int head = 0; head < metadata_.num_heads; ++head) {
          // Allocate buffer for key cache and value cache
          T* single_layer_k_cache_in =
              reinterpret_cast<T*>(buffer_manager->allocate(cache_in_bytes));
          T* single_layer_k_cache_out =
              reinterpret_cast<T*>(buffer_manager->allocate(cache_out_bytes));
          T* single_layer_v_cache_in =
              reinterpret_cast<T*>(buffer_manager->allocate(cache_in_bytes));
          T* single_layer_v_cache_out =
              reinterpret_cast<T*>(buffer_manager->allocate(cache_out_bytes));

          k_cache_[layer][head].buffer = single_layer_k_cache_in;
          k_cache_[layer][head].output_buffer = single_layer_k_cache_out;
          v_cache_[layer][head].buffer = single_layer_v_cache_in;
          v_cache_[layer][head].output_buffer = single_layer_v_cache_out;
        }
      }
      break;
    }
    case KVManagerMode::SHIFT_POINTER: {
      const size_t k_cache_in_size_in_bytes = metadata_.num_heads *
          (metadata_.head_dim + 1) * max_in_cache_block_in_bytes;
      const size_t k_cache_out_size_in_bytes = metadata_.num_heads *
          metadata_.head_dim * max_out_cache_block_in_bytes;
      const size_t v_cache_size_in_bytes = (metadata_.num_heads + 1) *
          metadata_.head_dim * metadata_.context_len * sizeof(T);
      const int32_t single_head_size_in =
          metadata_.head_dim * metadata_.max_cache_len;
      const int32_t single_head_size_out =
          metadata_.head_dim * metadata_.max_ar_len;
      for (int layer = 0; layer < metadata_.num_layers; ++layer) {
        // Allocate buffer for key cache and value cache
        T* single_layer_k_cache_in = reinterpret_cast<T*>(
            buffer_manager->allocate(k_cache_in_size_in_bytes));
        T* single_layer_k_cache_out = reinterpret_cast<T*>(
            buffer_manager->allocate(k_cache_out_size_in_bytes));
        // Note that using context length to prevent exceeding the range when
        // the AR-N model updates the last block in shift pointer mode.
        T* single_layer_v_cache = reinterpret_cast<T*>(
            buffer_manager->allocate(v_cache_size_in_bytes));
        for (int head = 0; head < metadata_.num_heads; ++head) {
          k_cache_[layer][head].buffer = single_layer_k_cache_in +
              head * (metadata_.head_dim + 1) * metadata_.max_cache_len;
          k_cache_[layer][head].output_buffer =
              single_layer_k_cache_out + head * single_head_size_out;
          // v_cache:
          // |cache_gap|h1_v_in_ptr|cache_len|h1_v_out_ptr|cache_gap|h2_v_in_ptr|cache_len|h2_v_out_ptr|...|
          const int32_t cache_gap = (cur_ar_len_ == metadata_.context_len)
              ? 0
              : metadata_.max_cache_len - (metadata_.context_len - cur_ar_len_);
          v_cache_[layer][head].buffer = single_layer_v_cache +
              head * metadata_.head_dim * metadata_.context_len +
              cache_gap * metadata_.head_dim;
          v_cache_[layer][head].output_buffer = single_layer_v_cache +
              head * metadata_.head_dim * metadata_.context_len +
              single_head_size_in;
        }
      }
      break;
    }
    default:
      break;
  }
}

template <typename T>
void KVManager<T>::rearrange_cache(int32_t ar_len_dst) {
  // Don't need to rearrange if cur_ar_len_ is equal to target ar_len
  if (cur_ar_len_ == ar_len_dst)
    return;
  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    for (int head = 0; head < metadata_.num_heads; ++head) {
      rearrange_key(k_cache_[layer][head], ar_len_dst);
      rearrange_value(v_cache_[layer][head], ar_len_dst);
    }
  }
  // rearrange done.
  cur_ar_len_ = ar_len_dst;
}

template <typename T>
void KVManager<T>::rearrange_key(KVCache<T>& k_cache, int32_t ar_len_dst) {
  // The output of key cache doesn't need to rearrange for both of SMART_MASK
  // and SHIFT_POINTER
  const int32_t src_cache_num = (cur_ar_len_ == metadata_.context_len)
      ? metadata_.context_len
      : metadata_.context_len - cur_ar_len_;
  const int32_t dst_cache_num = metadata_.context_len - ar_len_dst;
  T* k_cache_in_read_ptr = k_cache.buffer;
  T* k_cache_in_write_ptr = k_cache.buffer;

  if (src_cache_num > dst_cache_num) {
    if (kv_updater_ == KVManagerMode::SHIFT_POINTER) {
      // Left padded KV$
      k_cache_in_read_ptr += src_cache_num;
      k_cache_in_write_ptr += dst_cache_num;
    }
    // copy from first dimension
    for (int i = 0; i < metadata_.head_dim; i++) {
      std::memmove(k_cache_in_write_ptr, k_cache_in_read_ptr, dst_cache_num);
      k_cache_in_read_ptr += src_cache_num;
      k_cache_in_write_ptr += dst_cache_num;
    }
  } else {
    k_cache_in_read_ptr += (metadata_.head_dim - 1) * src_cache_num;
    k_cache_in_write_ptr += (metadata_.head_dim - 1) * dst_cache_num;
    if (kv_updater_ == KVManagerMode::SHIFT_POINTER) {
      k_cache_in_read_ptr += src_cache_num;
      k_cache_in_write_ptr += dst_cache_num;
    }
    // copy from last dimension
    for (int i = 0; i < metadata_.head_dim; i++) {
      std::memmove(k_cache_in_write_ptr, k_cache_in_read_ptr, src_cache_num);
      k_cache_in_read_ptr -= src_cache_num;
      k_cache_in_write_ptr -= dst_cache_num;
    }
  }
}

template <typename T>
void KVManager<T>::rearrange_value(KVCache<T>& v_cache, int32_t ar_len_dst) {
  // The input and output of the value cache don't need to rearrange for both
  // SMART_MASK and SHIFT_POINTER. However, the input pointer of the value cache
  // needs to be reset by ar_len_dst in SHIFT_POINTER mode. The output pointer
  // of the value cache remains unchanged regardless of ar_len.
  const int32_t ar_gap = (cur_ar_len_ == metadata_.context_len)
      ? ar_len_dst
      : ar_len_dst - cur_ar_len_;
  if (kv_updater_ == KVManagerMode::SHIFT_POINTER) {
    v_cache.buffer = v_cache.buffer + ar_gap * metadata_.head_dim;
  }
}

template <typename T>
bool KVManager<T>::update_cache_tensor(
    std::vector<std::vector<std::unique_ptr<executorch::aten::TensorImpl>>>&
        k_cache_in,
    std::vector<std::vector<std::unique_ptr<executorch::aten::TensorImpl>>>&
        k_cache_out,
    std::vector<std::vector<std::unique_ptr<executorch::aten::TensorImpl>>>&
        v_cache_in,
    std::vector<std::vector<std::unique_ptr<executorch::aten::TensorImpl>>>&
        v_cache_out,
    int32_t ar_len,
    int32_t n_past) {
  ET_CHECK_MSG(
      cur_ar_len_ == ar_len,
      "Current AR length (%d) is not matched with target AR length (%d). Please rearrange cache first.",
      cur_ar_len_,
      ar_len);
  bool updated = false;
  // Data pointer in the tensors need to update only for SHIFT_POINTER mode
  // The BERT model does not update the cache tensor because it does not use KV
  // cache inputs.
  if (kv_updater_ == KVManagerMode::SHIFT_POINTER &&
      metadata_.context_len != cur_ar_len_) {
    for (int layer = 0; layer < metadata_.num_layers; ++layer) {
      for (int head = 0; head < metadata_.num_heads; ++head) {
        k_cache_in[layer][head]->set_data(
            k_cache_[layer][head].buffer + n_past);
        v_cache_in[layer][head]->set_data(
            v_cache_[layer][head].buffer + n_past * metadata_.head_dim);
        v_cache_out[layer][head]->set_data(
            v_cache_[layer][head].output_buffer + n_past * metadata_.head_dim);
      }
    }
    updated = true;
  }
  return updated;
}

template <typename T>
void KVManager<T>::update_cache(
    int32_t ar_len,
    int32_t n_past,
    int32_t n_update,
    const std::vector<bool>& selected) {
  ET_CHECK_MSG(
      cur_ar_len_ == ar_len,
      "Current AR length (%d) is not matched with target AR length (%d). Please rearrange cache first.",
      cur_ar_len_,
      ar_len);
  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    for (int head = 0; head < metadata_.num_heads; ++head) {
      update_key(k_cache_[layer][head], n_past, n_update, selected);
      update_value(v_cache_[layer][head], n_past, n_update, selected);
    }
  }
}

template <typename T>
void KVManager<T>::update_key(
    KVCache<T>& k_cache,
    int32_t n_past,
    int32_t n_update,
    const std::vector<bool>& selected) {
  T* write_ptr = k_cache.buffer;
  T* read_ptr = k_cache.output_buffer;
  const int32_t copy_size = n_update * sizeof(T);
  const int32_t iter_size = (cur_ar_len_ == metadata_.context_len)
      ? metadata_.context_len
      : metadata_.context_len - cur_ar_len_;
  const int32_t out_size = cur_ar_len_;
  const int32_t past_size = n_past;
  const int32_t n_iter = metadata_.head_dim;

  if (kv_updater_ == KVManagerMode::SHIFT_POINTER)
    write_ptr += iter_size + past_size;
  if (kv_updater_ == KVManagerMode::SMART_MASK)
    write_ptr += past_size;
  if (selected.empty()) {
    for (int i = 0; i < n_iter; ++i) {
      std::memcpy(write_ptr, read_ptr, copy_size);
      write_ptr += iter_size;
      read_ptr += out_size;
    }
  } else {
    std::vector<int32_t> true_indices(n_update);
    for (int i = 0, j = 0; i < selected.size() && j < n_update; ++i) {
      if (selected[i]) {
        true_indices[j++] = i;
      }
    }
    for (int i = 0; i < n_iter; ++i) {
      auto wp = write_ptr, rp = read_ptr;
      for (auto ind : true_indices) {
        *wp++ = rp[ind];
      }
      write_ptr += iter_size;
      read_ptr += out_size;
    }
  }
}

template <typename T>
void KVManager<T>::update_value(
    KVCache<T>& v_cache,
    int32_t n_past,
    int32_t n_update,
    const std::vector<bool>& selected) {
  T* write_ptr = v_cache.buffer;
  T* read_ptr = v_cache.output_buffer;
  const int32_t copy_size = n_update * metadata_.head_dim * sizeof(T);
  const int32_t past_size = n_past * metadata_.head_dim;

  if (kv_updater_ == KVManagerMode::SMART_MASK)
    write_ptr += past_size;

  // Update the value cache for lookahead decoding in SHIFT_POINTER mode
  if (kv_updater_ == KVManagerMode::SHIFT_POINTER) {
    read_ptr += past_size;
    write_ptr = read_ptr;
  }

  if (selected.empty()) {
    // In general, value cache doesn't need to copy for SHIFT_POINTER mode
    if (kv_updater_ == KVManagerMode::SHIFT_POINTER)
      return;
    std::memcpy(write_ptr, read_ptr, copy_size);
  } else {
    int32_t update_times = n_update;
    auto wp = write_ptr, rp = read_ptr;
    for (auto sel : selected) {
      if (sel) {
        std::memcpy(wp, rp, metadata_.head_dim * sizeof(T));
        wp += metadata_.head_dim;
        update_times--;
        if (update_times == 0)
          break;
      }
      rp += metadata_.head_dim;
    }
  }
}

// Explicit instantiations
template class KVManager<uint16_t>;
template class KVManager<uint8_t>;

} // namespace example
