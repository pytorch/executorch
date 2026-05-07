/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/kv_manager.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/platform/assert.h>

using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
using executorch::runtime::TensorInfo;
namespace example {

namespace {
void fill_mask(
    executorch::aten::ScalarType scalar_type,
    std::byte* buf,
    size_t size,
    bool use_pos_value) {
  if (use_pos_value) {
    switch (scalar_type) {
      case executorch::aten::ScalarType::UInt16:
        std::fill_n(reinterpret_cast<uint16_t*>(buf), size, 65535u);
        break;
      case executorch::aten::ScalarType::Byte:
        std::fill_n(reinterpret_cast<uint8_t*>(buf), size, 255u);
        break;
      case executorch::aten::ScalarType::Float:
        std::fill_n(reinterpret_cast<float*>(buf), size, 0.0);
        break;
      default:
        ET_CHECK_MSG(
            false,
            "Unsupported scalar type %s",
            executorch::runtime::toString(scalar_type));
        break;
    }
  } else {
    switch (scalar_type) {
      case executorch::aten::ScalarType::UInt16:
        std::fill_n(reinterpret_cast<uint16_t*>(buf), size, 0u);
        break;
      case executorch::aten::ScalarType::Byte:
        std::fill_n(reinterpret_cast<uint8_t*>(buf), size, 0u);
        break;
      case executorch::aten::ScalarType::Float:
        std::fill_n(reinterpret_cast<float*>(buf), size, -65535.0);
        break;
      default:
        ET_CHECK_MSG(
            false,
            "Unsupported scalar type %s",
            executorch::runtime::toString(scalar_type));
        break;
    }
  }
}
} // namespace

KVManager::KVManager(Metadata metadata, std::unique_ptr<MethodMeta> method_meta)
    : metadata_(metadata) {
  Result<TensorInfo> attention_mask = method_meta->input_tensor_meta(1);
  attention_mask_dtype_ = attention_mask->scalar_type();

  // inputs are [input_tokens, attention_mask, (sliding window attention_mask),
  // (input_pos), kv_caches] search kv_cache in inputs
  for (int i = 2; i < method_meta->num_inputs(); i++) {
    Result<TensorInfo> tensor_meta = method_meta->input_tensor_meta(i);
    // Check if all kv_cache share same dtype
    if (kv_cache_dtype_ != executorch::aten::ScalarType::Undefined &&
        tensor_meta->scalar_type() != kv_cache_dtype_) {
      ET_CHECK_MSG(
          false, "Currently mixed scalar type of kv_cache is not allowed");
    }
    // k_cache: [1, n_heads, head_dim, seq_len]
    size_t tensor_nbytes = tensor_meta->nbytes();
    size_t expected_tensor_nbytes = metadata_.head_dim * metadata_.num_heads *
        metadata_.max_cache_len * getDtypeSize(tensor_meta->scalar_type());
    if (kv_cache_dtype_ == executorch::aten::ScalarType::Undefined &&
        tensor_nbytes == expected_tensor_nbytes) {
      kv_cache_dtype_ = tensor_meta->scalar_type();
      ET_CHECK_MSG(
          kv_cache_dtype_ != executorch::aten::ScalarType::Undefined,
          "kv_cache_scalar_type is undefined");
    }
  }
  k_cache_.resize(metadata_.num_layers);
  v_cache_.resize(metadata_.num_layers);

  // Calculate cache size
  size_t cache_in_bytes = metadata_.num_layers * metadata_.num_heads *
      metadata_.head_dim * metadata_.max_cache_len *
      getDtypeSize(kv_cache_dtype_);
  size_t cache_out_bytes = metadata_.num_layers * metadata_.num_heads *
      metadata_.head_dim * metadata_.max_ar_len * getDtypeSize(kv_cache_dtype_);
  total_cache_size_ = 2 * (cache_in_bytes + cache_out_bytes);
};

void KVManager::init_attention_mask(
    std::byte* attention_mask,
    const std::vector<int32_t>& attention_map,
    int32_t ar_len,
    int32_t n_past) {
  ET_CHECK_MSG(
      attention_map.size() <= ar_len,
      "The size of attention_map (%zu) doesn't match with ar_len (%d)",
      attention_map.size(),
      ar_len);
  // Clear the attention mask
  fill_mask(
      attention_mask_dtype_,
      attention_mask,
      ar_len * metadata_.context_len,
      /*use_pos_value=*/false);

  // SMART_MASK requires special handling of attention mask
  std::byte* past_ptr = attention_mask;
  std::byte* new_ptr = attention_mask +
      (metadata_.context_len - ar_len) * getDtypeSize(attention_mask_dtype_);
  // All inputs will necessarily attend to n_past and itself
  for (int i = 0; i < ar_len; i++) {
    // Iterate across ar_len
    if (attention_map[i] < 0) {
      // If negative, attend to only past tokens
      fill_mask(
          attention_mask_dtype_,
          past_ptr,
          n_past,
          /*use_pos_value=*/true);
    } else {
      // If positive, copy attention map from (relative to 0th input) parent
      // Parent token index
      const int32_t pidx = attention_map[i];
      std::byte* parent_ptr = attention_mask +
          pidx * metadata_.context_len * getDtypeSize(attention_mask_dtype_);
      ;
      std::memcpy(
          past_ptr,
          parent_ptr,
          metadata_.context_len * getDtypeSize(attention_mask_dtype_));
    }
    // Attend to itself
    fill_mask(
        attention_mask_dtype_,
        new_ptr + i * getDtypeSize(attention_mask_dtype_),
        1,
        /*use_pos_value=*/true);
    past_ptr += metadata_.context_len * getDtypeSize(attention_mask_dtype_);
    new_ptr += metadata_.context_len * getDtypeSize(attention_mask_dtype_);
  }
}

void KVManager::init_attention_mask(
    std::byte* attention_mask,
    const std::vector<int32_t>& attention_map,
    int32_t ar_len,
    int32_t n_past,
    int32_t sliding_window,
    const std::vector<int32_t>& position_offset) {
  ET_CHECK_MSG(
      attention_map.size() <= ar_len,
      "The size of attention_map (%zu) doesn't match with ar_len (%d)",
      attention_map.size(),
      ar_len);
  // Clear the attention mask
  fill_mask(
      attention_mask_dtype_,
      attention_mask,
      ar_len * metadata_.context_len,
      /*use_pos_value=*/false);

  // SMART_MASK requires special handling of attention mask
  std::byte* past_ptr = attention_mask;
  std::byte* new_ptr = attention_mask +
      (metadata_.context_len - ar_len) * getDtypeSize(attention_mask_dtype_);
  // All inputs will necessarily attend to n_past and itself
  for (int i = 0; i < ar_len; i++) {
    // Iterate across ar_len
    if (attention_map[i] < 0) {
      // If negative, attend to only past tokens
      fill_mask(
          attention_mask_dtype_,
          past_ptr,
          n_past,
          /*use_pos_value=*/true);
    } else {
      // If positive, copy attention map from (relative to 0th input) parent
      // Parent token index
      const int32_t pidx = attention_map[i];
      std::byte* parent_ptr = attention_mask +
          pidx * metadata_.context_len * getDtypeSize(attention_mask_dtype_);
      std::memcpy(
          past_ptr,
          parent_ptr,
          metadata_.context_len * getDtypeSize(attention_mask_dtype_));
    }
    // Attend to itself
    fill_mask(
        attention_mask_dtype_,
        new_ptr + i * getDtypeSize(attention_mask_dtype_),
        1,
        /*use_pos_value=*/true);

    // mask by limitation of sliding_window
    int32_t available_context_len = position_offset.empty()
        ? sliding_window - (i + 1) - n_past
        : sliding_window - (position_offset[i] + 1) - n_past;
    // if available_context_len is less than 0, it means we need to mask some
    // tokens in the past to avoid exceeding the sliding window
    if (available_context_len < 0) {
      fill_mask(
          attention_mask_dtype_,
          past_ptr,
          -available_context_len,
          /*use_pos_value=*/false);
    }

    past_ptr += metadata_.context_len * getDtypeSize(attention_mask_dtype_);
    new_ptr += metadata_.context_len * getDtypeSize(attention_mask_dtype_);
  }
}

void KVManager::update_attention_mask(
    std::byte* attention_mask,
    int32_t ar_len,
    int32_t n_past,
    int32_t n_update) {
  std::byte* cur_ptr =
      attention_mask + n_past * getDtypeSize(attention_mask_dtype_);

  for (int i = 0; i < ar_len; i++) {
    fill_mask(attention_mask_dtype_, cur_ptr, n_update, /*use_pos_value=*/true);
    cur_ptr += metadata_.context_len * getDtypeSize(attention_mask_dtype_);
  }
}

void KVManager::update_attention_mask(
    std::byte* attention_mask,
    int32_t ar_len,
    int32_t n_past,
    int32_t n_update,
    int32_t sliding_window,
    const std::vector<int32_t>& position_offset) {
  std::byte* cur_ptr =
      attention_mask + n_past * getDtypeSize(attention_mask_dtype_);

  for (int i = 0; i < ar_len; i++) {
    fill_mask(attention_mask_dtype_, cur_ptr, n_update, /*use_pos_value=*/true);
    int32_t available_cache_len = position_offset.empty()
        ? sliding_window - (i + 1)
        : sliding_window - (position_offset[i] + 1);
    if (n_past + n_update > available_cache_len) {
      fill_mask(
          attention_mask_dtype_,
          cur_ptr - n_past * getDtypeSize(attention_mask_dtype_),
          n_past + n_update,
          /*use_pos_value=*/false);
    }
    cur_ptr += metadata_.context_len * getDtypeSize(attention_mask_dtype_);
  }
}

void KVManager::init_cache(IMemAlloc* buffer_manager, int32_t ar_len) {
  cur_ar_len_ = ar_len;
  const size_t cache_in_bytes = metadata_.num_heads * metadata_.head_dim *
      metadata_.max_cache_len * getDtypeSize(kv_cache_dtype_);
  const size_t cache_out_bytes = metadata_.num_heads * metadata_.head_dim *
      metadata_.max_ar_len * getDtypeSize(kv_cache_dtype_);
  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    k_cache_[layer].buffer = buffer_manager->allocate(cache_in_bytes);
    k_cache_[layer].output_buffer = buffer_manager->allocate(cache_out_bytes);
    v_cache_[layer].buffer = buffer_manager->allocate(cache_in_bytes);
    v_cache_[layer].output_buffer = buffer_manager->allocate(cache_out_bytes);
  }
}

void KVManager::rearrange_cache(int32_t ar_len_dst) {
  // Don't need to rearrange if cur_ar_len_ is equal to target ar_len
  if (cur_ar_len_ == ar_len_dst)
    return;
  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    rearrange_key(k_cache_[layer], ar_len_dst);
    rearrange_value(v_cache_[layer], ar_len_dst);
  }
  // rearrange done.
  cur_ar_len_ = ar_len_dst;
}

void KVManager::rearrange_key(KVCache& k_cache, int32_t ar_len_dst) {
  const int32_t src_cache_num = (cur_ar_len_ == metadata_.context_len)
      ? metadata_.context_len
      : metadata_.context_len - cur_ar_len_;
  const int32_t dst_cache_num = metadata_.context_len - ar_len_dst;
  std::byte* k_cache_in_read_ptr = k_cache.buffer;
  std::byte* k_cache_in_write_ptr = k_cache.buffer;
  size_t src_cache_nbytes = src_cache_num * getDtypeSize(kv_cache_dtype_);
  size_t dst_cache_nbytes = dst_cache_num * getDtypeSize(kv_cache_dtype_);
  if (src_cache_num > dst_cache_num) {
    // copy from first dimension
    for (int i = 0; i < metadata_.head_dim * metadata_.num_heads; i++) {
      std::memmove(k_cache_in_write_ptr, k_cache_in_read_ptr, dst_cache_nbytes);
      k_cache_in_read_ptr += src_cache_nbytes;
      k_cache_in_write_ptr += dst_cache_nbytes;
    }
  } else {
    k_cache_in_read_ptr +=
        (metadata_.head_dim * metadata_.num_heads - 1) * src_cache_nbytes;
    k_cache_in_write_ptr +=
        (metadata_.head_dim * metadata_.num_heads - 1) * dst_cache_nbytes;
    // copy from last dimension
    for (int i = 0; i < metadata_.head_dim * metadata_.num_heads; i++) {
      std::memmove(k_cache_in_write_ptr, k_cache_in_read_ptr, src_cache_nbytes);
      k_cache_in_read_ptr -= src_cache_nbytes;
      k_cache_in_write_ptr -= dst_cache_nbytes;
    }
  }
}

void KVManager::rearrange_value(KVCache& v_cache, int32_t ar_len_dst) {
  const int32_t src_cache_num = (cur_ar_len_ == metadata_.context_len)
      ? metadata_.context_len
      : metadata_.context_len - cur_ar_len_;
  const int32_t dst_cache_num = metadata_.context_len - ar_len_dst;
  std::byte* v_cache_in_read_ptr = v_cache.buffer;
  std::byte* v_cache_in_write_ptr = v_cache.buffer;
  size_t src_cache_nbytes = src_cache_num * getDtypeSize(kv_cache_dtype_);
  size_t dst_cache_nbytes = dst_cache_num * getDtypeSize(kv_cache_dtype_);
  if (src_cache_num > dst_cache_num) {
    // copy from first dimension
    for (int i = 0; i < metadata_.num_heads; i++) {
      std::memmove(
          v_cache_in_write_ptr,
          v_cache_in_read_ptr,
          dst_cache_nbytes * metadata_.head_dim);
      v_cache_in_read_ptr += src_cache_nbytes * metadata_.head_dim;
      v_cache_in_write_ptr += dst_cache_nbytes * metadata_.head_dim;
    }
  } else {
    v_cache_in_read_ptr +=
        metadata_.head_dim * (metadata_.num_heads - 1) * src_cache_nbytes;
    v_cache_in_write_ptr +=
        metadata_.head_dim * (metadata_.num_heads - 1) * dst_cache_nbytes;
    // copy from last dimension
    for (int i = 0; i < metadata_.num_heads; i++) {
      std::memmove(
          v_cache_in_write_ptr,
          v_cache_in_read_ptr,
          src_cache_nbytes * metadata_.head_dim);
      v_cache_in_read_ptr -= src_cache_nbytes * metadata_.head_dim;
      v_cache_in_write_ptr -= dst_cache_nbytes * metadata_.head_dim;
    }
  }
}

void KVManager::update_cache(
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
    update_key(k_cache_[layer], n_past, n_update, selected);
    update_value(v_cache_[layer], n_past, n_update, selected);
  }
}

void KVManager::update_key(
    KVCache& k_cache,
    int32_t n_past,
    int32_t n_update,
    const std::vector<bool>& selected) {
  std::byte* write_ptr = k_cache.buffer;
  std::byte* read_ptr = k_cache.output_buffer;
  const int32_t copy_size = n_update * getDtypeSize(kv_cache_dtype_);
  const int32_t iter_size = (cur_ar_len_ == metadata_.context_len)
      ? metadata_.context_len * getDtypeSize(kv_cache_dtype_)
      : (metadata_.context_len - cur_ar_len_) * getDtypeSize(kv_cache_dtype_);
  const int32_t out_size = cur_ar_len_ * getDtypeSize(kv_cache_dtype_);
  const int32_t past_size = n_past * getDtypeSize(kv_cache_dtype_);
  const int32_t n_iter = metadata_.head_dim * metadata_.num_heads;

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
        std::memmove(
            wp,
            rp + ind * getDtypeSize(kv_cache_dtype_),
            getDtypeSize(kv_cache_dtype_));
        wp += getDtypeSize(kv_cache_dtype_);
      }
      write_ptr += iter_size;
      read_ptr += out_size;
    }
  }
}

void KVManager::update_value(
    KVCache& v_cache,
    int32_t n_past,
    int32_t n_update,
    const std::vector<bool>& selected) {
  std::byte* write_ptr = v_cache.buffer;
  std::byte* read_ptr = v_cache.output_buffer;
  const int32_t copy_size =
      n_update * metadata_.head_dim * getDtypeSize(kv_cache_dtype_);
  const int32_t past_size =
      n_past * metadata_.head_dim * getDtypeSize(kv_cache_dtype_);
  const int32_t n_iter = metadata_.num_heads;
  const int32_t iter_size = (cur_ar_len_ == metadata_.context_len)
      ? metadata_.context_len * metadata_.head_dim *
          getDtypeSize(kv_cache_dtype_)
      : (metadata_.context_len - cur_ar_len_) * metadata_.head_dim *
          getDtypeSize(kv_cache_dtype_);
  const int32_t out_size =
      cur_ar_len_ * metadata_.head_dim * getDtypeSize(kv_cache_dtype_);

  write_ptr += past_size;

  if (selected.empty()) {
    for (int i = 0; i < n_iter; i++) {
      std::memcpy(write_ptr, read_ptr, copy_size);
      write_ptr += iter_size;
      read_ptr += out_size;
    }
  } else {
    int32_t update_times = n_update;
    for (int i = 0; i < n_iter; ++i) {
      auto wp = write_ptr, rp = read_ptr;
      for (auto sel : selected) {
        if (sel) {
          std::memcpy(
              wp, rp, metadata_.head_dim * getDtypeSize(kv_cache_dtype_));
          wp += metadata_.head_dim * getDtypeSize(kv_cache_dtype_);
          update_times--;
          if (update_times == 0)
            break;
        }
        rp += metadata_.head_dim * getDtypeSize(kv_cache_dtype_);
      }
      write_ptr += iter_size;
      read_ptr += out_size;
    }
  }
}

} // namespace example
