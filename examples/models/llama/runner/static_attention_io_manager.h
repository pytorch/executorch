/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <executorch/runtime/core/span.h>
#include <executorch/runtime/executor/method.h>

namespace example {

enum class StaticAttentionUpdateStyle {
  /**
   * KV caches will have valid data at the end of the cache. New elements are
   * added at the end and the start of the cache will slide forward to maintain
   * this invariant. This potentially allows shorter caches to be passed into
   * the model by adjusting the start pointer.
   */
  SLIDING_CACHE,
  /**
   * I/O pointers do not change which can enable persistent memory mapping
   * between AP and NPU.
   */
  SMART_MASK,
};

template <typename T, typename AllocatorT = std::allocator<T>>
class StaticKVCache {
 public:
  /**
   * Helper class to handle KV cache I/O. Assumes batch size 1, same length and
   * head dimension for each cache. Supports multi-turn operation mixing prefill
   * and decode by sharing the same cache between methods with different input
   * length. Create one instance for key caches and another one for value
   * caches.
   */
  StaticKVCache(
      size_t n_caches,
      size_t cache_len,
      size_t head_dim,
      size_t max_input_len = 1,
      bool transpose = false,
      StaticAttentionUpdateStyle style =
          StaticAttentionUpdateStyle::SLIDING_CACHE)
      : n_caches_(n_caches),
        cache_len_(cache_len),
        max_input_len_(max_input_len),
        head_dim_(head_dim),
        transpose_(transpose),
        style_(style),
        input_ptrs_(n_caches_),
        output_ptrs_(n_caches_) {
    if (transpose_) {
      throw std::runtime_error("Not implemented.");
    }

    if (style_ == StaticAttentionUpdateStyle::SLIDING_CACHE) {
      // Allocates on extra copy to accomodate caches sliding forward.
      cache_data_size_ = (n_caches_ + 1) * cache_len_ * head_dim_;
    } else {
      cache_data_size_ = n_caches_ * cache_len_ * head_dim_;
    }
    update_data_size_ = n_caches_ * max_input_len_ * head_dim_;

    cache_data_ = allocator_.allocate(cache_data_size_);
    update_data_ = allocator_.allocate(update_data_size_);
    ET_CHECK(cache_data_ != nullptr);
    ET_CHECK(update_data_ != nullptr);
    init_ptrs();
  }

  StaticKVCache(const StaticKVCache& other) = delete;
  StaticKVCache& operator=(const StaticKVCache& other) = delete;
  StaticKVCache(StaticKVCache&& other) = delete;
  StaticKVCache& operator=(StaticKVCache&& other) = delete;

  ~StaticKVCache() {
    allocator_.deallocate(cache_data_, cache_data_size_);
    allocator_.deallocate(update_data_, update_data_size_);
  }

  /**
   * Set up data pointers for the KV cache related inputs and outputs based on
   * the current state of the cache. Call StaticKVCache<T>::update or
   * StaticKVCache<T>::reset as needed before calling this function.
   */
  void prepare(
      torch::executor::Method& method,
      const std::vector<size_t>& inputIndices,
      const std::vector<size_t>& output_indices) {
    ET_CHECK(inputIndices.size() == output_indices.size());
    auto methodMeta = method.method_meta();
    for (size_t i = 0; i < n_caches_; i++) {
      auto inIdx = inputIndices[i];
      auto outIdx = output_indices[i];
      auto inMeta = methodMeta.input_tensor_meta(inIdx);
      auto outMeta = methodMeta.output_tensor_meta(outIdx);
      ET_CHECK(inMeta.ok());
      ET_CHECK(outMeta.ok());

      auto inSizes = inMeta->sizes();
      auto outSizes = outMeta->sizes();
      ET_CHECK_MSG(inSizes[0] == 1, "Only support batch size 1.");
      ET_CHECK_MSG(outSizes[0] == 1, "Only support batch size 1.");
      if (transpose_) {
        ET_CHECK_MSG(inSizes[1] == head_dim_, "KV head dim mismatch.");
        ET_CHECK_MSG(outSizes[1] == head_dim_, "KV head dim mismatch.");
        ET_CHECK_MSG(inSizes[2] == cache_len_, "Cache length dim mismatch.");
      } else {
        ET_CHECK_MSG(inSizes[2] == head_dim_, "KV head dim mismatch.");
        ET_CHECK_MSG(outSizes[2] == head_dim_, "KV head dim mismatch.");
        ET_CHECK_MSG(inSizes[1] == cache_len_, "Cache length dim mismatch.");
      }

      auto impl = ::executorch::runtime::etensor::TensorImpl(
          inMeta->scalar_type(),
          inMeta->sizes().size(),
          const_cast<torch::executor::TensorImpl::SizesType*>(
              inMeta->sizes().data()),
          input_ptrs_[i],
          const_cast<torch::executor::TensorImpl::DimOrderType*>(
              inMeta->dim_order().data()));
      torch::executor::Tensor t(&impl);
      ET_CHECK(method.set_input(t, inIdx) == torch::executor::Error::Ok);
      ET_CHECK(
          method.set_output_data_ptr(
              output_ptrs_[i], outMeta->nbytes(), outIdx) ==
          torch::executor::Error::Ok);
    }
  }

  /**
   * Update the internal data pointers using the cache updates returned by the
   * model. This length of each individual update cannot exceed the max update
   * length specified during creation, and the total length cannot exceed the
   * cache length.
   */
  void update(
      torch::executor::Method& method,
      const std::vector<size_t>& output_indices,
      size_t update_len) {
    if (valid_len_ + update_len > cache_len_) {
      throw std::runtime_error("Cache capacity exceeded.");
    }

    if (style_ == StaticAttentionUpdateStyle::SLIDING_CACHE) {
      update_sliding_cache(method, output_indices, update_len);
    } else {
      update_smart_mask(method, output_indices, update_len);
    }
  }

  /**
   * Reset the cache. After this the cache contains no valid data and is ready
   * for number of tokens up to the cache length.
   */
  void reset() {
    valid_len_ = 0;
    if (style_ == StaticAttentionUpdateStyle::SLIDING_CACHE) {
      init_ptrs();
    }
  }

 private:
  void init_ptrs() {
    input_ptrs_.resize(n_caches_);
    output_ptrs_.resize(n_caches_);
    for (size_t i = 0; i < n_caches_; i++) {
      input_ptrs_[i] = cache_data_ + i * cache_len_ * head_dim_;
      output_ptrs_[i] = update_data_ + i * max_input_len_ * head_dim_;
    }
  }

  void update_sliding_cache(
      torch::executor::Method& method,
      const std::vector<size_t>& output_indices,
      size_t update_len) {
    ET_CHECK(n_caches_ == output_indices.size());
    for (size_t i = 0; i < n_caches_; i++) {
      const auto& updateTensor =
          method.get_output(output_indices[i]).toTensor();
      ET_CHECK(output_ptrs_[i] == updateTensor.const_data_ptr<T>());
      std::copy(
          output_ptrs_[i],
          output_ptrs_[i] + update_len * head_dim_,
          input_ptrs_[i] + cache_len_ * head_dim_);
      input_ptrs_[i] += update_len * head_dim_;
    }
    valid_len_ += update_len;
  }

  void update_smart_mask(
      torch::executor::Method& method,
      const std::vector<size_t>& output_indices,
      size_t update_len) {
    for (size_t i = 0; i < n_caches_; i++) {
      const auto& updateTensor =
          method.get_output(output_indices[i]).toTensor();
      ET_CHECK(output_ptrs_[i] == updateTensor.mutable_data_ptr<T>());
      std::copy(
          output_ptrs_[i],
          output_ptrs_[i] + update_len * head_dim_,
          input_ptrs_[i] + valid_len_ * head_dim_);
    }
    valid_len_ += update_len;
  }

  size_t n_caches_;
  size_t cache_len_;
  size_t max_input_len_;
  size_t head_dim_;
  bool transpose_;
  StaticAttentionUpdateStyle style_;
  AllocatorT allocator_;
  size_t cache_data_size_;
  T* cache_data_;
  size_t update_data_size_;
  T* update_data_;
  std::vector<T*> input_ptrs_;
  std::vector<T*> output_ptrs_;
  size_t valid_len_ = 0;
};

template <typename T, typename AllocatorT = std::allocator<T>>
class StaticAttentionMask {
 public:
  /**
   * Manages the attention mask for StaticKVCache. Create one mask for each
   * input length. Accepts zero_val and mask_val (which represents -inf) to
   * support quantized mask.
   *
   * The mask shape is (1, input_len, cache_len + input_len). This class manages
   * the slice of the mask at [:, :, :cache_len] to only allow valid cache
   * elements to participate in the attention. User can update the rest of the
   * mask (to implement causal mask for example).
   */
  StaticAttentionMask(
      size_t cache_len,
      size_t input_len,
      size_t head_dim,
      T zero_val,
      T mask_val,
      StaticAttentionUpdateStyle style =
          StaticAttentionUpdateStyle::SLIDING_CACHE)
      : cache_len_(cache_len),
        input_len_(input_len),
        head_dim_(head_dim),
        cache_valid_len_(0),
        zero_val_(zero_val),
        mask_val_(mask_val),
        style_(style) {
    data_size_ = input_len_ * (cache_len_ + input_len_);
    data_ = allocator_.allocate(data_size_);
    ET_CHECK(data_ != nullptr);
    reset();
  }

  StaticAttentionMask(const StaticAttentionMask& other) = delete;
  StaticAttentionMask& operator=(const StaticAttentionMask& other) = delete;
  StaticAttentionMask(StaticAttentionMask&& other) = delete;
  StaticAttentionMask& operator=(StaticAttentionMask&& other) = delete;

  ~StaticAttentionMask() {
    allocator_.deallocate(data_, data_size_);
  }

  /**
   * Reset the mask to the state where the cache contains no valid data.
   */
  void reset() {
    cache_valid_len_ = 0;
    for (size_t i = 0; i < input_len_; i++) {
      auto* p = data_ + (cache_len_ + input_len_) * i;
      std::fill(p, p + cache_len_, mask_val_);
    }
  }

  /**
   * Update the mask to indicate update_len elements have been added to the
   * cache. Note that update_len might be smaller than input_len_ when
   * prefilling with padded inputs.
   */
  void unmask(size_t update_len) {
    if (style_ == StaticAttentionUpdateStyle::SLIDING_CACHE) {
      for (size_t i = 0; i < input_len_; i++) {
        auto* p = data_ + (cache_len_ + input_len_) * i;
        std::fill(
            p + cache_len_ - cache_valid_len_ - update_len,
            p + cache_len_ - cache_valid_len_,
            zero_val_);
      }
    } else {
      for (size_t i = 0; i < input_len_; i++) {
        auto* p = data_ + (cache_len_ + input_len_) * i;
        std::fill(
            p + cache_valid_len_, p + cache_valid_len_ + update_len, zero_val_);
      }
    }
    cache_valid_len_ += update_len;
  }

  void set_causal_mask() {
    for (size_t i = 0; i < input_len_ - 1; i++) {
      auto* p = data_ + (cache_len_ + input_len_) * i;
      std::fill(p + cache_len_, p + cache_len_ + 1 + i, zero_val_);
      std::fill(p + cache_len_ + 1 + i, p + cache_len_ + input_len_, mask_val_);
    }
  }

  T* get() {
    return data_;
  }

 private:
  size_t cache_len_;
  size_t input_len_;
  size_t head_dim_;
  size_t cache_valid_len_;
  T zero_val_;
  T mask_val_;
  StaticAttentionUpdateStyle style_;
  AllocatorT allocator_;
  size_t data_size_ = 0;
  T* data_;
};

template <
    typename CacheT,
    typename MaskT,
    typename RopeT,
    typename CacheAllocatorT = std::allocator<CacheT>,
    typename MaskAllocatorT = std::allocator<MaskT>>
class StaticAttentionIOManager {
 public:
  StaticAttentionIOManager(
      size_t n_caches,
      size_t cache_len,
      size_t head_dim,
      size_t max_input_len,
      size_t rope_freqs_cos_index,
      size_t rope_freqs_sin_index,
      RopeT* rope_freqs_cos,
      RopeT* rope_freqs_sin,
      StaticAttentionUpdateStyle style =
          StaticAttentionUpdateStyle::SLIDING_CACHE)
      : cache_len_(cache_len),
        head_dim_(head_dim),
        style_(style),
        kCaches_(n_caches, cache_len, head_dim, max_input_len, false, style),
        vCaches_(n_caches, cache_len, head_dim, max_input_len, false, style),
        rope_freqs_cos_index_(rope_freqs_cos_index),
        rope_freqs_sin_index_(rope_freqs_sin_index),
        rope_freqs_cos_(rope_freqs_cos),
        rope_freqs_sin_(rope_freqs_sin) {}

  /**
   * Create a new StaticAttentionMask that will be managed by this object.
   */
  StaticAttentionMask<MaskT, MaskAllocatorT>&
  add_mask(size_t input_len, MaskT zero_val, MaskT mask_val) {
    auto it = attentionMasks_.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(input_len),
        std::forward_as_tuple(
            cache_len_, input_len, head_dim_, zero_val, mask_val, style_));
    return it.first->second;
  }

  /**
   * Retrieve a mask suitable for given input length.
   */
  StaticAttentionMask<MaskT, MaskAllocatorT>& getMask(size_t input_len) {
    return attentionMasks_.at(input_len);
  }

  /**
   * Set I/O pointers for KV cache and RoPE freqencies.
   */
  void prepare(
      torch::executor::Method& method,
      const std::vector<size_t>& k_cache_input_indices,
      const std::vector<size_t>& k_cache_output_indices,
      const std::vector<size_t>& v_cache_input_indices,
      const std::vector<size_t>& v_cache_output_indices) {
    kCaches_.prepare(method, k_cache_input_indices, k_cache_output_indices);
    vCaches_.prepare(method, v_cache_input_indices, v_cache_output_indices);
    set_input(
        method,
        rope_freqs_cos_index_,
        rope_freqs_cos_ + input_pos_ * head_dim_ / 2);
    set_input(
        method,
        rope_freqs_sin_index_,
        rope_freqs_sin_ + input_pos_ * head_dim_ / 2);
  }

  /**
   * Update all caches and masks under management to reflect that model produced
   * update_len new elements.
   */
  void update(
      torch::executor::Method& method,
      const std::vector<size_t>& k_cache_output_indices,
      const std::vector<size_t>& v_cache_output_indices,
      size_t update_len) {
    input_pos_ += update_len;
    kCaches_.update(method, k_cache_output_indices, update_len);
    vCaches_.update(method, v_cache_output_indices, update_len);
    for (auto& it : attentionMasks_) {
      it.second.unmask(update_len);
    }
  }

  /**
   * Reset all caches and masks under management.
   */
  void reset() {
    input_pos_ = 0;
    kCaches_.reset();
    vCaches_.reset();
    for (auto& it : attentionMasks_) {
      it.second.reset();
    }
  }

 private:
  template <typename T>
  void set_input(executorch::runtime::Method& method, size_t idx, T* data) {
    auto methodMeta = method.method_meta();
    auto inputMeta = methodMeta.input_tensor_meta(idx);
    auto impl = ::executorch::runtime::etensor::TensorImpl(
        inputMeta->scalar_type(),
        inputMeta->sizes().size(),
        const_cast<executorch::aten::TensorImpl::SizesType*>(
            inputMeta->sizes().data()),
        data,
        const_cast<executorch::aten::TensorImpl::DimOrderType*>(
            inputMeta->dim_order().data()));
    executorch::runtime::etensor::Tensor t(&impl);
    ET_CHECK(method.set_input(t, idx) == executorch::runtime::Error::Ok);
  }

  size_t cache_len_;
  size_t input_len_;
  size_t head_dim_;
  size_t input_pos_;
  StaticAttentionUpdateStyle style_;
  StaticKVCache<CacheT, CacheAllocatorT> kCaches_;
  StaticKVCache<CacheT, CacheAllocatorT> vCaches_;
  std::unordered_map<size_t, StaticAttentionMask<MaskT, MaskAllocatorT>>
      attentionMasks_;
  size_t rope_freqs_cos_index_;
  size_t rope_freqs_sin_index_;
  RopeT* rope_freqs_cos_;
  RopeT* rope_freqs_sin_;
};

} // namespace example
