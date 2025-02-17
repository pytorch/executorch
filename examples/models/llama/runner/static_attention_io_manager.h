// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <executorch/runtime/core/span.h>
#include <executorch/runtime/executor/method.h>

namespace example {

template <typename T, typename AllocatorT = std::allocator<T>>
class StaticKVCache {
 public:
  /**
   * Helper class to handle KV cache I/O. Assumes batch size 1, same context
   * length and head dimension for each cache. Supports hybrid operation mixing
   * prefill and decode. Create one instance for key caches and another one for
   * value caches.
   */
  StaticKVCache(
      size_t n_caches,
      size_t cache_len,
      size_t head_dim,
      size_t max_input_len = 1,
      bool transpose = false)
      : n_caches_(n_caches),
        cache_len_(cache_len),
        max_input_len_(max_input_len),
        head_dim_(head_dim),
        transpose_(transpose) {
    // Updates are appeneded at the end. Need one extra segment to support the
    // sliding window.
    data_size_ = (n_caches_ + 1) * cache_len_ * head_dim_ + max_input_len_;
    data_ = allocator_.allocate(data_size_);
    ET_CHECK(data_ != nullptr);
    reset();
  }

  ~StaticKVCache() {
    allocator_.deallocate(data_, data_size_);
  }

  /**
   * Set up data pointers for the KV cache related inputs and outputs based on
   * the current state of the cache. Call StaticKVCache<T>::update or
   * StaticKVCache<T>::reset first as needed before calling this function.
   */
  void prepare(
      torch::executor::Method& method,
      const std::vector<size_t>& inputIndices,
      const std::vector<size_t>& outputIndices) {
    ET_CHECK(inputIndices.size() == outputIndices.size());
    auto methodMeta = method.method_meta();
    for (size_t i = 0; i < n_caches_; i++) {
      auto inIdx = inputIndices[i];
      auto outIdx = outputIndices[i];
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
   * length specified during the creation, and the total length cannot exceed
   * the context length.
   */
  void update(
      torch::executor::Method& method,
      const std::vector<size_t>& outputIndices,
      size_t update_len) {
    if (valid_len_ + update_len > cache_len_) {
      throw std::runtime_error("Cache capacity exceeded.");
    }

    if (transpose_) {
      throw std::runtime_error("Not implemented.");
    } else {
      updateSeqDim(method, outputIndices, update_len);
    }
    valid_len_ += update_len;
  }

  /**
   * Reset the cache. After this the cache contains no valid data and is ready
   * for number of tokens up to the context length.
   */
  void reset() {
    valid_len_ = 0;
    if (transpose_) {
      throw std::runtime_error("Not implemented.");
    } else {
      initSeqDim();
    }
  }

 private:
  void initSeqDim() {
    auto cacheSize = cache_len_ * head_dim_;
    input_ptrs_.resize(n_caches_);
    output_ptrs_.resize(n_caches_);
    for (size_t i = 0; i < n_caches_; i++) {
      input_ptrs_[i] = data_ + i * cacheSize;
      output_ptrs_[i] = input_ptrs_[i] + cacheSize;
    }
  }

  void updateSeqDim(
      torch::executor::Method& method,
      const std::vector<size_t>& outputIndices,
      size_t update_len) {
    ET_CHECK(n_caches_ == outputIndices.size());
    for (size_t i = 0; i < n_caches_; i++) {
      const auto& updateTensor = method.get_output(outputIndices[i]).toTensor();
      ET_CHECK(
          input_ptrs_[i] + cache_len_ * head_dim_ ==
          updateTensor.mutable_data_ptr<T>());

      input_ptrs_[i] += update_len * head_dim_;
      output_ptrs_[i] += update_len * head_dim_;
    }
  }

  // std::vector<T> pool_;
  size_t n_caches_;
  size_t cache_len_;
  size_t max_input_len_;
  size_t head_dim_;
  bool transpose_;
  AllocatorT allocator_;
  size_t data_size_;
  T* data_;
  std::vector<T*> input_ptrs_;
  std::vector<T*> output_ptrs_;
  size_t valid_len_ = 0;
};

template <typename T, typename AllocatorT = std::allocator<T>>
class StaticAttentionMask {
 public:
  /**
   * Manages the attention mask in the same style of KV cache IO where valid
   * data is at the end of the cache. The mask has shape (1, maxSeqLen,
   * cache_len
   * + maxSeqLen) where maxSeqLen is 1 for decode or the prefill length. Accepts
   * zero_val and mask_val (which represents -inf) to support quantized mask.
   *
   * This class manages the slice of the mask at [:, :, : (cache_len -
   * validCacheLen)]. User can update the rest of the mask to implement causal
   * masking for example.
   */
  StaticAttentionMask(
      size_t cache_len,
      size_t input_len,
      size_t head_dim,
      T zero_val,
      T mask_val)
      : cache_len_(cache_len),
        input_len_(input_len),
        head_dim_(head_dim),
        cache_mask_len_(cache_len_),
        zero_val_(zero_val),
        mask_val_(mask_val) {
    data_size_ = input_len_ * (cache_len_ + input_len_);
    data_ = allocator_.allocate(data_size_);
    ET_CHECK(data_ != nullptr);
    reset();
  }

  /**
   * Reset the mask to the state where the cache contains no valid data.
   */
  void reset() {
    cache_mask_len_ = cache_len_;
    for (size_t i = 0; i < input_len_; i++) {
      auto* p = data_ + (cache_len_ + input_len_) * i;
      std::fill(p, p + cache_len_, mask_val_);
    }
  }

  /**
   * Update the mask to indicate update_len elements have been added to the
   * cache. Note that update_len might be smaller than maxSeqLen when prefilling
   * with padded inputs.
   */
  void updateCacheMask(size_t update_len) {
    for (size_t i = 0; i < input_len_; i++) {
      auto* p = data_ + (cache_len_ + input_len_) * i;
      std::fill(
          p + cache_mask_len_ - update_len, p + cache_mask_len_, zero_val_);
    }
    cache_mask_len_ -= update_len;
  }

  void setCausalMask() {
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
  size_t cache_mask_len_;
  T zero_val_;
  T mask_val_;
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
      RopeT* rope_freqs_sin)
      : cache_len_(cache_len),
        head_dim_(head_dim),
        kCaches_(n_caches, cache_len, head_dim, max_input_len),
        vCaches_(n_caches, cache_len, head_dim, max_input_len),
        rope_freqs_cos_index_(rope_freqs_cos_index),
        rope_freqs_sin_index_(rope_freqs_sin_index),
        rope_freqs_cos_(rope_freqs_cos),
        rope_freqs_sin_(rope_freqs_sin) {}

  StaticAttentionMask<MaskT, MaskAllocatorT>&
  addMask(size_t input_len, MaskT zero_val, MaskT mask_val) {
    auto it = attentionMasks_.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(input_len),
        std::forward_as_tuple(
            cache_len_, input_len, head_dim_, zero_val, mask_val));
    return it.first->second;
  }

  StaticAttentionMask<MaskT, MaskAllocatorT>& getMask(size_t input_len) {
    return attentionMasks_.at(input_len);
  }

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

  void update(
      torch::executor::Method& method,
      const std::vector<size_t>& k_cache_output_indices,
      const std::vector<size_t>& v_cache_output_indices,
      size_t update_len) {
    input_pos_ += update_len;
    kCaches_.update(method, k_cache_output_indices, update_len);
    vCaches_.update(method, v_cache_output_indices, update_len);
    for (auto it : attentionMasks_) {
      it.second.updateCacheMask(update_len);
    }
  }

  void reset() {
    input_pos_ = 0;
    kCaches_.reset();
    vCaches_.reset();
    for (auto it : attentionMasks_) {
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
