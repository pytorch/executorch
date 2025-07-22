/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <executorch/runtime/core/span.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/platform/log.h>

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
      size_t n_heads_per_cache = 1,
      bool transpose = false,
      StaticAttentionUpdateStyle style =
          StaticAttentionUpdateStyle::SLIDING_CACHE)
      : n_caches_(n_caches),
        cache_len_(cache_len),
        max_input_len_(max_input_len),
        n_heads_per_cache_(n_heads_per_cache),
        head_dim_(head_dim),
        transpose_(transpose),
        style_(style),
        input_ptrs_(n_caches_),
        output_ptrs_(n_caches_) {
    if (transpose_) {
      throw std::runtime_error("Not implemented.");
    }

    if (style_ == StaticAttentionUpdateStyle::SLIDING_CACHE &&
        n_heads_per_cache_ > 1) {
      throw std::runtime_error(
          "Sliding cache update strategy doesn't support more than one head per cache tensor.");
    }

    if (style_ == StaticAttentionUpdateStyle::SLIDING_CACHE) {
      // Allocates on extra copy to accomodate caches sliding forward.
      cache_data_size_ = (n_caches_ + 1) * cache_len_ * head_dim_;
    } else {
      cache_data_size_ =
          n_caches_ * n_heads_per_cache_ * cache_len_ * head_dim_;
    }
    update_data_size_ =
        n_caches_ * n_heads_per_cache_ * max_input_len_ * head_dim_;

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
      if (n_heads_per_cache_ > 1) {
        // More than 1 head per cache, meaning regular MHA is used. Tensor shape
        // is (1, n_heads, seq_len, head_dim).
        ET_CHECK_MSG(
            inSizes.size() == 4, "Cache input tensor expected to have rank 4.");
        ET_CHECK_MSG(
            outSizes.size() == 4,
            "Cache input tensor expected to have rank 4.");
        ET_CHECK_MSG(
            inSizes[1] == n_heads_per_cache_,
            "Number of heads per cache mismatch.");
        ET_CHECK_MSG(
            outSizes[1] == n_heads_per_cache_,
            "Number of heads per cache mismatch.");
      } else {
        // 1 head per cache, meaning MHA is split up into multiple SHAs for QNN.
        // Tensor shape is (1, seq_len, head_dim).
        ET_CHECK_MSG(
            inSizes.size() == 3, "Cache input tensor expected to have rank 3.");
        ET_CHECK_MSG(
            outSizes.size() == 3,
            "Cache input tensor expected to have rank 3.");
      }
      auto ndim = inSizes.size();
      if (transpose_) {
        ET_CHECK_MSG(inSizes[ndim - 2] == head_dim_, "KV head dim mismatch.");
        ET_CHECK_MSG(outSizes[ndim - 2] == head_dim_, "KV head dim mismatch.");
        ET_CHECK_MSG(
            inSizes[ndim - 1] == cache_len_, "Cache length dim mismatch.");
      } else {
        ET_CHECK_MSG(inSizes[ndim - 1] == head_dim_, "KV head dim mismatch.");
        ET_CHECK_MSG(outSizes[ndim - 1] == head_dim_, "KV head dim mismatch.");
        ET_CHECK_MSG(
            inSizes[ndim - 2] == cache_len_, "Cache length dim mismatch.");
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
      size_t update_len,
      size_t update_pos = 0) {
    if (valid_len_ + update_len > cache_len_) {
      throw std::runtime_error("Cache capacity exceeded.");
    }

    if (style_ == StaticAttentionUpdateStyle::SLIDING_CACHE) {
      update_sliding_cache(method, output_indices, update_len, update_pos);
    } else {
      update_smart_mask(method, output_indices, update_len, update_pos);
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

  size_t size() {
    return valid_len_;
  }

 private:
  void init_ptrs() {
    input_ptrs_.resize(n_caches_);
    output_ptrs_.resize(n_caches_);
    for (size_t i = 0; i < n_caches_; i++) {
      input_ptrs_[i] =
          cache_data_ + i * n_heads_per_cache_ * cache_len_ * head_dim_;
      output_ptrs_[i] =
          update_data_ + i * n_heads_per_cache_ * max_input_len_ * head_dim_;
    }
  }

  void update_sliding_cache(
      torch::executor::Method& method,
      const std::vector<size_t>& output_indices,
      size_t update_len,
      size_t update_pos) {
    ET_CHECK(n_caches_ == output_indices.size());
    for (size_t i = 0; i < n_caches_; i++) {
      const auto& updateTensor =
          method.get_output(output_indices[i]).toTensor();
      ET_CHECK(output_ptrs_[i] == updateTensor.const_data_ptr<T>());
      std::copy(
          output_ptrs_[i] + update_pos * head_dim_,
          output_ptrs_[i] + (update_pos + update_len) * head_dim_,
          input_ptrs_[i] + cache_len_ * head_dim_);
      input_ptrs_[i] += update_len * head_dim_;
    }
    valid_len_ += update_len;
  }

  void update_smart_mask(
      torch::executor::Method& method,
      const std::vector<size_t>& output_indices,
      size_t update_len,
      size_t update_pos) {
    for (size_t i = 0; i < n_caches_; i++) {
      const auto& updateTensor =
          method.get_output(output_indices[i]).toTensor();
      ET_CHECK(output_ptrs_[i] == updateTensor.mutable_data_ptr<T>());
      auto update_seq_len = updateTensor.size(updateTensor.dim() - 2);
      for (size_t j = 0; j < n_heads_per_cache_; j++) {
        auto* update_head = output_ptrs_[i] + update_seq_len * head_dim_ * j;
        auto* cache_head = input_ptrs_[i] + cache_len_ * head_dim_ * j;
        std::copy(
            update_head + update_pos * head_dim_,
            update_head + (update_pos + update_len) * head_dim_,
            cache_head + valid_len_ * head_dim_);
      }
    }
    valid_len_ += update_len;
  }

  size_t n_caches_;
  size_t cache_len_;
  size_t max_input_len_;
  size_t n_heads_per_cache_;
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

  T zero_val() {
    return zero_val_;
  }

  T mask_val() {
    return mask_val_;
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

template <typename TokenT>
class SuffixCache {
 public:
  SuffixCache(size_t n, size_t capacity)
      : n_(n), capacity_(capacity), pos_(0), cache_(n_ * capacity_) {}

  void add(executorch::runtime::Span<TokenT> suffix) {
    if (suffix.size() != n_ - 1) {
      throw std::runtime_error("Wrong suffix length.");
    }
    for (size_t i = 0; i < capacity_; i++) {
      auto* p = cache_.data() + (n_ - 1) * i;
      if (std::equal(p, p + (n_ - 1), suffix.begin())) {
        return;
      }
    }
    auto* dst = cache_.data() + (n_ - 1) * pos_;
    std::copy(suffix.begin(), suffix.end(), dst);
    pos_ = (pos_ + 1) % capacity_;
  }

  auto begin() {
    return cache_.begin();
  }
  auto end() {
    return cache_.end();
  }
  auto begin() const {
    return cache_.begin();
  }
  auto end() const {
    return cache_.end();
  }

  static void seed_suffix_caches(
      std::unordered_map<TokenT, example::SuffixCache<TokenT>>& suffix_caches,
      executorch::runtime::Span<TokenT> toks,
      size_t ngram_size,
      size_t cache_size) {
    for (size_t i = 0; i + ngram_size < toks.size(); i++) {
      auto& cache = suffix_caches.try_emplace(toks[i], ngram_size, cache_size)
                        .first->second;
      cache.add(executorch::runtime::Span(&toks[i + 1], ngram_size - 1));
    }
  }

 private:
  size_t n_;
  size_t capacity_;
  size_t pos_;
  std::vector<TokenT> cache_;
};

template <
    typename CacheT,
    typename MaskT,
    typename RopeT,
    typename CacheAllocatorT = std::allocator<CacheT>,
    typename MaskAllocatorT = std::allocator<MaskT>>
class StaticAttentionIOManager {
 public:
  struct StaticAttentionIOConfig {
    size_t n_caches{};
    size_t cache_len{};
    size_t head_dim{};
    size_t max_input_len{};
    size_t n_heads_per_cache{};
    size_t attn_mask_input_index{};
    size_t rope_freqs_cos_input_index{};
    size_t rope_freqs_sin_input_index{};
    std::vector<size_t> k_cache_input_indices;
    std::vector<size_t> k_cache_output_indices;
    std::vector<size_t> v_cache_input_indices;
    std::vector<size_t> v_cache_output_indices;
    RopeT* rope_freqs_cos;
    RopeT* rope_freqs_sin;
    StaticAttentionUpdateStyle style =
        StaticAttentionUpdateStyle::SLIDING_CACHE;
  };

  StaticAttentionIOManager(StaticAttentionIOConfig config)
      : config_(std::move(config)),
        kCaches_(
            config_.n_caches,
            config_.cache_len,
            config_.head_dim,
            config_.max_input_len,
            config_.n_heads_per_cache,
            false,
            config_.style),
        vCaches_(
            config_.n_caches,
            config_.cache_len,
            config_.head_dim,
            config_.max_input_len,
            config_.n_heads_per_cache,
            false,
            config_.style) {
    ET_LOG(
        Info,
        "Created StaticAttentionIOManager with"
        " max input length = %zu cache length = %zu",
        config_.max_input_len,
        config_.cache_len);
  }

  /**
   * Create a new StaticAttentionMask that will be managed by this object.
   */
  StaticAttentionMask<MaskT, MaskAllocatorT>&
  add_mask(size_t input_len, MaskT zero_val, MaskT mask_val) {
    auto it = attentionMasks_.emplace(
        std::piecewise_construct,
        std::forward_as_tuple(input_len),
        std::forward_as_tuple(
            config_.cache_len,
            input_len,
            config_.head_dim,
            zero_val,
            mask_val,
            config_.style));
    return it.first->second;
  }

  /**
   * Retrieve a mask suitable for given input length.
   */
  StaticAttentionMask<MaskT, MaskAllocatorT>& get_mask(size_t input_len) {
    return attentionMasks_.at(input_len);
  }

  /**
   * Set I/O pointers for KV cache and RoPE freqencies.
   */
  void prepare(
      torch::executor::Method& method,
      std::optional<const executorch::runtime::Span<size_t>> pos_offsets =
          std::nullopt) {
    kCaches_.prepare(
        method, config_.k_cache_input_indices, config_.k_cache_output_indices);
    vCaches_.prepare(
        method, config_.v_cache_input_indices, config_.v_cache_output_indices);

    size_t rope_dim = config_.head_dim / 2;
    if (pos_offsets) {
      rope_freqs_cos_override_.clear();
      rope_freqs_sin_override_.clear();
      for (auto offset : *pos_offsets) {
        auto pos = input_pos_ + offset;
        std::copy(
            config_.rope_freqs_cos + pos * rope_dim,
            config_.rope_freqs_cos + (pos + 1) * rope_dim,
            std::back_inserter(rope_freqs_cos_override_));
        std::copy(
            config_.rope_freqs_sin + pos * rope_dim,
            config_.rope_freqs_sin + (pos + 1) * rope_dim,
            std::back_inserter(rope_freqs_sin_override_));
      }
      set_input(
          method,
          config_.rope_freqs_cos_input_index,
          rope_freqs_cos_override_.data());
      set_input(
          method,
          config_.rope_freqs_sin_input_index,
          rope_freqs_sin_override_.data());
    } else {
      set_input(
          method,
          config_.rope_freqs_cos_input_index,
          config_.rope_freqs_cos + input_pos_ * rope_dim);
      set_input(
          method,
          config_.rope_freqs_sin_input_index,
          config_.rope_freqs_sin + input_pos_ * rope_dim);
    }
  }

  /**
   * Update all caches and masks under management to reflect that model produced
   * update_len new elements.
   */
  void update(
      torch::executor::Method& method,
      const std::vector<size_t>& k_cache_output_indices,
      const std::vector<size_t>& v_cache_output_indices,
      size_t update_len,
      size_t cache_update_pos = 0) {
    input_pos_ += update_len;
    kCaches_.update(
        method, k_cache_output_indices, update_len, cache_update_pos);
    vCaches_.update(
        method, v_cache_output_indices, update_len, cache_update_pos);
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

  template <typename TokenT>
  std::vector<TokenT> decode(
      TokenT prev_tok,
      executorch::runtime::Span<TokenT> input_buffer,
      executorch::runtime::Method& method,
      std::function<TokenT(executorch::runtime::Method&)>& sample,
      std::function<bool(TokenT)>& should_stop) {
    set_input(method, 0, input_buffer.data());
    auto& mask = get_mask(input_buffer.size());
    set_input(method, config_.attn_mask_input_index, mask.get());

    std::vector<TokenT> generated_tokens;
    while (kCaches_.size() + 1 <= config_.cache_len) {
      input_buffer[0] = prev_tok;
      prepare(method);
      ET_CHECK(method.execute() == executorch::runtime::Error::Ok);
      update(
          method,
          config_.k_cache_output_indices,
          config_.v_cache_output_indices,
          1);
      prev_tok = sample(method);
      generated_tokens.emplace_back(prev_tok);
      if (should_stop(prev_tok)) {
        break;
      }
    }
    return generated_tokens;
  }

  template <typename TokenT>
  std::vector<TokenT> lookahead_decode(
      TokenT prev_tok,
      executorch::runtime::Span<TokenT> input_buffer,
      executorch::runtime::Method& method,
      std::function<std::vector<TokenT>(executorch::runtime::Method&)>& sample,
      std::function<bool(TokenT)>& should_stop,
      size_t ngram_size,
      size_t window_size,
      size_t n_verifications,
      std::unordered_map<TokenT, SuffixCache<TokenT>> suffix_caches) {
    set_input(method, 0, input_buffer.data());
    size_t input_len = input_buffer.size();

    // Set up attention mask for current input length.
    auto& mask = get_mask(input_buffer.size());
    set_lookahead_decoding_mask(
        mask, input_len, ngram_size, window_size, n_verifications);
    set_input(method, config_.attn_mask_input_index, mask.get());

    // Position offsets relative to current position, for indexing RoPE
    // frequence tensors.
    auto pos_offsets = get_lookahead_pos_offsets(
        input_len, ngram_size, window_size, n_verifications);

    ET_LOG(
        Info,
        "Starting lookahead decoding with"
        " ngram_size = %zu"
        " window_size = %zu"
        " n_verifications = %zu",
        ngram_size,
        window_size,
        n_verifications);

    // Decoding loop.
    std::vector<TokenT> generated_tokens;
    size_t verification_offset =
        std::max(window_size * (ngram_size - 1), static_cast<size_t>(1));
    size_t n_inference = 0;
    std::fill(input_buffer.begin(), input_buffer.end(), prev_tok);
    while (kCaches_.size() + 1 <= config_.cache_len) {
      input_buffer[0] = prev_tok;
      // Initialize verification branches.
      if (auto it = suffix_caches.find(prev_tok); it != suffix_caches.end()) {
        auto& cache = it->second;
        std::copy(
            cache.begin(),
            cache.end(),
            input_buffer.data() + verification_offset);
      }

      // Setup input pointers and RoPE frequencies.
      prepare(
          method,
          executorch::runtime::Span(pos_offsets.data(), pos_offsets.size()));
      ET_CHECK(method.execute() == executorch::runtime::Error::Ok);
      n_inference++;
      // Update KV caches and mask for the 1st input position. If verification
      // branches produced additional matches they'll be updated seprately
      // because they are not contiguous in the KV cache.
      update(
          method,
          config_.k_cache_output_indices,
          config_.v_cache_output_indices,
          1);

      auto output_toks = sample(method);

      // Collect new n-grams from lookahead branches.
      std::vector<TokenT> new_suffix;
      for (size_t i = 0; i < window_size; i++) {
        new_suffix.clear();
        for (size_t j = 1; j < ngram_size - 1; j++) {
          new_suffix.emplace_back(input_buffer[i + window_size * j]);
        }
        new_suffix.emplace_back(
            output_toks[i + window_size * (ngram_size - 2)]);

        auto& cache =
            suffix_caches
                .try_emplace(input_buffer[i], ngram_size, n_verifications)
                .first->second;
        cache.add(executorch::runtime::Span(new_suffix.data(), ngram_size - 1));
      }

      // Update lookahead branches.
      for (size_t i = 0; i < ngram_size - 2; i++) {
        for (size_t j = 0; j < window_size; j++) {
          input_buffer[window_size * i + j] =
              input_buffer[window_size * (i + 1) + j];
        }
      }
      for (size_t j = 0; j < window_size; j++) {
        input_buffer[window_size * (ngram_size - 2) + j] =
            output_toks[window_size * (ngram_size - 2) + j];
      }

      // Check verification results.
      std::vector<TokenT> longest_match;
      size_t matched_branch = 0;
      for (size_t i = 0; i < n_verifications; i++) {
        std::vector<TokenT> match;
        match.emplace_back(output_toks[0]);
        size_t branch_offset = verification_offset + (ngram_size - 1) * i;
        for (size_t j = 0; j < ngram_size - 1 &&
             input_buffer[branch_offset + j] == match.back();
             j++) {
          match.emplace_back(output_toks[branch_offset + j]);
          if (should_stop(match.back())) {
            break;
          }
        }
        if (match.size() > longest_match.size()) {
          longest_match = std::move(match);
          matched_branch = i;
        }
      }

      bool generated_stop_tok = false;
      for (auto tok : longest_match) {
        generated_tokens.emplace_back(tok);
        if (should_stop(tok)) {
          generated_stop_tok = true;
          break;
        }
      }

      // Update KV caches and mask for additional matches.
      if (longest_match.size() > 1) {
        size_t branch_offset =
            verification_offset + (ngram_size - 1) * matched_branch;
        update(
            method,
            config_.k_cache_output_indices,
            config_.v_cache_output_indices,
            std::min(
                longest_match.size() - 1, config_.cache_len - kCaches_.size()),
            branch_offset);
      }

      if (generated_stop_tok) {
        break;
      }
      prev_tok = generated_tokens.back();
    }

    ET_LOG(
        Info,
        "Generated %zu tokens with %zu inferences(s).",
        generated_tokens.size(),
        n_inference);

    return generated_tokens;
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

  void set_lookahead_decoding_mask(
      StaticAttentionMask<MaskT, MaskAllocatorT>& mask,
      size_t input_len,
      size_t ngram_size,
      size_t window_size,
      size_t n_verifications) {
    class SubMask {
     public:
      SubMask(MaskT* data, size_t stride) : data_(data), stride_(stride) {}

      MaskT& at(size_t i, size_t j = 0) {
        return data_[i * stride_ + j];
      }

     private:
      MaskT* data_;
      size_t stride_;
    };

    size_t stride = config_.cache_len + input_len;
    auto input_submask = SubMask(mask.get() + config_.cache_len, stride);
    input_submask.at(0, 0) = mask.zero_val();

    // Fill entire input mask first.
    for (size_t i = 0; i < input_len; i++) {
      auto* p = &input_submask.at(i);
      std::fill(p, p + input_len, mask.mask_val());
    }

    auto set_causal_mask = [&](SubMask m, size_t size) {
      for (size_t i = 0; i < size; i++) {
        auto* p = &m.at(i);
        std::fill(p, p + i + 1, mask.zero_val());
      }
    };

    auto set_diagonal_mask = [&](SubMask m, size_t size) {
      for (size_t i = 0; i < size; i++) {
        m.at(i, i) = mask.zero_val();
      }
    };

    // Set lookahead submasks.
    for (size_t i = 0; i < ngram_size - 1; i++) {
      set_causal_mask(
          SubMask(&input_submask.at(window_size * i), stride), window_size);
      for (size_t j = 1; j < i + 1; j++) {
        set_diagonal_mask(
            SubMask(
                &input_submask.at(window_size * i, window_size * j), stride),
            window_size);
      }
    }

    // Set verification submasks
    size_t verification_offset =
        std::max(window_size * (ngram_size - 1), static_cast<size_t>(1));
    for (size_t i = 0; i < n_verifications; i++) {
      size_t branch_offset = verification_offset + i * (ngram_size - 1);
      set_causal_mask(
          SubMask(&input_submask.at(branch_offset, branch_offset), stride),
          ngram_size - 1);
    }
    for (size_t i = verification_offset; i < input_len; i++) {
      input_submask.at(i, 0) = mask.zero_val();
    }
  }

  std::vector<size_t> get_lookahead_pos_offsets(
      size_t input_len,
      size_t ngram_size,
      size_t window_size,
      size_t n_verifications) {
    std::vector<size_t> offsets(input_len);
    size_t idx = 0;

    // Lookahead branches: [i + 0, i + 1, ..., i + window_size - 1]
    if (window_size > 0) {
      for (size_t i = 0; i < ngram_size - 1; i++) {
        for (size_t j = 0; j < window_size; j++) {
          offsets[idx++] = i + j;
        }
      }
    } else {
      offsets[idx++] = 0;
    }

    // Verification branches: [1, 2, ..., ngram_size - 1]
    for (size_t i = 0; i < n_verifications; i++) {
      for (size_t j = 1; j < ngram_size; j++) {
        offsets[idx++] = j;
      }
    }

    return offsets;
  }

  StaticAttentionIOConfig config_;
  size_t input_pos_;
  StaticKVCache<CacheT, CacheAllocatorT> kCaches_;
  StaticKVCache<CacheT, CacheAllocatorT> vCaches_;
  std::unordered_map<size_t, StaticAttentionMask<MaskT, MaskAllocatorT>>
      attentionMasks_;
  std::vector<RopeT> rope_freqs_cos_override_;
  std::vector<RopeT> rope_freqs_sin_override_;
};

} // namespace example
