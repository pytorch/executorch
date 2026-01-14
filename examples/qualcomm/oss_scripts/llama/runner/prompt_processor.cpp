/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/prompt_processor.h>
#include <numeric>
using executorch::aten::TensorImpl;
using executorch::runtime::EValue;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::TensorInfo;
namespace example {

template <typename T>
PromptProcessor<T>::PromptProcessor(
    DecoderRunner* decoder_runner,
    KVManager<T>* kv_manager,
    const std::string& method_name,
    Metadata metadata)
    : decoder_runner_(decoder_runner),
      kv_manager_(kv_manager),
      method_name_(method_name),
      metadata_(metadata) {
  k_cache_in_.resize(metadata_.num_layers);
  v_cache_in_.resize(metadata_.num_layers);
  k_cache_out_.resize(metadata_.num_layers);
  v_cache_out_.resize(metadata_.num_layers);
  // Calculate I/O size
  input_toks_.size = metadata_.ar_len * sizeof(int64_t);
  if (is_bert())
    input_pos_.size = 0;
  else
    input_pos_.size = metadata_.ar_len * sizeof(int32_t);

  switch (metadata_.cache_mode) {
    case CacheMode::StaticCahce:
      attention_mask_.size =
          metadata_.ar_len * metadata_.context_len * sizeof(uint16_t);
      window_attention_mask_.size = 0;
      break;
    case CacheMode::HybridCache:
      attention_mask_.size =
          metadata_.ar_len * metadata_.context_len * sizeof(uint16_t);
      window_attention_mask_.size =
          metadata_.ar_len * metadata_.context_len * sizeof(uint16_t);
      break;
    default:
      ET_CHECK_MSG(false, "Unsupported llama cache mode");
      break;
  }

  logits_.size = metadata_.ar_len * metadata_.vocab_size * sizeof(uint16_t);
};
template <typename T>
void PromptProcessor<T>::init_io(
    IMemAlloc* buffer_manager,
    Result<MethodMeta> method_meta) {
  size_t idx = 0;
  input_tensors_.reserve(method_meta->num_inputs());
  output_tensors_.reserve(method_meta->num_outputs());
  // [I]: input_tokens
  Result<TensorInfo> input_toks = method_meta->input_tensor_meta(idx++);
  input_toks_.data =
      reinterpret_cast<int64_t*>(buffer_manager->allocate(input_toks_.size));
  input_toks_.tensor = std::make_unique<TensorImpl>(
      input_toks->scalar_type(),
      input_toks->sizes().size(),
      const_cast<TensorImpl::SizesType*>(input_toks->sizes().data()),
      input_toks_.data,
      const_cast<TensorImpl::DimOrderType*>(input_toks->dim_order().data()));
  input_tensors_.emplace_back(input_toks_.tensor.get());
  buffer_manager->add_memory_info(
      input_toks_.data, input_toks_.size, input_toks.get());

  // [I]: attention_mask
  Result<TensorInfo> attention_mask = method_meta->input_tensor_meta(idx++);
  attention_mask_.data = reinterpret_cast<uint16_t*>(
      buffer_manager->allocate(attention_mask_.size));
  attention_mask_.tensor = std::make_unique<TensorImpl>(
      attention_mask->scalar_type(),
      attention_mask->sizes().size(),
      const_cast<TensorImpl::SizesType*>(attention_mask->sizes().data()),
      attention_mask_.data,
      const_cast<TensorImpl::DimOrderType*>(
          attention_mask->dim_order().data()));
  input_tensors_.emplace_back(attention_mask_.tensor.get());
  buffer_manager->add_memory_info(
      attention_mask_.data, attention_mask_.size, attention_mask.get());

  // [I]: sliding window attention_mask
  if (metadata_.cache_mode == CacheMode::HybridCache) {
    Result<TensorInfo> window_attention_mask =
        method_meta->input_tensor_meta(idx++);
    window_attention_mask_.data = reinterpret_cast<uint16_t*>(
        buffer_manager->allocate(window_attention_mask_.size));
    window_attention_mask_.tensor = std::make_unique<TensorImpl>(
        window_attention_mask->scalar_type(),
        window_attention_mask->sizes().size(),
        const_cast<TensorImpl::SizesType*>(
            window_attention_mask->sizes().data()),
        window_attention_mask_.data,
        const_cast<TensorImpl::DimOrderType*>(
            window_attention_mask->dim_order().data()));
    input_tensors_.emplace_back(window_attention_mask_.tensor.get());
    buffer_manager->add_memory_info(
        window_attention_mask_.data,
        window_attention_mask_.size,
        window_attention_mask.get());
  }

  if (!is_bert()) {
    // [I]: input_pos
    Result<TensorInfo> input_pos = method_meta->input_tensor_meta(idx++);
    input_pos_.data =
        reinterpret_cast<int32_t*>(buffer_manager->allocate(input_pos_.size));
    input_pos_.tensor = std::make_unique<TensorImpl>(
        input_pos->scalar_type(),
        input_pos->sizes().size(),
        const_cast<TensorImpl::SizesType*>(input_pos->sizes().data()),
        input_pos_.data,
        const_cast<TensorImpl::DimOrderType*>(input_pos->dim_order().data()));
    input_tensors_.emplace_back(input_pos_.tensor.get());
    buffer_manager->add_memory_info(
        input_pos_.data, input_pos_.size, input_pos.get());

    // [I] kv_cache
    // Prepare the vector of EValue for kv cache to evict token
    cache_inputs_.reserve(2 * metadata_.num_layers);
    size_t index = idx; // bypass input_tokens, atten_mask, input_pos
    for (int cache_group = 0; cache_group < 2; ++cache_group) {
      std::vector<std::unique_ptr<TensorImpl>>& cache =
          (cache_group == 0 ? k_cache_in_ : v_cache_in_);
      std::vector<KVCache<T>> cache_ptrs = (cache_group == 0)
          ? kv_manager_->get_k_cache_()
          : kv_manager_->get_v_cache_();
      for (int layer = 0; layer < metadata_.num_layers; ++layer, ++index) {
        Result<TensorInfo> kv_cache = method_meta->input_tensor_meta(index);

        T* cache_ptr = cache_ptrs[layer].buffer;

        cache[layer] = std::make_unique<TensorImpl>(
            kv_cache->scalar_type(),
            kv_cache->sizes().size(),
            const_cast<TensorImpl::SizesType*>(kv_cache->sizes().data()),
            cache_ptr,
            const_cast<TensorImpl::DimOrderType*>(
                kv_cache->dim_order().data()));
        input_tensors_.emplace_back(cache[layer].get());
        cache_inputs_.emplace_back(input_tensors_.back());
        buffer_manager->add_memory_info(
            cache_ptr, cache[layer]->nbytes(), kv_cache.get());
      }
    }
  }

  // [O]: logits
  Result<TensorInfo> logits = method_meta->output_tensor_meta(0);
  logits_.data =
      reinterpret_cast<uint16_t*>(buffer_manager->allocate(logits_.size));
  logits_.tensor = std::make_unique<TensorImpl>(
      logits->scalar_type(),
      logits->sizes().size(),
      const_cast<TensorImpl::SizesType*>(logits->sizes().data()),
      logits_.data,
      const_cast<TensorImpl::DimOrderType*>(logits->dim_order().data()));
  output_tensors_.emplace_back(logits_.tensor.get());
  buffer_manager->add_memory_info(logits_.data, logits_.size, logits.get());

  // [O] kv_cache
  size_t index = 1;
  for (int cache_group = 0; cache_group < 2; ++cache_group) {
    std::vector<std::unique_ptr<TensorImpl>>& cache =
        (cache_group == 0 ? k_cache_out_ : v_cache_out_);
    std::vector<KVCache<T>> cache_ptrs = (cache_group == 0)
        ? kv_manager_->get_k_cache_()
        : kv_manager_->get_v_cache_();
    for (int layer = 0; layer < metadata_.num_layers; ++layer, ++index) {
      Result<TensorInfo> kv_cache = method_meta->output_tensor_meta(index);
      T* cache_ptr = cache_ptrs[layer].output_buffer;
      cache[layer] = std::make_unique<TensorImpl>(
          kv_cache->scalar_type(),
          kv_cache->sizes().size(),
          const_cast<TensorImpl::SizesType*>(kv_cache->sizes().data()),
          cache_ptr,
          const_cast<TensorImpl::DimOrderType*>(kv_cache->dim_order().data()));
      output_tensors_.emplace_back(cache[layer].get());
      buffer_manager->add_memory_info(
          cache_ptr, cache[layer]->nbytes(), kv_cache.get());
    }
  }
  // Prepare the vector of EValue to run inference
  inputs_.reserve(input_tensors_.size());
  for (auto& input_tensor : input_tensors_) {
    inputs_.emplace_back(std::move(input_tensor));
  }
}

template <typename T>
const std::vector<uint16_t>& PromptProcessor<T>::get_all_logits() {
  return prompt_all_logits_;
}

template <typename T>
void PromptProcessor<T>::prepare_io(
    const std::vector<uint64_t>& prompt_tokens,
    int64_t prompt_pos,
    int64_t start_pos) {
  for (int i = 0; i < metadata_.ar_len; i++) {
    if (!is_bert()) {
      // Prepare pos data
      input_pos_.data[i] = start_pos + i;
    }

    // Prepare input token data
    if (prompt_pos + i < prompt_tokens.size()) {
      // Support CPU 4-bit embedding, which requires int64 input.
      // However, for QNN embedding, only int32 input is needed.
      // Therefore, we need to cast to the correct type to write the data.
      if (metadata_.use_int64_token) {
        input_toks_.data[i] = prompt_tokens[prompt_pos + i];
      } else {
        int32_t* input_toks_ptr = reinterpret_cast<int32_t*>(input_toks_.data);
        input_toks_ptr[i] = static_cast<int32_t>(prompt_tokens[prompt_pos + i]);
      }
    }
  }
}

template <typename T>
Result<uint64_t> PromptProcessor<T>::prefill(
    std::vector<uint64_t> prompt_tokens,
    int64_t start_pos,
    bool dump_logits,
    AttentionSinkRopeRunner* attention_sink_rope_runner) {
  ET_CHECK_MSG(!prompt_tokens.empty(), "Prompt cannot be null");

  int64_t shifted_pos = start_pos;
  bool enable_attention_sink = attention_sink_rope_runner != nullptr;

  // Calculate number of blocks
  int32_t num_prompt_tokens = prompt_tokens.size();
  if (is_bert()) {
    ET_CHECK_MSG(
        start_pos == 0, "Bert model doesn't support multi-turn conversation.");
  } else if (!enable_attention_sink) {
    ET_CHECK_MSG(
        (start_pos + num_prompt_tokens) <=
            (metadata_.context_len - metadata_.ar_len),
        "The sequence length exceeds the maximum limit that the prompt processor can handle.");
  }

  // store the token
  int64_t cur_token;
  int64_t prompt_pos = 0;
  int32_t n_update = metadata_.ar_len;
  int num_iters = 1 + ((num_prompt_tokens - 1) / metadata_.ar_len);
  ET_LOG(
      Info,
      "Prompt Processor: total %d prompt tokens (AR-%d * %d iters)",
      num_prompt_tokens,
      metadata_.ar_len,
      num_iters);

  // Initialize attention sink rope runner if given and update position
  // accordingly
  if (enable_attention_sink) {
    ET_CHECK_MSG(
        attention_sink_rope_runner->set_outputs(method_name_, cache_inputs_) ==
            executorch::runtime::Error::Ok,
        "Failed to set output tensor for module %s",
        method_name_.c_str());
    shifted_pos =
        shifted_pos - attention_sink_rope_runner->get_position_shift();
  }

  // Rearrange KV cache first
  kv_manager_->rearrange_cache(metadata_.ar_len);
  std::vector<int32_t> attention_map(metadata_.ar_len);
  std::iota(attention_map.begin(), attention_map.end(), -1);
  // Initialize attention mask with current position
  kv_manager_->init_attention_mask(
      attention_mask_.data, attention_map, metadata_.ar_len, shifted_pos);
  // Initialize window attention mask with current position
  if (metadata_.cache_mode == CacheMode::HybridCache) {
    kv_manager_->init_attention_mask(
        window_attention_mask_.data,
        attention_map,
        metadata_.ar_len,
        shifted_pos,
        metadata_.sliding_window);
  }

  // Initialize the output of the module
  ET_CHECK_MSG(
      decoder_runner_->set_outputs(method_name_, output_tensors_) ==
          executorch::runtime::Error::Ok,
      "Failed to set output tensor for module %s",
      method_name_.c_str());

  for (int i = 0; i < num_iters; ++i) {
    // The current position plus the future generated cache exceeds the cache
    // size, which means we need to remove eviction_batch_size key-value cache
    // entries to make room for new tokens.
    if (enable_attention_sink &&
        shifted_pos + metadata_.ar_len >
            metadata_.context_len - metadata_.ar_len) {
      attention_sink_rope_runner->evict_token(method_name_, cache_inputs_);
      shifted_pos =
          shifted_pos - attention_sink_rope_runner->get_eviction_batch_size();
      // Initialize attention mask with current position
      kv_manager_->init_attention_mask(
          attention_mask_.data, attention_map, metadata_.ar_len, shifted_pos);
      // Initialize window attention mask with current position
      if (metadata_.cache_mode == CacheMode::HybridCache) {
        kv_manager_->init_attention_mask(
            window_attention_mask_.data,
            attention_map,
            metadata_.ar_len,
            shifted_pos,
            metadata_.sliding_window);
      }
    }

    // Fill in the token and position data
    prepare_io(prompt_tokens, prompt_pos, shifted_pos);

    // Run inference
    decoder_runner_->step(method_name_, inputs_);
    if (dump_logits) {
      prompt_all_logits_.insert(
          prompt_all_logits_.end(),
          logits_.data,
          logits_.data + metadata_.ar_len * metadata_.vocab_size);
    }
    // In the last run, offset to the meaningful logits.
    if (i == num_iters - 1) {
      n_update = 1 + ((num_prompt_tokens - 1) % metadata_.ar_len);
    }
    // Update KV Cache with the output results
    kv_manager_->update_cache(metadata_.ar_len, shifted_pos, n_update, {});

    // Update attention mask with current position
    kv_manager_->update_attention_mask(
        attention_mask_.data, metadata_.ar_len, shifted_pos, n_update);
    if (metadata_.cache_mode == CacheMode::HybridCache) {
      kv_manager_->update_attention_mask(
          window_attention_mask_.data,
          metadata_.ar_len,
          shifted_pos,
          n_update,
          metadata_.sliding_window);
    }
    prompt_pos += metadata_.ar_len;
    shifted_pos += metadata_.ar_len;
  }

  cur_token = decoder_runner_->logits_to_token(
      output_tensors_[0],
      (num_prompt_tokens + metadata_.ar_len - 1) % metadata_.ar_len);
  return cur_token;
}

// Explicit instantiations
template class PromptProcessor<uint16_t>;
template class PromptProcessor<uint8_t>;

} // namespace example
