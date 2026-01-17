/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_prompt_processor.h>
#include <numeric>

using executorch::aten::TensorImpl;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
using executorch::runtime::TensorInfo;

namespace example {

template <typename T>
MultimodalPromptProcessor<T>::MultimodalPromptProcessor(
    DecoderRunner* decoder_runner,
    KVManager<T>* kv_manager,
    const std::string& method_name,
    Metadata metadata)
    : PromptProcessor<T>(
          decoder_runner,
          kv_manager,
          method_name,
          {metadata.context_len,
           metadata.num_heads,
           metadata.num_layers,
           metadata.ar_len,
           metadata.vocab_size,
           metadata.use_int64_token,
           metadata.sliding_window,
           metadata.cache_mode}),
      metadata_(metadata) {
  // Set input_toks_.size to 0 since we use embeddings instead
  input_toks_.size = 0;
  input_embedding_.size =
      metadata_.ar_len * metadata_.embedding_dim * sizeof(float);
};

template <typename T>
void MultimodalPromptProcessor<T>::init_io(
    IMemAlloc* buffer_manager,
    Result<MethodMeta> method_meta) {
  size_t idx = 0;
  input_tensors_.reserve(method_meta->num_inputs());
  output_tensors_.reserve(method_meta->num_outputs());

  // [I]: input embedding
  Result<TensorInfo> input_embedding = method_meta->input_tensor_meta(idx++);
  input_embedding_.data =
      reinterpret_cast<float*>(buffer_manager->allocate(input_embedding_.size));
  input_embedding_.tensor = std::make_unique<TensorImpl>(
      input_embedding->scalar_type(),
      input_embedding->sizes().size(),
      const_cast<TensorImpl::SizesType*>(input_embedding->sizes().data()),
      input_embedding_.data,
      const_cast<TensorImpl::DimOrderType*>(
          input_embedding->dim_order().data()));
  input_tensors_.emplace_back(input_embedding_.tensor.get());
  buffer_manager->add_memory_info(
      input_embedding_.data, input_embedding_.size, input_embedding.get());

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

// prepare embedding
template <typename T>
void MultimodalPromptProcessor<T>::prepare_io(
    const TensorStruct<float>& prompt_embedding,
    int32_t num_prompt_tokens,
    int64_t prompt_pos,
    int64_t start_pos) {
  for (int i = 0; i < metadata_.ar_len; i++) {
    if (!is_bert()) {
      // Prepare pos data
      input_pos_.data[i] = start_pos + i;
    }

    // Prepare input token data
    if (prompt_pos + i < num_prompt_tokens) {
      std::memcpy(
          input_embedding_.data + i * metadata_.embedding_dim,
          prompt_embedding.data + (prompt_pos + i) * metadata_.embedding_dim,
          metadata_.embedding_dim * sizeof(float));
    }
  }
}

template <typename T>
Result<uint64_t> MultimodalPromptProcessor<T>::prefill(
    const TensorStruct<float>& prompt_embedding,
    int64_t start_pos,
    bool dump_logits) {
  int32_t num_prompt_tokens = prompt_embedding.tensor->size(1);
  if (!is_bert()) {
    ET_CHECK_MSG(
        (start_pos + num_prompt_tokens) <=
            (metadata_.context_len - metadata_.ar_len),
        "The sequence length exceeds the maximum limit that the prompt processor can handle.");
  } else {
    ET_CHECK_MSG(
        start_pos == 0, "Bert model doesn't support multi-turn conversation.");
  }

  // store the token
  int64_t cur_token;
  int64_t prompt_pos = 0;
  int64_t pos = start_pos;
  int32_t n_update = metadata_.ar_len;
  int num_iters = 1 + ((num_prompt_tokens - 1) / metadata_.ar_len);
  ET_LOG(
      Info,
      "Prompt Processor: total %d prompt tokens (AR-%d * %d iters)",
      num_prompt_tokens,
      metadata_.ar_len,
      num_iters);

  // Rearrange KV cache first
  kv_manager_->rearrange_cache(metadata_.ar_len);
  std::vector<int32_t> attention_map(metadata_.ar_len);
  std::iota(attention_map.begin(), attention_map.end(), -1);
  // Initialize attention mask with current position
  kv_manager_->init_attention_mask(
      attention_mask_.data, attention_map, metadata_.ar_len, pos);
  // Initialize window attention mask with current position
  if (metadata_.cache_mode == CacheMode::HybridCache) {
    kv_manager_->init_attention_mask(
        window_attention_mask_.data,
        attention_map,
        metadata_.ar_len,
        pos,
        metadata_.sliding_window);
  }

  // Initialize the output of the module
  ET_CHECK_MSG(
      decoder_runner_->set_outputs(method_name_, output_tensors_) ==
          executorch::runtime::Error::Ok,
      "Failed to set output tensor for module %s",
      method_name_.c_str());
  for (int i = 0; i < num_iters; ++i) {
    // Fill in the embedding and position data
    prepare_io(prompt_embedding, num_prompt_tokens, prompt_pos, pos);

    // Run inference
    for (int layer = 0; layer < metadata_.num_layers; ++layer) {
      std::vector<KVCache<T>> k_cache_ptrs = kv_manager_->get_k_cache_();
      T* k_cache_data = k_cache_ptrs[layer].buffer;
    }
    for (int layer = 0; layer < metadata_.num_layers; ++layer) {
      std::vector<KVCache<T>> v_cache_ptrs = kv_manager_->get_v_cache_();
      T* v_cache_data = v_cache_ptrs[layer].buffer;
    }

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
    kv_manager_->update_cache(metadata_.ar_len, pos, n_update, {});

    // Update attention mask with current position
    kv_manager_->update_attention_mask(
        attention_mask_.data, metadata_.ar_len, pos, n_update);
    if (metadata_.cache_mode == CacheMode::HybridCache) {
      kv_manager_->update_attention_mask(
          window_attention_mask_.data,
          metadata_.ar_len,
          pos,
          n_update,
          metadata_.sliding_window);
    }
    prompt_pos += metadata_.ar_len;
    pos += metadata_.ar_len;
  }

  cur_token = decoder_runner_->logits_to_token(
      output_tensors_[0],
      (num_prompt_tokens + metadata_.ar_len - 1) % metadata_.ar_len);
  return cur_token;
}

// Explicit instantiations
template class MultimodalPromptProcessor<uint16_t>;
template class MultimodalPromptProcessor<uint8_t>;

} // namespace example
