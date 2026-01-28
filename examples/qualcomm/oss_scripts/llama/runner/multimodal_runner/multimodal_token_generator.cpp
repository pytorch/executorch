/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_token_generator.h>
#include <numeric>
using executorch::aten::TensorImpl;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
using executorch::runtime::TensorInfo;

namespace example {
// Constructor with embedding runner support
template <typename T>
MultimodalTokenGenerator<T>::MultimodalTokenGenerator(
    tokenizers::Tokenizer* tokenizer,
    EmbeddingProcessor* embedding_runner,
    DecoderRunner* decoder_runner,
    KVManager<T>* kv_manager,
    const std::string& method_name,
    std::unique_ptr<std::unordered_set<uint64_t>>&& eos_ids,
    Metadata metadata,
    executorch::llm::Stats* stats)
    : TokenGenerator<T>(
          tokenizer,
          decoder_runner,
          kv_manager,
          method_name,
          std::move(eos_ids),
          {metadata.context_len,
           metadata.num_heads,
           metadata.num_layers,
           metadata.ar_len,
           metadata.vocab_size,
           metadata.use_int64_token,
           metadata.sliding_window,
           metadata.cache_mode},
          stats),
      embedding_runner_(embedding_runner),
      metadata_(metadata) {
  // Set input_toks_.size to 0 since we use embeddings instead
  input_toks_.size = 0;
  input_embedding_.size =
      metadata_.ar_len * metadata_.embedding_dim * sizeof(float);
}

template <typename T>
void MultimodalTokenGenerator<T>::init_io(
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
          const_cast<TensorImpl::DimOrderType*>(kv_cache->dim_order().data()));
      input_tensors_.emplace_back(cache[layer].get());
      buffer_manager->add_memory_info(
          cache_ptr, cache[layer]->nbytes(), kv_cache.get());
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
  index = 1;
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

// This function only considers the case where token_generator_ar_len equals 1.
template <typename T>
void MultimodalTokenGenerator<T>::prepare_io(
    uint64_t cur_token,
    int64_t start_pos) {
  // Generate embedding for current token using embedding runner
  embedding_runner_->prefill({cur_token});
  const TensorStruct<float>& text_embeddings =
      embedding_runner_->get_prompt_embeddings();
  int64_t embedding_dim = text_embeddings.tensor->size(2);
  // Copy embedding to input buffer
  std::memcpy(
      input_embedding_.data,
      text_embeddings.data,
      metadata_.ar_len * embedding_dim * sizeof(float));

  // update position_ids
  *input_pos_.data = static_cast<int32_t>(start_pos);
}

// Explicit instantiations
template class MultimodalTokenGenerator<uint16_t>;
template class MultimodalTokenGenerator<uint8_t>;

} // namespace example
