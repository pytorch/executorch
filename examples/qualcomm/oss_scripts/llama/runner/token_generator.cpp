/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/token_generator.h>
#include <numeric>
using executorch::aten::TensorImpl;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
using executorch::runtime::TensorInfo;

namespace example {
template <typename T>
TokenGenerator<T>::TokenGenerator(
    tokenizers::Tokenizer* tokenizer,
    DecoderRunner* decoder_runner,
    KVManager<T>* kv_manager,
    const std::string& method_name,
    std::unique_ptr<std::unordered_set<uint64_t>>&& eos_ids,
    Metadata metadata,
    executorch::llm::Stats* stats)
    : tokenizer_(tokenizer),
      decoder_runner_(decoder_runner),
      kv_manager_(kv_manager),
      method_name_(method_name),
      eos_ids_(std::move(eos_ids)),
      stats_(stats),
      metadata_(metadata) {
  k_cache_in_.resize(metadata_.num_layers);
  v_cache_in_.resize(metadata_.num_layers);
  k_cache_out_.resize(metadata_.num_layers);
  v_cache_out_.resize(metadata_.num_layers);

  // Calculate I/O size
  input_toks_.size = metadata_.ar_len * sizeof(int64_t);
  input_pos_.size = metadata_.ar_len * sizeof(int32_t);
  attention_mask_.size =
      metadata_.ar_len * metadata_.context_len * sizeof(uint16_t);
  logits_.size = metadata_.ar_len * metadata_.vocab_size * sizeof(uint16_t);
}

template <typename T>
void TokenGenerator<T>::init_io(
    IMemAlloc* buffer_manager,
    Result<MethodMeta> method_meta) {
  input_tensors_.reserve(method_meta->num_inputs());
  output_tensors_.reserve(method_meta->num_outputs());
  // [I]: input_tokens
  Result<TensorInfo> input_toks = method_meta->input_tensor_meta(0);
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
  Result<TensorInfo> attention_mask = method_meta->input_tensor_meta(1);
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

  // [I]: input_pos
  Result<TensorInfo> input_pos = method_meta->input_tensor_meta(2);
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
  int index = 3; // bypass input_tokens, atten_mask, input_pos
  for (int cache_group = 0; cache_group < 2; ++cache_group) {
    std::vector<std::vector<std::unique_ptr<TensorImpl>>>& cache =
        (cache_group == 0 ? k_cache_in_ : v_cache_in_);
    std::vector<std::vector<KVCache<T>>> cache_ptrs = (cache_group == 0)
        ? kv_manager_->get_k_cache_()
        : kv_manager_->get_v_cache_();
    for (int layer = 0; layer < metadata_.num_layers; ++layer) {
      for (int head = 0; head < metadata_.num_heads; ++head, ++index) {
        Result<TensorInfo> kv_cache = method_meta->input_tensor_meta(index);

        T* cache_ptr = cache_ptrs[layer][head].buffer;

        cache[layer].emplace_back(std::make_unique<TensorImpl>(
            kv_cache->scalar_type(),
            kv_cache->sizes().size(),
            const_cast<TensorImpl::SizesType*>(kv_cache->sizes().data()),
            cache_ptr,
            const_cast<TensorImpl::DimOrderType*>(
                kv_cache->dim_order().data())));
        input_tensors_.emplace_back(cache[layer][head].get());
        buffer_manager->add_memory_info(
            cache_ptr, cache[layer][head]->nbytes(), kv_cache.get());
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
  index = 1;
  for (int cache_group = 0; cache_group < 2; ++cache_group) {
    std::vector<std::vector<std::unique_ptr<TensorImpl>>>& cache =
        (cache_group == 0 ? k_cache_out_ : v_cache_out_);
    std::vector<std::vector<KVCache<T>>> cache_ptrs = (cache_group == 0)
        ? kv_manager_->get_k_cache_()
        : kv_manager_->get_v_cache_();
    for (int layer = 0; layer < metadata_.num_layers; ++layer) {
      for (int head = 0; head < metadata_.num_heads; ++head, ++index) {
        Result<TensorInfo> kv_cache = method_meta->output_tensor_meta(index);
        T* cache_ptr = cache_ptrs[layer][head].output_buffer;
        cache[layer].emplace_back(std::make_unique<TensorImpl>(
            kv_cache->scalar_type(),
            kv_cache->sizes().size(),
            const_cast<TensorImpl::SizesType*>(kv_cache->sizes().data()),
            cache_ptr,
            const_cast<TensorImpl::DimOrderType*>(
                kv_cache->dim_order().data())));
        output_tensors_.emplace_back(cache[layer][head].get());
        buffer_manager->add_memory_info(
            cache_ptr, cache[layer][head]->nbytes(), kv_cache.get());
      }
    }
  }
  // Prepare the vector of EValue to run inference
  inputs_.reserve(input_tensors_.size());
  for (auto& input_tensor : input_tensors_) {
    inputs_.emplace_back(std::move(input_tensor));
  }
}

template <typename T>
const std::vector<uint16_t>& TokenGenerator<T>::get_all_logits() {
  return token_all_logits_;
}

// This function only considers the case where token_generator_ar_len equals 1.
template <typename T>
void TokenGenerator<T>::prepare_io(uint64_t cur_token, int64_t start_pos) {
  // update input_tok
  *input_toks_.data =
      metadata_.use_int64_token ? cur_token : static_cast<int32_t>(cur_token);
  // update position_ids
  *input_pos_.data = static_cast<int32_t>(start_pos);
}

template <typename T>
Result<int64_t> TokenGenerator<T>::generate(
    std::vector<uint64_t> tokens,
    int64_t start_pos,
    int32_t seq_len,
    std::function<void(const std::string&)> token_callback,
    bool dump_logits) {
  ET_CHECK_MSG(
      !tokens.empty(), "Token generation loop shouldn't take empty tokens");
  int64_t pos = start_pos; // position in the sequence

  // Token after prefill
  uint64_t cur_token = tokens.back();
  uint64_t prev_token;
  // Rearrange KV cache first
  kv_manager_->rearrange_cache(metadata_.ar_len);
  std::vector<int32_t> attention_map(metadata_.ar_len);
  std::iota(attention_map.begin(), attention_map.end(), -1);
  // Initialize attention mask with current position
  kv_manager_->init_attention_mask(
      attention_mask_.data, attention_map, metadata_.ar_len, pos);
  // Initialize the output of the module
  ET_CHECK_MSG(
      decoder_runner_->set_outputs(method_name_, output_tensors_) ==
          executorch::runtime::Error::Ok,
      "Failed to set output tensor for module %s",
      method_name_.c_str());
  // Generate our tokens
  while (pos < seq_len - 1) {
    // Fill in the token and position data
    prepare_io(cur_token, pos);
    // Only update data pointer of the cache to the tensor for SHIFT_POINTER
    // mode
    bool updated = kv_manager_->update_cache_tensor(
        k_cache_in_,
        k_cache_out_,
        v_cache_in_,
        v_cache_out_,
        metadata_.ar_len,
        pos);
    // Only update the output of module for SHIFT_POINTER mode
    if (updated) {
      // Update the output of the module
      ET_CHECK_MSG(
          decoder_runner_->set_outputs(method_name_, output_tensors_) ==
              executorch::runtime::Error::Ok,
          "Failed to set output tensor for module %s",
          method_name_.c_str());
    }
    // Run inference
    auto logits_res = decoder_runner_->step(method_name_, inputs_);
    if (dump_logits) {
      token_all_logits_.insert(
          token_all_logits_.end(),
          logits_.data,
          logits_.data + metadata_.ar_len * metadata_.vocab_size);
    }
    ET_CHECK_OK_OR_RETURN_ERROR(logits_res.error());
    executorch::aten::Tensor& logits_tensor = logits_res.get();

    prev_token = cur_token;

    stats_->on_sampling_begin();
    cur_token =
        decoder_runner_->logits_to_token(logits_tensor, metadata_.ar_len);
    stats_->on_sampling_end();

    // Update KV Cache with the output results
    kv_manager_->update_cache(metadata_.ar_len, pos, metadata_.ar_len, {});
    // Update attention mask with current position
    kv_manager_->update_attention_mask(
        attention_mask_.data, metadata_.ar_len, pos, metadata_.ar_len);
    pos++;

    // print the token as string, decode it with the Tokenizer object
    token_callback(
        ET_UNWRAP_TOKENIZER(tokenizer_->decode(prev_token, cur_token)));

    // data-dependent terminating condition: we have n_eos_ number of EOS
    if (eos_ids_->count(cur_token) > 0) {
      printf("\n");
      ET_LOG(Info, "\nReached to the end of generation");
      break;
    }
  }
  return pos - start_pos;
}

// Explicit instantiations
template class TokenGenerator<uint16_t>;
template class TokenGenerator<uint8_t>;

} // namespace example
