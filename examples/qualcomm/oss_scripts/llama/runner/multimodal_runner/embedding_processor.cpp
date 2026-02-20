/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/embedding_processor.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>

using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::runtime::Error;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
using executorch::runtime::TensorInfo;

namespace example {

EmbeddingProcessor::EmbeddingProcessor(
    EmbeddingRunner* embedding_runner,
    const std::string& method_name,
    Metadata metadata)
    : embedding_runner_(embedding_runner),
      method_name_(method_name),
      metadata_(metadata) {
  input_toks_.size = metadata_.ar_len * sizeof(int64_t);
  embeddings_.size = metadata_.ar_len * metadata_.embedding_dim * sizeof(float);
  prompt_embeddings_.size = 0; // Will be set in prefill()
}

void EmbeddingProcessor::init_io(
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

  // [O]: embeddings
  Result<TensorInfo> embeddings = method_meta->output_tensor_meta(0);
  embeddings_.data =
      reinterpret_cast<float*>(buffer_manager->allocate(embeddings_.size));

  embeddings_.tensor = std::make_unique<TensorImpl>(
      embeddings->scalar_type(),
      embeddings->sizes().size(),
      const_cast<TensorImpl::SizesType*>(embeddings->sizes().data()),
      embeddings_.data,
      const_cast<TensorImpl::DimOrderType*>(embeddings->dim_order().data()));
  output_tensors_.emplace_back(embeddings_.tensor.get());
  buffer_manager->add_memory_info(
      embeddings_.data, embeddings_.size, embeddings.get());

  inputs_.reserve(input_tensors_.size());
  for (auto& input_tensor : input_tensors_) {
    inputs_.emplace_back(std::move(input_tensor));
  }
}

void EmbeddingProcessor::update_prompt_embedding(
    int32_t num_prompt_tokens,
    int64_t prompt_pos) {
  for (int i = 0; i < metadata_.ar_len; i++) {
    if (prompt_pos + i < num_prompt_tokens) {
      std::memcpy(
          prompt_embeddings_.data + (prompt_pos + i) * metadata_.embedding_dim,
          embeddings_.data + i * metadata_.embedding_dim,
          metadata_.embedding_dim * sizeof(float));
    }
  }
}

void EmbeddingProcessor::prefill(const std::vector<uint64_t>& prompt_tokens) {
  int64_t prompt_pos = 0;
  int32_t num_prompt_tokens = prompt_tokens.size();
  prompt_embeddings_.size =
      num_prompt_tokens * metadata_.embedding_dim * sizeof(float);

  // Allocate memory using std::vector for smart pointer management
  prompt_embeddings_buffer_.resize(num_prompt_tokens * metadata_.embedding_dim);
  prompt_embeddings_.data = prompt_embeddings_buffer_.data();

  // Create TensorImpl for prompt_embeddings_ with shape [1, num_prompt_tokens,
  // dim] Store sizes and dim_order as member variables to keep them
  // alive
  prompt_embeddings_sizes_ = {1, num_prompt_tokens, metadata_.embedding_dim};
  prompt_embeddings_dim_order_ = {0, 1, 2};
  prompt_embeddings_.tensor = std::make_unique<TensorImpl>(
      executorch::aten::ScalarType::Float,
      prompt_embeddings_sizes_.size(),
      prompt_embeddings_sizes_.data(),
      prompt_embeddings_.data,
      prompt_embeddings_dim_order_.data());

  int num_iters = 1 + ((num_prompt_tokens - 1) / metadata_.ar_len);

  ET_CHECK_MSG(
      embedding_runner_->set_outputs(method_name_, output_tensors_) ==
          executorch::runtime::Error::Ok,
      "Failed to set output tensor for module %s",
      method_name_.c_str());

  for (int32_t i = 0; i < num_iters; ++i) {
    prepare_io(prompt_tokens, prompt_pos);

    embedding_runner_->step(method_name_, inputs_);

    // Update prompt_embedding
    update_prompt_embedding(num_prompt_tokens, prompt_pos);

    prompt_pos += metadata_.ar_len;
  }
}

void EmbeddingProcessor::prepare_io(
    const std::vector<uint64_t>& prompt_tokens,
    int64_t prompt_pos) {
  for (int i = 0; i < metadata_.ar_len; i++) {
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

} // namespace example
