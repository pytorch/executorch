/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_embedding_merger.h>
#include <executorch/runtime/platform/log.h>

namespace example {

MultimodalEmbeddingMerger::MultimodalEmbeddingMerger(int32_t embedding_dim)
    : embedding_dim_(embedding_dim) {
  ET_CHECK_MSG(embedding_dim_ > 0, "Embedding dimension must be positive");
}

void MultimodalEmbeddingMerger::reset() {
  text_embedding_buffers_.clear();
  text_embedding_token_counts_.clear();
  image_embedding_buffers_.clear();
  image_embedding_token_counts_.clear();
  total_tokens_ = 0;
}

void MultimodalEmbeddingMerger::add_embeddings(
    const executorch::aten::Tensor& embeddings,
    const float* data,
    EmbeddingType type) {
  // shape: [1, num_tokens, embedding_dim]
  ET_CHECK_MSG(embeddings.dim() == 3, "Embeddings must be a 3D tensor");

  size_t batch_size = embeddings.sizes()[0];
  size_t num_tokens = embeddings.sizes()[1];
  size_t dim = embeddings.sizes()[2];

  ET_CHECK_MSG(batch_size == 1, "Batch size must be 1");
  ET_CHECK_MSG(
      dim == embedding_dim_,
      "Embedding dimension mismatch: expected %zu, got %zu",
      embedding_dim_,
      dim);

  // Copy embedding data to prevent it from being overwritten
  size_t num_elements = num_tokens * dim;
  std::vector<float> buffer(data, data + num_elements);

  std::string type_str = (type == EmbeddingType::kText) ? "text" : "image";
  if (type == EmbeddingType::kText) {
    text_embedding_buffers_.emplace_back(std::move(buffer));
    text_embedding_token_counts_.push_back(num_tokens);
  } else {
    image_embedding_buffers_.emplace_back(std::move(buffer));
    image_embedding_token_counts_.push_back(num_tokens);
  }

  ET_LOG(
      Info,
      "Added %s embeddings: num_tokens=%zu",
      type_str.c_str(),
      num_tokens);
}

void MultimodalEmbeddingMerger::add_text_embeddings(
    const TensorStruct<float>& text_embeddings) {
  ET_CHECK_MSG(
      text_embeddings.tensor != nullptr,
      "Text embeddings tensor cannot be null");
  ET_CHECK_MSG(
      text_embeddings.data != nullptr, "Text embeddings data cannot be null");

  executorch::aten::Tensor tensor_wrapper(text_embeddings.tensor.get());

  add_embeddings(tensor_wrapper, text_embeddings.data, EmbeddingType::kText);
}

void MultimodalEmbeddingMerger::add_image_embeddings(
    const executorch::aten::Tensor& image_embeddings) {
  add_embeddings(
      image_embeddings,
      image_embeddings.const_data_ptr<float>(),
      EmbeddingType::kImage);
}

TensorStruct<float> MultimodalEmbeddingMerger::merge(
    const std::vector<uint64_t>& input_ids,
    uint64_t image_token_id) {
  ET_CHECK_MSG(!input_ids.empty(), "input_ids cannot be empty");
  ET_CHECK_MSG(
      !text_embedding_buffers_.empty(),
      "No text embeddings added. Call add_text_embeddings() first.");

  // Final merged embeddings
  std::vector<float> merged_buffer;
  std::vector<executorch::aten::TensorImpl::SizesType> sizes;
  TensorStruct<float> merged_embeddings;

  size_t num_placeholder_tokens = 0;
  if (image_token_id != 0) {
    for (uint64_t token_id : input_ids) {
      if (token_id == image_token_id) {
        num_placeholder_tokens++;
      }
    }
  }

  ET_CHECK_MSG(
      num_placeholder_tokens == image_embedding_buffers_.size(),
      "Number of placeholder tokens (%zu) must match number of image embeddings (%zu)",
      num_placeholder_tokens,
      image_embedding_buffers_.size());

  // Calculate total tokens: sum of all text tokens + all image tokens
  for (int64_t count : text_embedding_token_counts_) {
    total_tokens_ += count;
  }
  for (int64_t count : image_embedding_token_counts_) {
    total_tokens_ += count;
  }
  total_tokens_ = total_tokens_ - num_placeholder_tokens;

  size_t total_elements = total_tokens_ * embedding_dim_;
  merged_buffer.resize(total_elements);

  // Merge embeddings based on input_ids
  size_t text_emb_idx = 0; // Which text embedding chunk in current turn
  size_t text_token_idx = 0; // Token index within current text embedding chunk
  size_t image_emb_idx = 0; // Which image embedding chunk in current turn
  size_t output_offset = 0; // Output buffer offset

  for (int i = 0; i < input_ids.size(); i++) {
    uint64_t token_id = input_ids[i];

    if (image_token_id != 0 && token_id == image_token_id) {
      // Insert entire image embedding
      ET_CHECK_MSG(
          image_emb_idx < image_embedding_buffers_.size(),
          "Image index out of bounds");

      const std::vector<float>& image_buffer =
          image_embedding_buffers_[image_emb_idx];
      int64_t num_image_tokens = image_embedding_token_counts_[image_emb_idx];

      size_t num_elements = num_image_tokens * embedding_dim_;
      std::memcpy(
          merged_buffer.data() + output_offset,
          image_buffer.data(),
          num_elements * sizeof(float));

      output_offset += num_elements;
      image_emb_idx++;
      text_token_idx++; // Skip this image placeholder token
    } else {
      // Insert one text token embedding
      ET_CHECK_MSG(
          text_emb_idx < text_embedding_buffers_.size(),
          "Text embedding index out of bounds");

      const std::vector<float>& text_buffer =
          text_embedding_buffers_[text_emb_idx];
      std::memcpy(
          merged_buffer.data() + output_offset,
          text_buffer.data() + text_token_idx * embedding_dim_,
          embedding_dim_ * sizeof(float));

      output_offset += embedding_dim_;
      text_token_idx++;
    }
  }

  ET_CHECK_MSG(
      image_emb_idx == image_embedding_buffers_.size(),
      "Not all image embeddings were used: used %zu, expected %zu",
      image_emb_idx,
      image_embedding_buffers_.size());

  // Setup tensor metadata
  merged_embeddings.data = merged_buffer.data();
  merged_embeddings.size = total_elements * sizeof(float);

  // Setup sizes and dim_order: [1, total_tokens, embedding_dim]
  sizes = {1, total_tokens_, embedding_dim_};

  // Create TensorImpl
  merged_embeddings.tensor = std::make_unique<executorch::aten::TensorImpl>(
      executorch::aten::ScalarType::Float,
      sizes.size(),
      sizes.data(),
      merged_embeddings.data);

  ET_LOG(
      Info,
      "Merged embeddings: total_tokens=%d, text=%zu, images=%zu, embedding_dim=%d",
      total_tokens_,
      text_embedding_buffers_.size(),
      image_embedding_buffers_.size(),
      embedding_dim_);

  return merged_embeddings;
}

} // namespace example
