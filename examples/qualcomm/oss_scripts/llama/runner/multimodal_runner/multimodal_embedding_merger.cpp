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
  embeddings_.clear();
  total_tokens_ = 0;
}

void MultimodalEmbeddingMerger::append_data(
    const float* data,
    ssize_t ndim,
    const executorch::aten::SizesType* sizes) {
  ET_CHECK_MSG(ndim == 3, "Embeddings must be a 3D tensor");
  ET_CHECK_MSG(sizes[0] == 1, "Batch size must be 1");
  const int32_t num_tokens = static_cast<int32_t>(sizes[1]);
  const int32_t dim = static_cast<int32_t>(sizes[2]);
  ET_CHECK_MSG(
      dim == embedding_dim_,
      "Embedding dimension mismatch: expected %d, got %d",
      embedding_dim_,
      dim);
  const size_t num_elements =
      static_cast<size_t>(num_tokens) * static_cast<size_t>(embedding_dim_);
  embeddings_.insert(embeddings_.end(), data, data + num_elements);
  total_tokens_ += num_tokens;
}

void MultimodalEmbeddingMerger::add_embeddings(
    const TensorStruct<float>& embedding) {
  ET_CHECK_MSG(embedding.tensor != nullptr, "Embedding tensor cannot be null");
  ET_CHECK_MSG(embedding.data != nullptr, "Embedding data cannot be null");
  append_data(
      embedding.data,
      embedding.tensor->dim(),
      embedding.tensor->sizes().data());
  ET_LOG(Info, "Merged TensorStruct embedding: total_tokens=%d", total_tokens_);
}

void MultimodalEmbeddingMerger::add_embeddings(
    const executorch::aten::Tensor& embedding) {
  append_data(
      embedding.const_data_ptr<float>(),
      embedding.dim(),
      embedding.sizes().data());
  ET_LOG(Info, "Merged Tensor embedding: total_tokens=%d", total_tokens_);
}

TensorStruct<float> MultimodalEmbeddingMerger::get_merged_embeddings() {
  ET_CHECK_MSG(
      !embeddings_.empty(),
      "No embeddings to return. Call add_embeddings() first.");

  sizes_ = {
      static_cast<executorch::aten::TensorImpl::SizesType>(1),
      static_cast<executorch::aten::TensorImpl::SizesType>(total_tokens_),
      static_cast<executorch::aten::TensorImpl::SizesType>(embedding_dim_)};

  TensorStruct<float> result;
  result.data = embeddings_.data();
  result.size = embeddings_.size() * sizeof(float);
  result.tensor = std::make_unique<executorch::aten::TensorImpl>(
      executorch::aten::ScalarType::Float,
      sizes_.size(),
      sizes_.data(),
      result.data);

  ET_LOG(
      Info,
      "Get merged embeddings: total_tokens=%d, embedding_dim=%d",
      total_tokens_,
      embedding_dim_);

  return result;
}

} // namespace example
