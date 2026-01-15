/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/utils.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

#include <memory>
#include <string>
#include <vector>

namespace example {

/**
 * @class MultimodalEmbeddingMerger
 * @brief Merges text and image embeddings based on token IDs
 *
 * This class collects text and image embeddings separately, then merges them
 * based on input token IDs. When a placeholder token ID is encountered,
 * it inserts the corresponding image embedding. Otherwise, it inserts the text
 * embedding for that token position.
 */
enum class EmbeddingType { kText, kImage };

class MultimodalEmbeddingMerger {
 public:
  /**
   * @brief Construct a new Multimodal Embedding Merger
   *
   * @param embedding_dim Expected embedding dimension for all inputs
   */
  explicit MultimodalEmbeddingMerger(int32_t embedding_dim);

  /**
   * @brief Reset the merger state for a new sequence
   */
  void reset();

  /**
   * @brief Add text embeddings to the collection
   *
   * @param text_embeddings Text embedding tensor [1, num_tokens, embedding_dim]
   */
  void add_text_embeddings(const TensorStruct<float>& text_embeddings);

  /**
   * @brief Add image embeddings to the collection
   *
   * @param image_embeddings Image embedding tensor [1, num_tokens,
   * embedding_dim]
   */
  void add_image_embeddings(const executorch::aten::Tensor& image_embeddings);

  /**
   * @brief Merge collected embeddings based on input token IDs
   *
   * This method examines each token ID in input_ids. When it encounters
   * placeholder_token_id, it inserts the next image embedding. Otherwise,
   * it inserts the text embedding at the corresponding position.
   *
   * @param input_ids Vector of token IDs (including placeholder tokens)
   * @param image_token_id Token ID that represents image modality placeholder
   * @return TensorStruct<float> Merged embeddings [1, total_tokens,
   * embedding_dim]
   */
  TensorStruct<float> merge(
      const std::vector<uint64_t>& input_ids,
      uint64_t image_token_id);

  /**
   * @brief Get the total number of tokens after merging
   * @return int64_t Total token count
   */
  inline size_t get_total_tokens() const {
    return total_tokens_;
  }

 private:
  void add_embeddings(
      const executorch::aten::Tensor& embeddings,
      const float* data,
      EmbeddingType type);

  // Expected embedding dimension
  int32_t embedding_dim_;

  // Total tokens after merge
  int32_t total_tokens_{0};

  // Collected embeddings before merge
  // Text embeddings are copied to prevent external modifications
  std::vector<std::vector<float>> text_embedding_buffers_;
  std::vector<int64_t> text_embedding_token_counts_;

  // Image embeddings are copied since they're temporary
  std::vector<std::vector<float>> image_embedding_buffers_;
  std::vector<int64_t> image_embedding_token_counts_;
};

} // namespace example
