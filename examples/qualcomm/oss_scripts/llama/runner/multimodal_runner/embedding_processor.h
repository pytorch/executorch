/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/cache_utils.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/imem_alloc.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/embedding_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/utils.h>
#include <memory>
#include <string>

namespace example {
/**
 * @class EmbeddingProcessor
 * @brief Class for processing prompts to generate embeddings using embedding
 * runner.
 */
class EmbeddingProcessor {
 public:
  struct Metadata {
    int32_t context_len;
    int32_t ar_len;
    int32_t vocab_size;
    bool use_int64_token;
    int32_t embedding_dim;
  };

  EmbeddingProcessor(
      EmbeddingRunner* embedding_runner,
      const std::string& method_name,
      Metadata metadata);

  /**
   * @brief Initialize I/O tensor and allocate I/O data buffer.
   * @param buffer_manager Pointer to IMemAlloc instance.
   * @param method_meta Method metadata.
   */
  void init_io(
      IMemAlloc* buffer_manager,
      executorch::runtime::Result<executorch::runtime::MethodMeta> method_meta);

  void update_prompt_embedding(int32_t num_prompt_tokens, int64_t prompt_pos);

  /**
   * Process prompt tokens to generate embeddings.
   * @param prompt_tokens The text prompt tokens. Encoded by tokenizer.
   * @param ar_len AR length for chunking.
   * @return The embedding tensor result.
   */
  void prefill(const std::vector<uint64_t>& prompt_tokens);

  /**
   * @brief Get total I/O size in bytes.
   * @return Total I/O size in bytes.
   */
  inline const size_t total_embedding_processor_io_size_in_bytes() const {
    return input_toks_.size + embeddings_.size;
  }

  inline const TensorStruct<float>& get_prompt_embeddings() const {
    return prompt_embeddings_;
  }

 private:
  /**
   * @brief Fill in I/O buffers with prompt tokens.
   * @param prompt_tokens Vector of prompt tokens.
   */
  void prepare_io(
      const std::vector<uint64_t>& prompt_tokens,
      int64_t prompt_pos);

  EmbeddingRunner* embedding_runner_;
  std::string method_name_;

  // metadata
  Metadata metadata_;

  // inputs and outputs
  TensorStruct<int64_t> input_toks_;
  TensorStruct<float> embeddings_;
  TensorStruct<float> prompt_embeddings_;
  std::vector<float> prompt_embeddings_buffer_;
  std::vector<executorch::aten::TensorImpl::SizesType> prompt_embeddings_sizes_;
  std::vector<executorch::aten::TensorImpl::DimOrderType>
      prompt_embeddings_dim_order_;

  std::vector<executorch::runtime::EValue> inputs_;
  std::vector<executorch::aten::Tensor> input_tensors_;
  std::vector<executorch::aten::Tensor> output_tensors_;
};
} // namespace example
