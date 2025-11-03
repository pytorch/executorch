/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/embedding_processor.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/token_generator.h>

namespace example {

/**
 * @class MultimodalTokenGenerator
 * @brief Extended TokenGenerator with multimodal embedding support
 */
template <typename T>
class MultimodalTokenGenerator : public example::TokenGenerator<T> {
 public:
  struct Metadata {
    int32_t context_len;
    int64_t num_heads;
    int64_t num_layers;
    int32_t ar_len;
    int32_t vocab_size;
    bool use_int64_token;
    int sliding_window;
    CacheMode cache_mode;
    int32_t embedding_dim = 0;
  };

  // Constructor with embedding generator support
  MultimodalTokenGenerator(
      tokenizers::Tokenizer* tokenizer,
      EmbeddingProcessor* embedding_runner,
      DecoderRunner* decoder_runner,
      KVManager<T>* kv_manager,
      const std::string& method_name,
      std::unique_ptr<std::unordered_set<uint64_t>>&& eos_ids,
      Metadata metadata,
      executorch::llm::Stats* stats);

  virtual ~MultimodalTokenGenerator() = default;

  /**
   * @brief Initialize I/O tensor and allocate I/O data buffer with embedding
   * support.
   */
  void init_io(
      IMemAlloc* buffer_manager,
      executorch::runtime::Result<executorch::runtime::MethodMeta> method_meta)
      override;

  inline const size_t total_token_generator_io_size_in_bytes() const {
    if (metadata_.cache_mode == CacheMode::HybridCache) {
      return input_toks_.size + input_pos_.size + attention_mask_.size +
          window_attention_mask_.size + logits_.size + input_embedding_.size;
    } else {
      return input_toks_.size + input_pos_.size + attention_mask_.size +
          logits_.size + input_embedding_.size;
    }
  }

 protected:
  // Reuse members from token_generator
  using TokenGenerator<T>::kv_manager_;
  using TokenGenerator<T>::input_pos_;
  using TokenGenerator<T>::attention_mask_;
  using TokenGenerator<T>::window_attention_mask_;
  using TokenGenerator<T>::inputs_;
  using TokenGenerator<T>::input_tensors_;
  using TokenGenerator<T>::output_tensors_;

  // Additional members specific to multimodal
  TensorStruct<float> input_embedding_;

 private:
  // Reuse members from token_generator
  using TokenGenerator<T>::input_toks_;
  using TokenGenerator<T>::logits_;
  using TokenGenerator<T>::k_cache_in_;
  using TokenGenerator<T>::v_cache_in_;
  using TokenGenerator<T>::k_cache_out_;
  using TokenGenerator<T>::v_cache_out_;

  // Additional members specific to multimodal
  EmbeddingProcessor* embedding_runner_;

  /**
   * @brief Fill in I/O buffers with prompt token and position.
   * @param cur_token Current token.
   * @param start_pos Starting position.
   */
  void prepare_io(uint64_t cur_token, int64_t start_pos) override;

  // metadata specific to multimodal
  Metadata metadata_;
};
} // namespace example
