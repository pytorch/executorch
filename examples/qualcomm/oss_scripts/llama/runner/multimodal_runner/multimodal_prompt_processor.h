/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/tok_embedding_processor.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/prompt_processor.h>

namespace example {

/**
 * @class MultimodalPromptProcessor
 * @brief Extended PromptProcessor with multimodal embedding support
 */
class MultimodalPromptProcessor : public example::PromptProcessor {
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

  MultimodalPromptProcessor(
      DecoderRunner* decoder_runner,
      KVManager* kv_manager,
      const std::string& method_name,
      Metadata metadata,
      std::unique_ptr<executorch::extension::MethodMeta> method_meta);

  int64_t get_num_heads() const {
    return metadata_.num_heads;
  }
  int64_t get_num_layers() const {
    return metadata_.num_layers;
  }

  /**
   * @brief Initialize I/O tensor and allocate I/O data buffer.
   * @param buffer_manager Pointer to IMemAlloc instance which depends on
   * kv_updater.
   * @param method_meta Method metadata.
   */
  void init_io(
      IMemAlloc* buffer_manager,
      executorch::runtime::Result<executorch::runtime::MethodMeta> method_meta);

  /**
   * Prefill an Decoder Module with the given embedding input.
   * @param prompt_embedding The embedding tensor from embedding module.
   * @param start_pos The starting position in KV cache of the input in the LLM
   * Module.
   * @param dump_logits Used to save all logits. Only enable when analyzing
   * accuracy.
   * @return The next token of the LLM Module after prefill.
   */
  executorch::runtime::Result<uint64_t> prefill(
      const TensorStruct<float>& prompt_embedding,
      int64_t start_pos,
      bool dump_logits,
      AttentionSinkRopeRunner* attention_sink_rope_runner);

  /**
   * @brief Get total I/O size in bytes (excluding the KV cache size)
   * @return Total I/O size in bytes.
   */
  inline const size_t total_prompt_processor_io_size_in_bytes() const {
    return input_toks_.size + input_pos_.size + attention_mask_.size +
        window_attention_mask_.size + logits_.size + input_embedding_.size;
  }

 private:
  // Reuse members from token_generator
  using PromptProcessor::attention_mask_;
  using PromptProcessor::decoder_runner_;
  using PromptProcessor::input_pos_;
  using PromptProcessor::input_tensors_;
  using PromptProcessor::input_toks_;
  using PromptProcessor::inputs_;
  using PromptProcessor::is_bert;
  using PromptProcessor::k_cache_in_;
  using PromptProcessor::k_cache_out_;
  using PromptProcessor::kv_manager_;
  using PromptProcessor::logits_;
  using PromptProcessor::method_name_;
  using PromptProcessor::output_tensors_;
  using PromptProcessor::prompt_all_logits_;
  using PromptProcessor::v_cache_in_;
  using PromptProcessor::v_cache_out_;
  using PromptProcessor::window_attention_mask_;

  /**
   * @brief Fill in I/O buffers with embedding data and position.
   * @param prompt_embedding The embedding tensor.
   * @param prompt_pos Position of the prompt.
   * @param start_pos Starting position.
   */
  void prepare_io(
      const TensorStruct<float>& prompt_embedding,
      int32_t num_prompt_tokens,
      int64_t prompt_pos,
      int64_t start_pos);

  // metadata specific to multimodal
  Metadata metadata_;

  // Additional input for multimodal
  TensorStruct<float> input_embedding_;
};
} // namespace example
