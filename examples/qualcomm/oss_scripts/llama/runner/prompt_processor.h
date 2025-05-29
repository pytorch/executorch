/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/decoder_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/imem_alloc.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/kv_manager.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/utils.h>
#include <memory>
#include <string>

namespace example {
/**
 * @class PromptProcessor
 * @brief Class for processing prompts using decoder and key-value manager.
 */
class PromptProcessor {
 public:
  struct Metadata {
    int32_t context_len;
    int64_t num_heads;
    int64_t num_layers;
    int32_t ar_len;
    int32_t vocab_size;
    bool use_int64_token;
  };
  PromptProcessor(
      DecoderRunner* decoder_runner,
      KVManager* kv_manager,
      const std::string& method_name,
      Metadata metadata);

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
   * Prefill an LLM Module with the given text input.
   * @param prompt_tokens The text prompt tokens to the LLM Module. Encoded by
   * tokenizer.
   * @param start_pos The starting position in KV cache of the input in the LLM
   * Module.
   * @return The next token of the LLM Module after prefill.
   */
  executorch::runtime::Result<uint64_t> prefill(
      std::vector<uint64_t> prompt_tokens,
      int64_t start_pos);
  /**
   * @brief Get total I/O size in bytes (excluding the KV cache size)
   * @return Total I/O size in bytes.
   */
  inline const size_t total_prompt_processor_io_size_in_bytes() const {
    return input_toks_.size + input_pos_.size + attention_mask_.size +
        logits_.size;
  }

 private:
  // If the cache length is zero, it indicates a BERT model, which does not use
  // position ids or KV cache inputs.
  bool is_bert() const {
    return metadata_.context_len == metadata_.ar_len;
  }
  /**
   * @brief Fill in I/O buffers with prompt token and position.
   * @param prompt_tokens Vector of prompt tokens.
   * @param prompt_pos Position of the prompt.
   * @param start_pos Starting position.
   */
  void prepare_io(
      const std::vector<uint64_t>& prompt_tokens,
      int64_t prompt_pos,
      int64_t start_pos);
  DecoderRunner* decoder_runner_;
  KVManager* kv_manager_;
  std::string method_name_;

  // metadata
  Metadata metadata_;

  // inputs and outputs
  TensorStruct<int64_t> input_toks_;
  TensorStruct<int32_t> input_pos_;
  TensorStruct<uint16_t> attention_mask_;
  TensorStruct<uint16_t> logits_;

  // layer -> head -> TensorImpl
  std::vector<std::vector<std::unique_ptr<executorch::aten::TensorImpl>>>
      k_cache_in_;
  std::vector<std::vector<std::unique_ptr<executorch::aten::TensorImpl>>>
      v_cache_in_;
  std::vector<std::vector<std::unique_ptr<executorch::aten::TensorImpl>>>
      k_cache_out_;
  std::vector<std::vector<std::unique_ptr<executorch::aten::TensorImpl>>>
      v_cache_out_;

  std::vector<executorch::runtime::EValue> inputs_;
  std::vector<executorch::aten::Tensor> input_tensors_;
  std::vector<executorch::aten::Tensor> output_tensors_;
};
} // namespace example
