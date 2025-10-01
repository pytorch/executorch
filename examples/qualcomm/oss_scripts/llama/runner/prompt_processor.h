/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/cache_utils.h>
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
template <typename T>
class PromptProcessor {
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
  };
  PromptProcessor(
      DecoderRunner* decoder_runner,
      KVManager<T>* kv_manager,
      const std::string& method_name,
      Metadata metadata);

  /**
   * @brief Initialize I/O tensor and allocate I/O data buffer.
   * @param buffer_manager Pointer to IMemAlloc instance; by default, it uses a
   * shared buffer with RPC memory.
   * @param method_meta Method metadata.
   */
  void init_io(
      IMemAlloc* buffer_manager,
      executorch::runtime::Result<executorch::runtime::MethodMeta> method_meta);

  /**
   * @brief Get the all logits generated
   *
   * @return std::vector<uint16_t>& all the logits generated
   */
  virtual const std::vector<uint16_t>& get_all_logits();

  /**
   * Prefill an LLM Module with the given text input.
   * @param prompt_tokens The text prompt tokens to the LLM Module. Encoded by
   * tokenizer.
   * @param start_pos The starting position in KV cache of the input in the LLM
   * Module.
   * @param dump_logits Used to save all logits. Only enable when analyzing
   * accuracy.
   * @return The next token of the LLM Module after prefill.
   */
  executorch::runtime::Result<uint64_t> prefill(
      std::vector<uint64_t> prompt_tokens,
      int64_t start_pos,
      bool dump_logits);
  /**
   * @brief Get total I/O size in bytes (excluding the KV cache size)
   * @return Total I/O size in bytes.
   */
  inline const size_t total_prompt_processor_io_size_in_bytes() const {
    if (metadata_.cache_mode == CacheMode::HybridCache) {
      return input_toks_.size + input_pos_.size + attention_mask_.size +
          window_attention_mask_.size + logits_.size;
    } else {
      return input_toks_.size + input_pos_.size + attention_mask_.size +
          logits_.size;
    }
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
  KVManager<T>* kv_manager_;
  std::string method_name_;

  // metadata
  Metadata metadata_;

  // inputs and outputs
  TensorStruct<int64_t> input_toks_;
  TensorStruct<int32_t> input_pos_;
  TensorStruct<uint16_t> attention_mask_;
  TensorStruct<uint16_t> window_attention_mask_;
  TensorStruct<uint16_t> logits_;

  // layer -> TensorImpl
  std::vector<std::unique_ptr<executorch::aten::TensorImpl>> k_cache_in_;
  std::vector<std::unique_ptr<executorch::aten::TensorImpl>> v_cache_in_;
  std::vector<std::unique_ptr<executorch::aten::TensorImpl>> k_cache_out_;
  std::vector<std::unique_ptr<executorch::aten::TensorImpl>> v_cache_out_;

  std::vector<executorch::runtime::EValue> inputs_;
  std::vector<executorch::aten::Tensor> input_tensors_;
  std::vector<executorch::aten::Tensor> output_tensors_;

  // Unused by default, only used when dump_logits_path is provided.
  std::vector<uint16_t> prompt_all_logits_;
};
} // namespace example
