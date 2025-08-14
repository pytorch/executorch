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
#include <executorch/extension/llm/runner/stats.h>
#include <pytorch/tokenizers/tokenizer.h>
#include <string>

namespace example {
/**
 * @class TokenGenerator
 * @brief Class for generating the token using decoder and key-value manager.
 */
template <typename T>
class TokenGenerator {
 public:
  struct Metadata {
    int32_t context_len;
    int64_t num_heads;
    int64_t num_layers;
    int32_t ar_len;
    int32_t vocab_size;
    bool use_int64_token;
  };
  TokenGenerator(
      tokenizers::Tokenizer* tokenizer,
      DecoderRunner* decoder_runner,
      KVManager<T>* kv_manager,
      const std::string& method_name,
      std::unique_ptr<std::unordered_set<uint64_t>>&& eos_ids,
      Metadata metadata,
      executorch::llm::Stats* stats);

  virtual ~TokenGenerator() = default;
  /**
   * @brief Initialize I/O tensor and allocate I/O data buffer.
   * @param buffer_manager Pointer to IMemAlloc instance which depends on
   * kv_updater.
   * @param method_meta Method metadata.
   */
  virtual void init_io(
      IMemAlloc* buffer_manager,
      executorch::runtime::Result<executorch::runtime::MethodMeta> method_meta);

  /**
   * @brief Get the all logits generated
   *
   * @return std::vector<uint16_t>& all the logits generated
   */
  virtual const std::vector<uint16_t>& get_all_logits();

  /**
     * @brief Generate tokens.
     * @param tokens Vector of input tokens.
     * @param start_pos Starting position for generation.
     * @param seq_len Length of the sequence to generate.
     * @param token_callback Callback function for generated tokens.
     * @return The number of tokens generated.
     */
  virtual executorch::runtime::Result<int64_t> generate(
      std::vector<uint64_t> tokens,
      int64_t start_pos,
      int32_t seq_len,
      std::function<void(const std::string&)> token_callback,
      bool dump_logits);
  inline const size_t total_token_generator_io_size_in_bytes() const {
    return input_toks_.size + input_pos_.size + attention_mask_.size +
        logits_.size;
  }

 protected:
  tokenizers::Tokenizer* tokenizer_;
  DecoderRunner* decoder_runner_;
  KVManager<T>* kv_manager_;
  std::string method_name_;
  std::unique_ptr<std::unordered_set<uint64_t>> eos_ids_;

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

  // stats
  executorch::llm::Stats* stats_;

 private:
  /**
   * @brief Fill in I/O buffers with prompt token and position.
   * @param cur_token Current token.
   * @param start_pos Starting position.
   */
  void prepare_io(uint64_t cur_token, int64_t start_pos);

  // metadata
  Metadata metadata_;

  // Unused by default, only used when dump_logits_path is provided.
  std::vector<uint16_t> token_all_logits_;
};
} // namespace example
