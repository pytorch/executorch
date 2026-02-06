/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Multimodal runner that extends the base llama runner with vision capabilities

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/cache_utils.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/decoder_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/imem_alloc.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/kv_manager.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/embedding_processor.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/embedding_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_prompt_processor.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_token_generator.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/module/module.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace example {

// Extend DecoderModelVersion enum with multimodal models
enum MultimodalDecoderModelVersion {
  kSmolvlm = 0,
  kInternvl3,
};

enum KvBitWidth {
  kWidth8 = 8,
  kWidth16 = 16,
};

template <typename T>
class MultimodalRunner : public executorch::extension::llm::IRunner {
 public:
  explicit MultimodalRunner(
      std::unique_ptr<executorch::extension::Module> module,
      std::unique_ptr<executorch::extension::Module> embedding_module,
      const std::string& decoder_model,
      const std::string& model_path,
      const std::string& tokenizer_path,
      const std::string& performance_output_path,
      const std::string& dump_logits_path,
      const float temperature = 0.8f,
      const int eval_mode = EvalMode::kHybrid,
      const bool shared_buffer = false,
      const int ngram = 0,
      const int window = 0,
      const int gcap = 0,
      std::unique_ptr<executorch::aten::Tensor> image_hidden_states = nullptr);

  bool is_loaded() const override;
  executorch::runtime::Error load() override;

  // Override generate to support multimodal inputs
  executorch::runtime::Error generate(
      const std::string& prompt,
      const executorch::extension::llm::GenerationConfig& config,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const executorch::llm::Stats&)> stats_callback = {})
      override;

  // Multimodal-specific generation with image embeddings
  executorch::runtime::Error generate_from_prompt_or_file(
      const std::string& prompt,
      bool tokenized_prompt,
      const executorch::extension::llm::GenerationConfig& config,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const executorch::llm::Stats&)> stats_callback = {});
  void stop() override {};
  void reset() override {};
  executorch::runtime::Result<MultimodalDecoderModelVersion>
  get_decoder_model_version();

  // Multimodal-specific method for merging embeddings
  void merge_multimodal_embeddings(
      const std::vector<uint64_t>& input_ids,
      const TensorStruct<float>& text_embeddings,
      uint64_t placeholder_token_id);

 private:
  enum EvalMode {
    kKVCached = 0,
    kHybrid,
    kLookaheadDecoding,
    kUnsupported,
  };

  // Modules
  std::unique_ptr<executorch::extension::Module> module_;
  std::unique_ptr<executorch::extension::Module> embedding_module_;

  int32_t context_len_{0};

  int ngram_{0};
  int window_{0};
  int gcap_{0};

  // Defaults to StaticCahce, indicating that the model does not use a
  // global/local architecture.
  CacheMode cache_mode_{CacheMode::StaticCahce};
  int64_t cur_pos_{0};

  std::string tokenizer_path_;
  std::string performance_output_path_;
  std::string dump_logits_path_;
  float temperature_;
  EvalMode eval_mode_;
  bool shared_buffer_;

  MultimodalDecoderModelVersion decoder_model_version_;
  std::unique_ptr<IMemAlloc> buffer_manager_;
  std::unique_ptr<KVManager<T>> kv_manager_;
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
  std::unique_ptr<DecoderRunner> decoder_runner_;
  std::unique_ptr<MultimodalPromptProcessor<T>> prompt_processor_;
  std::unique_ptr<MultimodalTokenGenerator<T>> token_generator_;
  std::unique_ptr<EmbeddingRunner> embedding_runner_;
  std::unique_ptr<EmbeddingProcessor> embedding_processor_;
  std::unique_ptr<EmbeddingProcessor> embedding_generator_;

  // Image hidden states storage
  std::unique_ptr<executorch::aten::Tensor> image_hidden_states_;

  // Multimodal embeddings storage
  std::vector<float> multimodal_embeddings_buffer_;
  std::vector<executorch::aten::TensorImpl::SizesType>
      multimodal_embeddings_sizes_;
  std::vector<executorch::aten::TensorImpl::DimOrderType>
      multimodal_embeddings_dim_order_;
  TensorStruct<float> merged_embeddings_;

  // scale and zero point for quantized KV cache
  std::vector<float> input_k_cache_scales_;
  std::vector<T> input_k_cache_zero_points_;
  std::vector<float> input_v_cache_scales_;
  std::vector<T> input_v_cache_zero_points_;
  std::vector<float> output_k_cache_scales_;
  std::vector<T> output_k_cache_zero_points_;
  std::vector<float> output_v_cache_scales_;
  std::vector<T> output_v_cache_zero_points_;

  // stats
  executorch::llm::Stats stats_;
};
} // namespace example
