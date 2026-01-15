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
#include <variant>

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/cache_utils.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/decoder_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/imem_alloc.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/kv_manager.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/encoder.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_embedding_merger.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_prompt_processor.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/multimodal_token_generator.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/tok_embedding_processor.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/multimodal_runner/tok_embedding_runner.h>
#include <executorch/extension/llm/runner/image.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/module/module.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace example {

enum class Modality {
  kAudio = 0,
  kVision,
};

enum class VisionLanguageModel {
  kSmolvlm = 0,
  kInternvl3,
};

// TODO: Add audio models when they are supported
enum class AudioLanguageModel {};

using ModelVersion = std::variant<VisionLanguageModel, AudioLanguageModel>;

constexpr Modality modality_of(const VisionLanguageModel& vlm) {
  return Modality::kVision;
}

constexpr Modality modality_of(const AudioLanguageModel& alm) {
  return Modality::kAudio;
}

inline Modality modality_of(const ModelVersion& model_version) {
  return std::visit(
      [](const auto& model) { return modality_of(model); }, model_version);
}

enum KvBitWidth {
  kWidth8 = 8,
  kWidth16 = 16,
};

template <typename T>
class MultimodalRunner : public executorch::extension::llm::IRunner {
 public:
  explicit MultimodalRunner(
      std::unique_ptr<executorch::extension::Module> encoder,
      std::unique_ptr<executorch::extension::Module> tok_embedding,
      std::unique_ptr<executorch::extension::Module> text_decoder,
      const std::string& model_version,
      const std::string& tokenizer_path,
      const std::string& performance_output_path,
      const std::string& dump_logits_path,
      const float temperature = 0.8f,
      const int eval_mode = EvalMode::kHybrid,
      const bool shared_buffer = false,
      const int ngram = 0,
      const int window = 0,
      const int gcap = 0);

  bool is_loaded() const override;
  executorch::runtime::Error load() override;

  executorch::runtime::Error generate(
      const std::string& prompt,
      const executorch::extension::llm::GenerationConfig& config,
      std::function<void(const std::string&)> token_callback,
      std::function<void(const executorch::llm::Stats&)> stats_callback)
      override;

  executorch::runtime::Error generate_from_prompt_or_file(
      std::vector<executorch::extension::llm::MultimodalInput> inputs,
      bool tokenized_prompt,
      const executorch::extension::llm::GenerationConfig& config,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const executorch::llm::Stats&)> stats_callback = {});
  void stop() override {};
  void reset() override {};
  executorch::runtime::Result<ModelVersion> get_model_version();
  executorch::runtime::Result<executorch::runtime::MethodMeta>
  get_encoder_method_meta();

 private:
  enum EvalMode {
    kKVCached = 0,
    kHybrid,
    kLookaheadDecoding,
    kUnsupported,
  };

  // Modules
  std::unique_ptr<executorch::extension::Module> encoder_;
  std::unique_ptr<executorch::extension::Module> tok_embedding_;
  std::unique_ptr<executorch::extension::Module> text_decoder_;

  inline static const std::string kEncoderForwardName = "forward";

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

  ModelVersion model_version_;
  std::unique_ptr<IMemAlloc> buffer_manager_;
  std::unique_ptr<KVManager<T>> kv_manager_;
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
  std::unique_ptr<DecoderRunner> decoder_runner_;
  std::unique_ptr<MultimodalPromptProcessor<T>> prompt_processor_;
  std::unique_ptr<MultimodalTokenGenerator<T>> token_generator_;
  std::unique_ptr<EncoderRunner> encoder_runner_;
  std::unique_ptr<TokenEmbeddingRunner> tok_embedding_runner_;
  std::unique_ptr<TokenEmbeddingProcessor> tok_embedding_processor_;
  std::unique_ptr<TokenEmbeddingProcessor> tok_embedding_generator_;
  std::unique_ptr<MultimodalEmbeddingMerger> embedding_merger_;

  // Placeholder token ID for image inputs. This value will be set from the
  // model's metadata. A default of 0 indicates that the vision modality is not
  // supported.
  uint64_t image_token_id_{0};

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
