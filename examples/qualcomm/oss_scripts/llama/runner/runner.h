/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple llama3.2 runner that includes preprocessing and post processing
// logic. The module takes in a string as input and emits a string as output.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/cache_utils.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/decoder_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/imem_alloc.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/kv_manager.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/prompt_processor.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/token_generator.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/module/module.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace example {

enum DecoderModelVersion {
  kLlama2 = 0,
  kLlama3,
  kGemma3,
  kPhi4,
  kQwen2_5,
  kQwen3,
  kSmollm2_135m,
  kSmollm3
};

enum KvBitWidth {
  kWidth8 = 8,
  kWidth16 = 16,
};

template <typename T>
class Runner : public executorch::extension::llm::IRunner {
 public:
  explicit Runner(
      std::unique_ptr<executorch::extension::Module> module,
      const std::string& decoder_model,
      const std::string& model_path,
      const std::string& tokenizer_path,
      const std::string& performance_output_path,
      const std::string& dump_logits_path,
      const float temperature = 0.8f,
      const int eval_mode = EvalMode::kHybrid,
      const std::string& kv_updater = "SmartMask",
      const int ngram = 0,
      const int window = 0,
      const int gcap = 0,
      std::unique_ptr<tokenizers::Tokenizer> tokenizer = nullptr);

  bool is_loaded() const override;
  executorch::runtime::Error load() override;
  // TODO: Support echo and warming
  executorch::runtime::Error generate(
      const std::string& prompt,
      const executorch::extension::llm::GenerationConfig& config,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const executorch::llm::Stats&)> stats_callback = {})
      override;

  executorch::runtime::Error generate_from_prompt_or_file(
      const std::string& prompt,
      bool tokenized_prompt,
      const executorch::extension::llm::GenerationConfig& config,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const executorch::llm::Stats&)> stats_callback = {});

  executorch::runtime::Error prefill(
      const std::string& prompt,
      const GenerationConfig& config = {}) override;
  void stop() override {};
  void stop() override {};
  void reset() override {};
  executorch::runtime::Result<DecoderModelVersion> get_decoder_model_version();

 private:
  enum EvalMode {
    kKVCached = 0,
    kHybrid,
    kLookaheadDecoding,
    kUnsupported,
  };

  std::unique_ptr<executorch::extension::Module> module_;
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

  DecoderModelVersion decoder_model_version_;
  KVManagerMode kv_updater_;
  std::unique_ptr<IMemAlloc> buffer_manager_;
  std::unique_ptr<KVManager<T>> kv_manager_;
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
  std::unique_ptr<DecoderRunner> decoder_runner_;
  std::unique_ptr<PromptProcessor<T>> prompt_processor_;
  std::unique_ptr<TokenGenerator<T>> token_generator_;

  // stats
  executorch::llm::Stats stats_;
};
} // namespace example
