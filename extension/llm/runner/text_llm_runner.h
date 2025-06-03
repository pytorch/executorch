/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple llama2 runner that includes preprocessing and post processing logic.
// The module takes in a string as input and emits a string as output.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/text_decoder_runner.h>
#include <executorch/extension/llm/runner/text_prefiller.h>
#include <executorch/extension/llm/runner/text_token_generator.h>
#include <executorch/extension/module/module.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace executorch::extension::llm {

static constexpr auto kEnableDynamicShape = "enable_dynamic_shape";
static constexpr auto kBosId = "get_bos_id";
static constexpr auto kEosIds = "get_eos_ids";
static constexpr auto kMaxSeqLen = "get_max_seq_len";
static constexpr auto kMaxContextLen = "get_max_context_len";
static constexpr auto kVocabSize = "get_vocab_size";
static constexpr auto kUseKVCache = "use_kv_cache";
static constexpr auto kUseSDPAWithKVCache = "use_sdpa_with_kv_cache";

class ET_EXPERIMENTAL TextLLMRunner : public IRunner {
 public:
  // Constructor with dependency injection
  explicit TextLLMRunner(
      std::unordered_map<std::string, int64_t> metadata,
      std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
      std::unique_ptr<::executorch::extension::Module> module,
      std::unique_ptr<TextDecoderRunner> text_decoder_runner,
      std::unique_ptr<TextPrefiller> text_prefiller,
      std::unique_ptr<TextTokenGenerator> text_token_generator,
      std::unique_ptr<Stats> stats,
      float temperature = -1.0f);

  bool is_loaded() const override;
  ::executorch::runtime::Error load() override;
  ::executorch::runtime::Error generate(
      const std::string& prompt,
      const GenerationConfig& config,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const Stats&)> stats_callback = {}) override;
  ::executorch::runtime::Error warmup(
      const std::string& prompt,
      int32_t max_new_tokens);
  void stop() override;

 private:
  bool shouldStop_{false};

  // Components
  std::unique_ptr<::tokenizers::Tokenizer> tokenizer_;
  std::unordered_map<std::string, int64_t> metadata_;
  std::unique_ptr<::executorch::extension::Module>
      module_; // Manage module's lifecycle, make sure it outlives
               // text_decoder_runner_.
  std::unique_ptr<TextDecoderRunner>
      text_decoder_runner_; // Manage text_decoder_runner_'s lifecycle, make
                            // sure it outlives text_prefiller_ &
                            // text_token_generator_.
  std::unique_ptr<TextPrefiller> text_prefiller_;
  std::unique_ptr<TextTokenGenerator> text_token_generator_;

  // Stats
  std::unique_ptr<Stats> stats_;

  // temperature.
  // Deprecated, we should rely on the temperature in GenerationConfig instead.
  float temperature_ = -1.0f;
};

std::unique_ptr<tokenizers::Tokenizer> load_tokenizer(
    const std::string& tokenizer_path,
    std::unique_ptr<std::vector<std::string>> special_tokens = nullptr,
    std::optional<std::string> pattern = std::nullopt,
    size_t bos_token_index = 0,
    size_t eos_token_index = 1);

std::unordered_map<std::string, int64_t> get_llm_metadata(
    tokenizers::Tokenizer* tokenizer,
    Module* module);

std::unordered_set<uint64_t> get_eos_ids(
    tokenizers::Tokenizer* tokenizer,
    Module* module);

std::unique_ptr<TextLLMRunner> create_text_llm_runner(
    const std::string& model_path,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    std::optional<const std::string> data_path = std::nullopt,
    float temperature = -1.0f);

} // namespace executorch::extension::llm
