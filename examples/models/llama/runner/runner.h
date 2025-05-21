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
#include <optional>
#include <string>
#include <unordered_map>

#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/text_decoder_runner.h>
#include <executorch/extension/llm/runner/text_prefiller.h>
#include <executorch/extension/llm/runner/text_token_generator.h>
#include <executorch/extension/module/module.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace example {

class ET_EXPERIMENTAL Runner : public executorch::extension::llm::IRunner {
 public:
  // Static factory method to create a Runner instance
  static std::unique_ptr<Runner> create(
      const std::string& model_path,
      const std::string& tokenizer_path,
      std::optional<const std::string> data_path = std::nullopt,
      float temperature = -1.0f);

  // Constructor with dependency injection
  explicit Runner(
      std::unordered_map<std::string, int64_t> metadata,
      std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
      std::unique_ptr<::executorch::extension::Module> module,
      std::unique_ptr<::executorch::extension::llm::TextDecoderRunner>
          text_decoder_runner,
      std::unique_ptr<::executorch::extension::llm::TextPrefiller>
          text_prefiller,
      std::unique_ptr<::executorch::extension::llm::TextTokenGenerator>
          text_token_generator,
      std::unique_ptr<::executorch::extension::llm::Stats> stats,
      float temperature = -1.0f);

  bool is_loaded() const override;
  ::executorch::runtime::Error load() override;
  ::executorch::runtime::Error generate(
      const std::string& prompt,
      const ::executorch::extension::llm::GenerationConfig& config,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const ::executorch::extension::llm::Stats&)>
          stats_callback = {}) override;
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
  std::unique_ptr<::executorch::extension::llm::TextDecoderRunner>
      text_decoder_runner_; // Manage text_decoder_runner_'s lifecycle, make
                            // sure it outlives text_prefiller_ &
                            // text_token_generator_.
  std::unique_ptr<::executorch::extension::llm::TextPrefiller> text_prefiller_;
  std::unique_ptr<::executorch::extension::llm::TextTokenGenerator>
      text_token_generator_;

  // Stats
  std::unique_ptr<::executorch::extension::llm::Stats> stats_;

  // temperature.
  // Deprecated, we should rely on the temperature in GenerationConfig instead.
  float temperature_ = -1.0f;
};

} // namespace example
