/*
 * Copyright (c) 2026 Samsung Electronics Co. LTD
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <unordered_map>

#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/llm/runner/text_decoder_runner.h>
#include <executorch/extension/module/module.h>
#include <pytorch/tokenizers/tokenizer.h>

#include <executorch/examples/samsung/executor_runner/llm/runner/config.h>
#include <executorch/examples/samsung/executor_runner/llm/runner/prefiller.h>
#include <executorch/examples/samsung/executor_runner/llm/runner/token_generator.h>

namespace example {

namespace llm = ::executorch::extension::llm;
namespace runtime = ::executorch::runtime;
namespace extension = ::executorch::extension;

class LLMRunner : public llm::IRunner {
 public:
  explicit LLMRunner(
      std::unordered_map<std::string, int64_t> metadata,
      std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
      std::unique_ptr<extension::Module> module,
      std::unique_ptr<llm::TextDecoderRunner> text_decoder_runner,
      std::unique_ptr<llm::TextPrefiller> text_prefiller,
      std::unique_ptr<example::TokenGenerator> text_token_generator,
      std::unique_ptr<llm::Stats> stats,
      std::unique_ptr<llm::IOManager> io_manager,
      RunnerConfig config,
      DecoderModelVersion model_version,
      float temperature = -1.0f);

  bool is_loaded() const;

  runtime::Error load();

  runtime::Error generate(
      const std::string& prompt,
      const llm::GenerationConfig& config,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const llm::Stats&)> stats_callback = {});

  // TODO (brucekk.kim): consider removing unncessary args
  runtime::Error generate_from_pos(
      const std::string& prompt,
      int64_t start_pos,
      const llm::GenerationConfig& config,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const llm::Stats&)> stats_callback = {});

  runtime::Error warmup(const std::string& prompt, int32_t max_new_tokens);

  void stop();

  const RunnerConfig& get_config() const {
    return config_;
  }
  extension::Module* get_module() const {
    return module_.get();
  }
  ::tokenizers::Tokenizer* get_tokenizer() const {
    return tokenizer_.get();
  }
  const std::unordered_map<std::string, int64_t>& get_metadata() const {
    return metadata_;
  }

  void set_io_manager(std::unique_ptr<llm::IOManager> io_manager) {
    io_manager_ = std::move(io_manager);
  }
  void set_text_decoder_runner(std::unique_ptr<llm::TextDecoderRunner> runner) {
    text_decoder_runner_ = std::move(runner);
  }
  void set_text_prefiller(std::unique_ptr<llm::TextPrefiller> prefiller) {
    text_prefiller_ = std::move(prefiller);
  }
  void set_text_token_generator(
      std::unique_ptr<example::TokenGenerator> generator) {
    text_token_generator_ = std::move(generator);
  }

  // TODO (brucekk.kim): should it be static?
  static std::string get_formatted_prompt(
      const std::string& prompt,
      const std::string& system_prompt,
      DecoderModelVersion version);

  void reset() {};

 private:
  bool shouldStop_{false};

  RunnerConfig config_;
  std::unique_ptr<::tokenizers::Tokenizer> tokenizer_;
  std::unordered_map<std::string, int64_t> metadata_;
  std::unique_ptr<extension::Module> module_;
  std::unique_ptr<llm::TextDecoderRunner> text_decoder_runner_;
  std::unique_ptr<llm::TextPrefiller> text_prefiller_;
  std::unique_ptr<example::TokenGenerator> text_token_generator_;
  std::unique_ptr<llm::IOManager> io_manager_;

  std::unique_ptr<llm::Stats> stats_;

  float temperature_ = -1.0f;
  DecoderModelVersion model_version_;
};

std::unique_ptr<llm::IRunner> create_llm_runner(
    const RunnerPaths& runner_paths,
    RunnerConfig runner_config);

std::unique_ptr<llm::IRunner> create_llm_runner(
    const RunnerPaths& runner_paths,
    RunnerConfig runner_config,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer);

} // namespace example
