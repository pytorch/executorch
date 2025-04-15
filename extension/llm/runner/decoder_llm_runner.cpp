/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Implementation of the generalized decoder-only runner.

#include <executorch/extension/llm/runner/decoder_llm_runner.h>

#include <cmath>
#include <string>

#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

namespace executorch {
namespace extension {
namespace llm {

using exec_aten::ScalarType;
using exec_aten::Tensor;
using runtime::Error;
using runtime::Result;

DecoderLLMRunner::DecoderLLMRunner(
    std::unique_ptr<executorch::extension::Module> module,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    std::unique_ptr<std::unordered_set<uint64_t>> eos_ids,
    RunnerConfig runner_config)
    : module_(std::move(module)),
      tokenizer_(std::move(tokenizer)),
      runner_config_(std::move(runner_config)),
      current_temperature_(0.8f) {
  
  // Create component objects
  text_decoder_runner_ = std::make_unique<TextDecoderRunner>(
      module_.get(), runner_config_.use_kv_cache, runner_config_.use_sdpa_with_kv_cache);
  
  text_prefiller_ = std::make_unique<TextPrefiller>(
      module_.get(), text_decoder_runner_.get());
  
  text_token_generator_ = std::make_unique<TextTokenGenerator>(
      tokenizer_.get(), 
      text_decoder_runner_.get(),
      std::move(eos_ids),
      runner_config_.max_seq_len);
}

Result<std::unique_ptr<DecoderLLMRunner>> DecoderLLMRunner::create(
    const std::string& model_path,
    const std::string& tokenizer_path,
    std::optional<RunnerConfig> runner_config) {
  
  // Load model from path
  auto moduleResult = executorch::extension::Module::load_from_file(model_path);
  if (!moduleResult.ok()) {
    return moduleResult.error();
  }
  auto module = std::make_unique<executorch::extension::Module>(std::move(moduleResult.get()));
  
  // Load tokenizer from path
  ::tokenizers::Result<::tokenizers::Tokenizer*> tokenizer_result =
      ::tokenizers::Tokenizer::from_file(tokenizer_path.c_str());
  if (!tokenizer_result.is_ok()) {
    return runtime::Error(
        runtime::ErrorCode::InvalidArgument,
        "Failed to load tokenizer from path: " + tokenizer_path);
  }
  auto tokenizer = std::unique_ptr<::tokenizers::Tokenizer>(tokenizer_result.value);
  
  // Create default config if not provided
  RunnerConfig final_config;
  if (runner_config.has_value()) {
    final_config = runner_config.value();
  }
  
  // Create the runner
  auto runner = std::unique_ptr<DecoderLLMRunner>(new DecoderLLMRunner(
      std::move(module),
      std::move(tokenizer),
      nullptr,  // No EOS IDs provided
      final_config));
  
  // Initialize runner components
  runner->load_and_merge_metadata();
  
  return runner;
}

bool DecoderLLMRunner::is_loaded() const {
  return module_ && module_->is_loaded() && tokenizer_ && text_decoder_runner_ && 
         text_prefiller_ && text_token_generator_;
}

Error DecoderLLMRunner::load() {
  if (!module_) {
    return Error(runtime::ErrorCode::InvalidState, "Module not initialized");
  }

  Error err = module_->load();
  if (!err.ok()) {
    return err;
  }
  
  // Load metadata and update config if needed
  load_and_merge_metadata();
  
  return Error::Ok();
}

Error DecoderLLMRunner::generate(
    const std::string& prompt,
    const GenerationConfig& config,
    std::function<void(const std::string&)> token_callback,
    std::function<void(const Stats&)> stats_callback) {
  
  if (!is_loaded()) {
    return Error(runtime::ErrorCode::InvalidState, "Model is not loaded");
  }
  
  // Reset state
  shouldStop_ = false;
  stats_ = Stats();
  current_temperature_ = config.temperature;
  
  // Record if this is a warmup run
  const bool is_warming = config.warming;
  
  // Run the text generation process
  auto start_prefill = std::chrono::high_resolution_clock::now();
  
  // Tokenize and prefill the prompt
  auto tokenization_result = text_prefiller_->prefill(prompt);
  if (!tokenization_result.ok()) {
    return tokenization_result.error();
  }
  
  auto prefill_result = tokenization_result.get();
  stats_.prompt_tokens = prefill_result.input_tokens.size();
  
  auto end_prefill = std::chrono::high_resolution_clock::now();
  stats_.prefill_time_ms = 
      std::chrono::duration_cast<std::chrono::milliseconds>(
          end_prefill - start_prefill).count();
  
  // Echo the prompt if requested
  if (config.echo && !is_warming) {
    token_callback(prompt);
  }
  
  // Generate tokens
  auto start_decode = std::chrono::high_resolution_clock::now();
  auto err = text_token_generator_->generate(
      prefill_result,
      config.max_new_tokens,
      current_temperature_,
      [&](const std::string& token) {
        stats_.generation_tokens++;
        if (!is_warming) {
          token_callback(token);
        }
      },
      [this]() { return shouldStop_; });
  
  auto end_decode = std::chrono::high_resolution_clock::now();
  stats_.decode_time_ms = 
      std::chrono::duration_cast<std::chrono::milliseconds>(
          end_decode - start_decode).count();
  
  // Report stats if requested
  if (stats_callback) {
    stats_callback(stats_);
  }
  
  return err;
}

void DecoderLLMRunner::stop() {
  shouldStop_ = true;
}

Error DecoderLLMRunner::warmup(const std::string& prompt, int32_t max_new_tokens) {
  GenerationConfig config;
  config.max_new_tokens = max_new_tokens;
  config.warming = true;
  config.echo = false;
  
  return generate(
      prompt,
      config,
      [](const std::string&) {}, // Empty token callback
      [](const Stats&) {}); // Empty stats callback
}

void DecoderLLMRunner::load_and_merge_metadata() {
  if (!module_ || !module_->is_loaded()) {
    return;
  }
  
  // Try to load model-specific configuration from module metadata
  // This can override default values in runner_config_
  
  // Example: load max_seq_len from model metadata
  auto metadata = module_->get_metadata();
  if (metadata.find("max_seq_len") != metadata.end()) {
    try {
      runner_config_.max_seq_len = std::stoi(metadata["max_seq_len"]);
    } catch (...) {
      // Ignore conversion errors
    }
  }
  
  // Example: load max_context_len from model metadata
  if (metadata.find("max_context_len") != metadata.end()) {
    try {
      runner_config_.max_context_len = std::stoi(metadata["max_context_len"]);
    } catch (...) {
      // Ignore conversion errors
    }
  }
  
  // Update components with new config parameters
  if (text_token_generator_) {
    text_token_generator_->set_max_seq_len(runner_config_.max_seq_len);
  }
}

} // namespace llm
} // namespace extension
} // namespace executorch