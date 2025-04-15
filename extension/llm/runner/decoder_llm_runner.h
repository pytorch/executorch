/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A generalized decoder-only runner that includes preprocessing and post-processing logic.
// Not tied to any specific model architecture.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/text_decoder_runner.h>
#include <executorch/extension/llm/runner/text_prefiller.h>
#include <executorch/extension/llm/runner/text_token_generator.h>
#include <executorch/extension/module/module.h>
#include <pytorch/tokenizers/tokenizer.h>

namespace executorch {
namespace extension {
namespace llm {

// Configuration struct for model-specific runner parameters
struct RunnerConfig {
  int32_t max_seq_len = 2048; // Maximum sequence length supported by the model
  int32_t max_context_len = 2048; // Maximum context length (may be less than max_seq_len)
  bool use_kv_cache = true; // Whether to use KV cache for efficient decoding
  bool enable_dynamic_shape = false; // Whether to enable dynamic shape handling
  bool use_sdpa_with_kv_cache = true; // Whether to use SDPA with KV cache (for performance)
};

// Configuration struct for generation parameters
struct GenerationConfig {
  // Temperature for sampling (higher = more random)
  float temperature = 0.8f;
  
  // Whether to echo the input prompt in the output
  bool echo = true;
  
  // Maximum number of new tokens to generate
  int32_t max_new_tokens = 20;
  
  // Whether this is a warmup run (affects logging)
  bool warming = false;
};

class ET_API DecoderLLMRunner : public IRunner {
 public:
  /**
   * Create a new DecoderLLMRunner with the provided components.
   *
   * @param module The ExecuTorch module to use for inference
   * @param tokenizer The tokenizer to use for encoding/decoding text
   * @param eos_ids Optional set of token IDs that signal the end of a sequence
   * @param runner_config Runner configuration parameters
   */
  DecoderLLMRunner(
      std::unique_ptr<executorch::extension::Module> module,
      std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
      std::unique_ptr<std::unordered_set<uint64_t>> eos_ids = nullptr,
      RunnerConfig runner_config = RunnerConfig());

  /**
   * Create a DecoderLLMRunner from a model path.
   * This is a factory method that creates a new runner instance.
   *
   * @param model_path Path to the model file
   * @param tokenizer_path Path to the tokenizer file
   * @param runner_config Runner configuration parameters (optional)
   * @return A result containing either the runner or an error
   */
  static runtime::Result<std::unique_ptr<DecoderLLMRunner>> create(
      const std::string& model_path,
      const std::string& tokenizer_path,
      std::optional<RunnerConfig> runner_config = std::nullopt);

  /**
   * Check if the runner is loaded and ready for inference.
   *
   * @return true if the runner is loaded, false otherwise
   */
  bool is_loaded() const override;

  /**
   * Load the model and prepare for inference.
   *
   * @return Error::Ok if successful, an error otherwise
   */
  runtime::Error load();

  /**
   * Generate text based on the provided prompt and generation config.
   *
   * @param prompt The input prompt to generate from
   * @param config Generation configuration parameters 
   * @param token_callback Callback function called for each generated token
   * @param stats_callback Callback function for generation statistics
   * @return Error::Ok if successful, an error otherwise
   */
  runtime::Error generate(
      const std::string& prompt,
      const GenerationConfig& config,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const Stats&)> stats_callback = {});

  /**
   * Run a warmup pass with the given prompt.
   *
   * @param prompt The input prompt to use for warming up
   * @param max_new_tokens Maximum number of new tokens to generate
   * @return Error::Ok if successful, an error otherwise
   */
  runtime::Error warmup(
      const std::string& prompt, 
      int32_t max_new_tokens = 20);

  /**
   * Stop the generation process.
   */
  void stop() override;

  /**
   * Get the current runner configuration
   */
  const RunnerConfig& get_runner_config() const {
    return runner_config_;
  }

 private:
  // Read model metadata from the module and merge with runner_config
  void load_and_merge_metadata();
  
  std::unique_ptr<executorch::extension::Module> module_;
  std::unique_ptr<::tokenizers::Tokenizer> tokenizer_;
  std::unique_ptr<TextDecoderRunner> text_decoder_runner_;
  std::unique_ptr<TextPrefiller> text_prefiller_;
  std::unique_ptr<TextTokenGenerator> text_token_generator_;
  RunnerConfig runner_config_;
  bool shouldStop_{false};
  Stats stats_;
  float current_temperature_{0.8f};
};

} // namespace llm
} // namespace extension
} // namespace executorch