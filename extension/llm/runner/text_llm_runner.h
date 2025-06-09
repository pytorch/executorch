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

class ET_EXPERIMENTAL TextLLMRunner : public IRunner {
 public:
  /**
   * @brief Constructor for TextLLMRunner with dependency injection
   *
   * Creates a TextLLMRunner instance with all required components for text
   * generation.
   *
   * @param metadata Key-value pairs containing model metadata (e.g.,
   * vocab_size, context_length)
   * @param tokenizer Tokenizer for converting between text and token IDs
   * @param module The underlying model module that performs inference
   * @param text_decoder_runner Component responsible for running the decoder
   * part of the model
   * @param text_prefiller Component for handling the prefill phase of text
   * generation
   * @param text_token_generator Component for generating tokens during the
   * decode phase
   * @param stats Statistics tracking object for performance monitoring
   * @param temperature Temperature parameter for controlling randomness in
   * generation (deprecated). Please use GenerationConfig.temperature instead.
   */
  explicit TextLLMRunner(
      std::unordered_map<std::string, int64_t> metadata,
      std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
      std::unique_ptr<::executorch::extension::Module> module,
      std::unique_ptr<TextDecoderRunner> text_decoder_runner,
      std::unique_ptr<TextPrefiller> text_prefiller,
      std::unique_ptr<TextTokenGenerator> text_token_generator,
      std::unique_ptr<Stats> stats,
      float temperature = -1.0f);

  /**
   * @brief Checks if the model is loaded and ready for inference
   *
   * @return bool True if the model is loaded, false otherwise
   */
  bool is_loaded() const override;
  /**
   * @brief Loads the model and prepares it for inference
   *
   * This method initializes all components and prepares the model for text
   * generation.
   *
   * @return ::executorch::runtime::Error Success or error status
   */
  ::executorch::runtime::Error load() override;
  /**
   * @brief Generates text based on the provided prompt
   *
   * This method performs text generation using the loaded model. It processes
   * the input prompt, runs the model in prefill and decode phases, and returns
   * generated text through callbacks.
   *
   * @param prompt The input text to generate from
   * @param config Configuration parameters for text generation (e.g.,
   * max_new_tokens, temperature)
   * @param token_callback Function called for each generated token with the
   * decoded text
   * @param stats_callback Function called with performance statistics
   * @return ::executorch::runtime::Error Success or error status
   */
  ::executorch::runtime::Error generate(
      const std::string& prompt,
      const GenerationConfig& config,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const Stats&)> stats_callback = {}) override;
  /**
   * @brief Warms up the model with a sample prompt
   *
   * This method runs a complete generation cycle without returning results,
   * which helps initialize internal caches and optimize subsequent inferences.
   *
   * @param prompt The sample prompt to use for warmup
   * @param max_new_tokens Maximum number of tokens to generate during warmup
   * @return ::executorch::runtime::Error Success or error status
   */
  ::executorch::runtime::Error warmup(
      const std::string& prompt,
      int32_t max_new_tokens);
  /**
   * @brief Stops the ongoing text generation process
   *
   * This method signals the generator to stop producing new tokens and
   * terminate the current generation process.
   */
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

/**
 * @brief Loads a tokenizer from the specified path
 *
 * This function creates and initializes a tokenizer from a file, with options
 * to customize special tokens and regex patterns.
 *
 * @param tokenizer_path Path to the tokenizer file
 * @param special_tokens Optional list of special tokens to add to the tokenizer
 * @param pattern Optional regex pattern for tokenization
 * @param bos_token_index Index of the beginning-of-sequence token
 * @param eos_token_index Index of the end-of-sequence token
 * @return std::unique_ptr<tokenizers::Tokenizer> Initialized tokenizer instance
 */
ET_EXPERIMENTAL std::unique_ptr<tokenizers::Tokenizer> load_tokenizer(
    const std::string& tokenizer_path,
    std::unique_ptr<std::vector<std::string>> special_tokens = nullptr,
    std::optional<std::string> pattern = std::nullopt,
    size_t bos_token_index = 0,
    size_t eos_token_index = 1);

/**
 * @brief Creates a TextLLMRunner instance with the specified model and
 * tokenizer
 *
 * This factory function creates and initializes a TextLLMRunner with all
 * necessary components for text generation using the specified model and
 * tokenizer.
 *
 * @param model_path Path to the model file
 * @param tokenizer Initialized tokenizer instance
 * @param data_path Optional path to additional data required by the model
 * @param temperature Optional temperature parameter for controlling randomness
 * (deprecated)
 * @return std::unique_ptr<TextLLMRunner> Initialized TextLLMRunner instance
 */
ET_EXPERIMENTAL std::unique_ptr<TextLLMRunner> create_text_llm_runner(
    const std::string& model_path,
    std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
    std::optional<const std::string> data_path = std::nullopt,
    float temperature = -1.0f);

} // namespace executorch::extension::llm
