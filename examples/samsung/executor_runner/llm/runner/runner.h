/*
 * Copyright (c) 2025 Samsung Electronics Co. LTD
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

#include <executorch/examples/samsung/executor_runner/llm/runner/constants.h>
#include <executorch/examples/samsung/executor_runner/llm/runner/prefiller.h>
#include <executorch/examples/samsung/executor_runner/llm/runner/token_generator.h>

namespace example {

namespace llm = ::executorch::extension::llm;
namespace runtime = ::executorch::runtime;
namespace extension = ::executorch::extension;

/**
 * @brief Holds path-related configuration for creating an LLM runner.
 */
struct RunnerPaths {
  std::string model_path;
  std::string tokenizer_path;
  std::optional<std::string> pos_embedding_path = std::nullopt;
  std::optional<std::string> data_path = std::nullopt;
};

/**
 * @brief Holds non-path configuration for creating an LLM runner.
 */
struct RunnerConfig {
  std::string config_path;
  bool is_decode_only = false;
  float temperature = -1.0f;

  // Strategies
  DecoderModelVersion model_version = DecoderModelVersion::kLlama3;
  EmbeddingStrategy embedding_strategy = EmbeddingStrategy::kGlobalOnly;
  MaskingStrategy masking_strategy = MaskingStrategy::kCausal;

  // Parsed from JSON
  int64_t hidden_dim = 0;
  int64_t n_heads = 0;
  int64_t n_kv_heads = 0;
  int64_t n_layers = 0;
  int64_t head_dim = 0;
  int64_t vocab_size = 0;
  int64_t max_seq_len = 128;
  int64_t max_context_len = 128;
  int64_t prefill_seq_len = 128;
  int64_t decode_seq_len = 1;
  int64_t attn_mask_rows = 128;
  int64_t attn_mask_cols = 1024;
  std::optional<int64_t> sliding_window;
  std::string tokenizer_type;

  void load_from_json();
};

class LLMRunner : public llm::IRunner {
 public:
  /**
   * @brief Constructor for LLMRunner with dependency injection
   *
   * Creates a LLMRunner instance with all required components for text
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
   * @param io_manager IOManager manages input/output operations for LLM
   * inference.
   * @param model_version Decoder model family selecting the prompt template
   * @param temperature Temperature parameter for controlling randomness in
   * generation (deprecated). Please use GenerationConfig.temperature instead.
   */
  explicit LLMRunner(
      std::unordered_map<std::string, int64_t> metadata,
      std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
      std::unique_ptr<extension::Module> module,
      std::unique_ptr<llm::TextDecoderRunner> text_decoder_runner,
      std::unique_ptr<llm::TextPrefiller> text_prefiller,
      std::unique_ptr<example::TokenGenerator> text_token_generator,
      std::unique_ptr<llm::Stats> stats,
      std::unique_ptr<llm::IOManager> io_manager,
      DecoderModelVersion model_version,
      float temperature = -1.0f);

  /**
   * @brief Checks if the model is loaded and ready for inference
   *
   * @return bool True if the model is loaded, false otherwise
   */
  bool is_loaded() const;
  /**
   * @brief Loads the model and prepares it for inference
   *
   * This method initializes all components and prepares the model for text
   * generation.
   *
   * @return ::executorch::runtime::Error Success or error status
   */
  runtime::Error load();
  /**
   * @brief Generates text based on the provided prompt
   *
   * This method performs text generation using the loaded model. It processes
   * the input prompt, runs the model in prefill and decode phases until max
   * tokens to generate is reached or eos token is generated, then returns
   * generated text and perf stats through callbacks.
   *
   * @param prompt The input text to generate from
   * @param config Configuration parameters for text generation (e.g.,
   * max_new_tokens, temperature)
   * @param token_callback Function called for each generated token with the
   * decoded text
   * @param stats_callback Function called with performance statistics
   * @return ::executorch::runtime::Error Success or error status
   */
  runtime::Error generate(
      const std::string& prompt,
      const llm::GenerationConfig& config,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const llm::Stats&)> stats_callback = {});

  /**
   * @brief Generates text based on the provided prompt and start position
   *
   * This method performs text generation using the loaded model. It processes
   * the input prompt, runs the model in prefill and decode phases using the
   * start position until max tokens to generate is reached or eos token is
   * generated, then returns generated text and perf stats through callbacks.
   *
   * @param prompt The input text to generate from
   * @param start_pos The starting position in KV cache of the input
   * @param config Configuration parameters for text generation (e.g.,
   * max_new_tokens, temperature)
   * @param token_callback Function called for each generated token with the
   * decoded text
   * @param stats_callback Function called with performance statistics
   * @return ::executorch::runtime::Error Success or error status
   */
  runtime::Error generate_from_pos(
      const std::string& prompt,
      int64_t start_pos,
      const llm::GenerationConfig& config,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const llm::Stats&)> stats_callback = {});

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
  runtime::Error warmup(const std::string& prompt, int32_t max_new_tokens);
  /**
   * @brief Stops the ongoing text generation process
   *
   * This method signals the generator to stop producing new tokens and
   * terminate the current generation process.
   */
  void stop();

  /**
   * @brief Formats the prompt based on the model version.
   *
   * @param prompt The input text
   * @param system_prompt The system prompt (optional)
   * @param version The version of the decoder model
   * @return Formatted prompt string
   */
  static std::string get_formatted_prompt(
      const std::string& prompt,
      const std::string& system_prompt,
      DecoderModelVersion version);

  /**
   * @brief Force remove prefilled tokens and reset KV cache start position
   *
   * This mehtod removes the prefilled tokens from the KV cache and resets the
   * start position to 0.
   */
  void reset() {};

 private:
  bool shouldStop_{false};

  // Components
  std::unique_ptr<::tokenizers::Tokenizer> tokenizer_;
  std::unordered_map<std::string, int64_t> metadata_;
  std::unique_ptr<extension::Module>
      module_; // Manage module's lifecycle, make sure it outlives
               // text_decoder_runner_llm::.
  std::unique_ptr<llm::TextDecoderRunner>
      text_decoder_runner_; // Manage text_decoder_runner_'s lifecycle, make
                            // sure it outlives text_prefiller_ &
                            // text_token_generator_.
  std::unique_ptr<llm::TextPrefiller> text_prefiller_;
  std::unique_ptr<example::TokenGenerator> text_token_generator_;
  std::unique_ptr<llm::IOManager> io_manager_;

  // Stats
  std::unique_ptr<llm::Stats> stats_;

  // temperature.
  // Deprecated, we should rely on the temperature in GenerationConfig instead.
  float temperature_ = -1.0f;
  DecoderModelVersion model_version_;
};

std::unique_ptr<llm::IRunner> create_llm_runner(
    const RunnerPaths& paths,
    RunnerConfig config);

} // namespace example
