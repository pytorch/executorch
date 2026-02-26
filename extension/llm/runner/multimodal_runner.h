/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple multimodal LLM runner that includes preprocessing and post
// processing logic. The module takes in a string as input and emits a string as
// output.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include <executorch/extension/llm/runner/image.h>
#include <executorch/extension/llm/runner/image_prefiller.h>
#include <executorch/extension/llm/runner/io_manager/io_manager.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/llm/runner/multimodal_decoder_runner.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
#include <executorch/extension/llm/runner/multimodal_prefiller.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/text_decoder_runner.h>
#include <executorch/extension/llm/runner/text_prefiller.h>
#include <executorch/extension/llm/runner/text_token_generator.h>
#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/extension/module/module.h>
#include <pytorch/tokenizers/tokenizer.h>
// Helper functions are now in llm_runner_helper.h
// These are provided for backward compatibility
#include <executorch/extension/llm/runner/llm_runner_helper.h>

#ifdef CUDA_AVAILABLE
#include <executorch/backends/cuda/runtime/memory_tracker.h>
#endif

namespace executorch {
namespace extension {
namespace llm {

/**
 * MultimodalRunner - A runner for multimodal input and text output LLMs
 *
 * This class is designed for Large Language Models that can process multimodal
 * inputs (text, images, audio) and generate text outputs. It supports models
 * like LLaVA, CLIP-based vision-language models, and speech-to-text models.
 *
 * Supported Model Architecture see README.md
 *
 * Key Features:
 * - Supports mixed multimodal inputs in any order via
 * std::vector<MultimodalInput>
 * - Encoder handles non-text modalities (images, audio) → embeddings
 * - Text tokenizer converts text tokens → embeddings
 * - Embeddings are stitched together based on input ordering
 * - Text decoder performs autoregressive generation with KV cache
 * - Internal pos_ state tracks KV cache position across calls
 * - GenerationConfig provides comprehensive control over generation parameters
 *
 * Usage:
 *   std::vector<MultimodalInput> inputs;
 *   inputs.emplace_back(make_text_input("Describe this image:"));
 *   inputs.emplace_back(make_image_input(std::move(image)));
 *
 *   GenerationConfig config;
 *   config.max_new_tokens = 100;
 *   config.temperature = 0.7f;
 *
 *   runner->generate(inputs, config, token_callback, stats_callback);
 */
class ET_EXPERIMENTAL MultimodalRunner : public IRunner {
 public:
  /**
   * @brief Constructor for MultimodalRunner with dependency injection
   *
   * Creates a MultimodalRunner instance with all required components for
   * multimodal text generation. Note that we don't directly call into
   * `module` or `text_decoder_runner`, we take them to manage their lifecycles.
   *
   * @param metadata Key-value pairs containing model metadata (e.g.,
   * vocab_size, context_length)
   * @param tokenizer Tokenizer for converting between text and token IDs
   * @param module The underlying model module that performs inference
   * @param text_decoder_runner Component responsible for running the decoder
   * part of the model
   * @param multimodal_prefiller Component for prefilling multimodal inputs
   * @param io_manager Component for handling I/O operations
   * @param text_token_generator Component for generating tokens during the
   * @param stats Statistics tracking object for performance monitoring
   * decode phase
   */
  explicit MultimodalRunner(
      std::unordered_map<std::string, int64_t> metadata,
      std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
      std::unique_ptr<Module> module,
      std::unique_ptr<MultimodalDecoderRunner> text_decoder_runner,
      std::unique_ptr<MultimodalPrefiller> multimodal_prefiller,
      std::unique_ptr<IOManager> io_manager,
      std::unique_ptr<TextTokenGenerator> text_token_generator,
      std::unique_ptr<Stats> stats);

  bool is_loaded() const override;
  ::executorch::runtime::Error load() override;

  /**
   * Generate tokens from a text prompt. Wraps the prompt as a MultimodalInput
   * and delegates to generate(vector). Empty prompt is allowed if prefill()
   * was called beforehand.
   * @param prompt The text prompt to generate from.
   * @param config Generation configuration parameters.
   * @param token_callback Callback function called for each generated token.
   * @param stats_callback Callback function for generation statistics.
   * @return The error code. KV cache position is tracked internally in pos_.
   */
  ::executorch::runtime::Error generate(
      const std::string& prompt,
      const GenerationConfig& config,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const Stats&)> stats_callback = {}) override;

  /**
   * Generate tokens from the given multimodal inputs using GenerationConfig.
   * @param inputs A vector of MultimodalInput objects containing images and
   * text.
   * @param config Generation configuration parameters.
   * @param token_callback Callback function called for each generated token.
   * @param stats_callback Callback function for generation statistics.
   * @return The error code. KV cache position is tracked internally in pos_.
   */
  virtual ::executorch::runtime::Error generate(
      const std::vector<MultimodalInput>& inputs,
      const GenerationConfig& config,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const Stats&)> stats_callback = {});

  /**
   * Prefill multimodal inputs to fill the KV cache, for example to reload
   * chat history. Call generate() with a non-empty prompt afterwards to
   * start decoding.
   * @param inputs A vector of MultimodalInput objects containing images and
   * text.
   * @param num_bos Number of BOS tokens to prepend during encoding.
   * @param num_eos Number of EOS tokens to append during encoding.
   * @return The next token predicted after prefill, or an error.
   *         KV cache position is tracked internally in pos_.
   */
  ::executorch::runtime::Result<uint64_t> prefill(
      const std::vector<MultimodalInput>& inputs,
      int32_t num_bos = 0,
      int32_t num_eos = 0) override;

  /**
   * Convenience overload: prefill a single text prompt.
   */
  ::executorch::runtime::Result<uint64_t>
  prefill(const std::string& prompt, int32_t num_bos = 0, int32_t num_eos = 0) {
    std::vector<MultimodalInput> inputs;
    inputs.emplace_back(MultimodalInput(prompt));
    return prefill(inputs, num_bos, num_eos);
  }

  void stop() override {
    text_token_generator_->stop();
  }

  void reset() override {
    pos_ = 0;
    stats_->reset();
    prefill_next_token_.reset();
  }

  ~MultimodalRunner() override = default;

 protected:
  // Components
  std::unordered_map<std::string, int64_t> metadata_;
  std::unique_ptr<::tokenizers::Tokenizer> tokenizer_;
  std::unique_ptr<Module> module_;
  std::unique_ptr<MultimodalDecoderRunner> text_decoder_runner_;
  std::unique_ptr<MultimodalPrefiller> multimodal_prefiller_;
  std::unique_ptr<IOManager> io_manager_;
  std::unique_ptr<TextTokenGenerator> text_token_generator_;
  std::unique_ptr<Stats> stats_;

#ifdef CUDA_AVAILABLE
  std::unique_ptr<::executorch::backends::cuda::CudaMemoryTracker>
      cuda_memory_tracker_;
#endif

  // Internal state
  std::optional<uint64_t> prefill_next_token_;
  int64_t pos_;

 private:
  ::executorch::runtime::Error decode_from_token(
      uint64_t cur_token,
      const GenerationConfig& config,
      std::function<void(const std::string&)> wrapped_callback,
      std::function<void(const Stats&)> stats_callback);
};

} // namespace llm
} // namespace extension
} // namespace executorch
