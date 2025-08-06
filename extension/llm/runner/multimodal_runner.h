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
#include <string>
#include <unordered_map>

#include <executorch/extension/llm/runner/image.h>
#include <executorch/extension/llm/runner/image_prefiller.h>
#include <executorch/extension/llm/runner/io_manager/io_manager.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/llm/runner/multimodal_input.h>
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
 * Supported Model Architecture:
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │                        Multimodal LLM Architecture                      │
 * └─────────────────────────────────────────────────────────────────────────┘
 *
 *    Input: std::vector<MultimodalInput>
 *           ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
 *           │     Image       │  │     Audio       │  │      Text       │
 *           │    [224x        │  │    [16kHz       │  │     "What"      │
 *           │     224x3]      │  │     audio]      │  │                 │
 *           └─────────────────┘  └─────────────────┘  └─────────────────┘
 *                    │                    │                    │
 *                    ▼                    ▼                    ▼
 *           ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ ◄─┐
 *           │     Encoder     │  │     Encoder     │  │ Text Tokenizer  │   │
 *           │   (Vision)      │  │   (Audio)       │  │   & Embedding   │   │
 *           │                 │  │                 │  │                 │   │
 *           │ pixels → embed  │  │ waveform→embed  │  │ tokens → embed  │   │
 *           └─────────────────┘  └─────────────────┘  └─────────────────┘   │
 *                    │                    │                    │            │
 *                    ▼                    ▼                    ▼            │
 *           ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
 *           │     [D_emb]     │  │     [D_emb]     │  │     [D_emb]     │   │
 *           │    Embedding    │  │    Embedding    │  │    Embedding    │   │
 *           └─────────────────┘  └─────────────────┘  └─────────────────┘   │
 *                    │                    │                    │            │
 *                    └────────────────────┼────────────────────┘            │
 *                                         │                                 │
 *                                         ▼                                 │
 *                    ┌─────────────────────────────┐                        │
 *                    │      Text Decoder Block     │                        │
 *                    │    (Transformer Layers)     │                        │
 *                    │                             │                        │
 *                    │  ┌─────────────────────┐    │                        │
 *                    │  │   Self-Attention    │    │                        │
 *                    │  │   + Feed Forward    │    │                        │
 *                    │  │   (with KV Cache)   │    │                        │
 *                    │  └─────────────────────┘    │                        │
 *                    │           │                 │                        │
 *                    │           ▼                 │                        │
 *                    │    Token Generation         │                        │
 *                    │    (pos_ tracking)          │                        │
 *                    └─────────────────────────────┘                        │
 *                                   │───────────────────────────────────────┘
 *                                   │          (Autoregressive)
 *                                   ▼
 *                          ┌─────────────────┐
 *                          │  Generated Text │
 *                          │ "This image     │
 *                          │  shows a cat    │
 *                          │  sitting..."    │
 *                          └─────────────────┘
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
class ET_EXPERIMENTAL MultimodalRunner {
 public:
  /**
   * @brief Constructor for MultimodalRunner with dependency injection
   *
   * Creates a MultimodalRunner instance with all required components for
   * multimodal text generation.
   *
   * @param metadata Key-value pairs containing model metadata (e.g.,
   * vocab_size, context_length)
   * @param tokenizer Tokenizer for converting between text and token IDs
   * @param module The underlying model module that performs inference
   * @param text_decoder_runner Component responsible for running the decoder
   * part of the model
   * @param text_prefiller Component for handling the prefill phase of text
   * generation
   * @param image_prefiller Component for handling image prefill
   * @param text_token_generator Component for generating tokens during the
   * @param stats Statistics tracking object for performance monitoring
   * decode phase
   */
  explicit MultimodalRunner(
      std::unordered_map<std::string, int64_t> metadata,
      std::unique_ptr<::tokenizers::Tokenizer> tokenizer,
      std::unique_ptr<Module> module,
      std::unique_ptr<TextDecoderRunner> text_decoder_runner,
      std::unique_ptr<TextPrefiller> text_prefiller,
      std::unique_ptr<ImagePrefiller> image_prefiller,
      std::unique_ptr<TextTokenGenerator> text_token_generator,
      std::unique_ptr<Stats> stats);

  virtual bool is_loaded();
  virtual ::executorch::runtime::Error load();

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
      std::function<void(const std::string&)>& token_callback,
      std::function<void(const Stats&)>& stats_callback);

  /**
   * Prefill a text decoder module with the given text input.
   * @param input.
   * @param bos The number of BOS (begin of sequence) token.
   * @param eos The number of EOS (end of sequence) token.
   * @return The generated token after prefilling the input.
   */
  virtual runtime::Result<uint64_t>
  prefill_input(const MultimodalInput& input, int8_t bos = 0, int8_t eos = 0);

  inline void stop() {
    text_token_generator_->stop();
  }

  inline void reset() {
    pos_ = 0;
    stats_->reset();
  }

  virtual ~MultimodalRunner() = default;

 protected:
  // Components
  std::unordered_map<std::string, int64_t> metadata_;
  std::unique_ptr<::tokenizers::Tokenizer> tokenizer_;
  std::unique_ptr<Module> module_;
  std::unique_ptr<TextDecoderRunner> text_decoder_runner_;
  std::unique_ptr<TextPrefiller> text_prefiller_;
  std::unique_ptr<ImagePrefiller> image_prefiller_;
  std::unique_ptr<IOManager> io_manager_;
  std::unique_ptr<TextTokenGenerator> text_token_generator_;
  std::unique_ptr<Stats> stats_;

  // Internal state
  int64_t pos_;
};

} // namespace llm
} // namespace extension
} // namespace executorch
