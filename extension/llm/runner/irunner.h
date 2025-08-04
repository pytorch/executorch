/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Interface for text generation runners.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include <executorch/extension/llm/runner/stats.h>
#include <executorch/runtime/core/error.h>

namespace executorch {
namespace extension {
namespace llm {

// Configuration struct for generation parameters, fields should be sorted in
// alphabetic order
struct GenerationConfig {
  // Whether to echo the input prompt in the output
  bool echo = true;

  // Maximum number of new tokens to generate
  // If the max_context_len metadata that's serialized in the .pte file exists,
  // then the number of prompt tokens + max_new_tokens won't exceed
  // max_context_len. If this field is -1, it means we will rely on
  // max_context_len metadata and seq_len value. Check resolve_max_new_tokens
  // for details.
  int32_t max_new_tokens = -1;

  // Whether this is a warmup run (affects perf benchmarking)
  bool warming = false;

  // Maximum number of total tokens
  // If the .pte file contains the max_context_len metadata, it will override
  // this value if it's smaller. If this field is -1, we will use the
  // max_context_len metadata directly. Check resolve_max_new_tokens for
  // details.
  int32_t seq_len = -1;

  // Temperature for sampling (higher = more random)
  float temperature = 0.8f;

  // Number of eos and bos to add to the prompt
  int32_t num_bos = 0;
  int32_t num_eos = 0;

  /**
   * Resolve the maximum number of new tokens to generate based on constraints.
   *
   * This method calculates the maximum number of new tokens that can be
   * generated considering both seq_len and max_new_tokens constraints, as well
   * as the model's maximum context length and the number of tokens in the
   * prompt.
   *
   * @param max_context_len The maximum context length supported by the model
   * @param num_prompt_tokens The number of tokens in the input prompt
   * @return The resolved maximum number of new tokens to generate
   */
  int32_t resolve_max_new_tokens(
      int32_t max_context_len,
      int32_t num_prompt_tokens) const {
    int32_t result;

    if (seq_len == -1 && max_new_tokens == -1) {
      // Both are -1, use max context len minus prompt tokens
      result = max_context_len - num_prompt_tokens;
    } else if (seq_len == -1 && max_new_tokens != -1) {
      // Only max_new_tokens is specified
      result = std::min(max_new_tokens, max_context_len - num_prompt_tokens);
    } else if (seq_len != -1 && max_new_tokens == -1) {
      // Only seq_len is specified
      result = std::min(seq_len, max_context_len) - num_prompt_tokens;
    } else {
      // Both are specified
      result = std::min(
          std::min(seq_len, max_context_len) - num_prompt_tokens,
          max_new_tokens);
    }

    // Ensure result is not negative
    return std::max(0, result);
  }
};

// Base interface for LLM runners
class ET_EXPERIMENTAL IRunner {
 public:
  virtual ~IRunner() = default;

  /**
   * Check if the runner is loaded and ready for inference.
   *
   * @return true if the runner is loaded, false otherwise
   */
  virtual bool is_loaded() const = 0;

  /**
   * Load the model and prepare for inference.
   *
   * @return Error::Ok if successful, an error otherwise
   */
  virtual runtime::Error load() = 0;

  /**
   * Generate text based on the provided prompt and generation config.
   *
   * @param prompt The input prompt to generate from
   * @param config Generation configuration parameters
   * @param token_callback Callback function called for each generated token
   * @param stats_callback Callback function for generation statistics
   * @return Error::Ok if successful, an error otherwise
   */
  virtual runtime::Error generate(
      const std::string& prompt,
      const GenerationConfig& config,
      std::function<void(const std::string&)> token_callback,
      std::function<void(const Stats&)> stats_callback) = 0;

  /**
   * Generate text based on the provided prompt and generation config, from a
   * given position in KV cache.
   *
   * @param prompt The input prompt to generate from
   * @param start_pos The starting position in KV cache of the input
   * @param config Generation configuration parameters
   * @param token_callback Callback function called for each generated token
   * @param stats_callback Callback function for generation statistics
   * @return Error::Ok if successful, an error otherwise
   */
  virtual runtime::Error generate_from_pos(
      const std::string& prompt,
      int64_t start_pos,
      const GenerationConfig& config,
      std::function<void(const std::string&)> token_callback,
      std::function<void(const Stats&)> stats_callback) = 0;
  /**
   * Stop the generation process.
   */
  virtual void stop() = 0;
};

} // namespace llm
} // namespace extension
} // namespace executorch
