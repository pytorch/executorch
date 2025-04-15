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

// Base interface for LLM runners
class ET_API IRunner {
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
   * Stop the generation process.
   */
  virtual void stop() = 0;
};

} // namespace llm
} // namespace extension
} // namespace executorch
