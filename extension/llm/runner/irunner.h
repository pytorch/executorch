/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// An interface for LLM runners. Developers can create their own runner that
// implements their own load and generation logic to run the model.

#pragma once

#include <functional>
#include <string>

#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/module/module.h>

namespace executorch {
namespace extension {
namespace llm {

class ET_EXPERIMENTAL IRunner {
 public:
  virtual ~IRunner() = default;

  // Checks if the model is loaded.
  virtual bool is_loaded() const = 0;

  // Load the model and tokenizer.
  virtual ::executorch::runtime::Error load() = 0;

  // Generate the output tokens.
  virtual ::executorch::runtime::Error generate(
      const std::string& prompt,
      int32_t seq_len,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const ::executorch::extension::llm::Stats&)>
          stats_callback = {},
      bool echo = true,
      bool warming = false) = 0;

  // Stop the generation.
  virtual void stop() = 0;
};

} // namespace llm
} // namespace extension
} // namespace executorch
