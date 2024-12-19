/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// A simple multimodal LLM runner that includes preprocessing and post
// processing logic.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>

#include <executorch/extension/llm/runner/multimodal_runner.h>

namespace example {

class ET_EXPERIMENTAL LlavaRunner
    : public ::executorch::extension::llm::MultimodalRunner {
 public:
  explicit LlavaRunner(
      const std::string& model_path,
      const std::string& tokenizer_path,
      const float temperature = 0.8f)
      : MultimodalRunner(model_path, tokenizer_path, temperature){};

  bool is_loaded() override;

  ::executorch::runtime::Error load() override;

  ::executorch::runtime::Error generate(
      std::vector<::executorch::extension::llm::Image> images,
      const std::string& prompt,
      int32_t seq_len = 1024,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const ::executorch::extension::llm::Stats&)>
          stats_callback = {},
      bool echo = true) override;

  ::executorch::runtime::Error prefill_images(
      std::vector<::executorch::extension::llm::Image>& images,
      int64_t& start_pos) override;

  ::executorch::runtime::Result<uint64_t> prefill_prompt(
      const std::string& prompt,
      int64_t& start_pos,
      int8_t bos = 0,
      int8_t eos = 0) override;

  ::executorch::runtime::Error generate_from_pos(
      const std::string& prompt,
      int32_t seq_len = 1024,
      int64_t start_pos = 0,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const ::executorch::extension::llm::Stats&)>
          stats_callback = {},
      bool echo = true) override;

 private:
  inline static const std::string kPresetPrompt =
      "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: ";
};

} // namespace example
