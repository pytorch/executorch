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

#include <executorch/examples/models/llava/runner/llava_image_prefiller.h>
#include <executorch/extension/llm/runner/image.h>
#include <executorch/extension/llm/runner/io_manager/io_manager.h>
#include <executorch/extension/llm/runner/irunner.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/text_decoder_runner.h>
#include <executorch/extension/llm/runner/text_prefiller.h>
#include <executorch/extension/llm/runner/text_token_generator.h>
#include <executorch/extension/module/module.h>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace example {

using executorch::extension::Module;
using executorch::extension::llm::ImagePrefiller;
using executorch::extension::llm::IOManager;
using executorch::extension::llm::Stats;
using executorch::extension::llm::TextDecoderRunner;
using executorch::extension::llm::TextPrefiller;
using executorch::extension::llm::TextTokenGenerator;

class ET_EXPERIMENTAL LlavaRunner {
 public:
  explicit LlavaRunner(
      const std::string& model_path,
      const std::string& tokenizer_path,
      const float temperature = 0.8f)
      : temperature_(temperature),
        module_(std::make_unique<Module>(model_path, Module::LoadMode::File)),
        io_manager_(std::make_unique<IOManager>(*module_)),
        tokenizer_path_(tokenizer_path) {
    ET_LOG(
        Info,
        "Creating Llava runner: model_path=%s, tokenizer_path=%s",
        model_path.c_str(),
        tokenizer_path.c_str());
  }

  bool is_loaded();

  ::executorch::runtime::Error load();

  ::executorch::runtime::Error generate(
      std::vector<::executorch::extension::llm::Image> images,
      const std::string& prompt,
      int32_t seq_len = 1024,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const ::executorch::extension::llm::Stats&)>
          stats_callback = {},
      bool echo = true);

  ::executorch::runtime::Error prefill_images(
      std::vector<::executorch::extension::llm::Image>& images,
      int64_t& start_pos);

  ::executorch::runtime::Result<uint64_t> prefill_prompt(
      const std::string& prompt,
      int64_t& start_pos,
      int8_t bos = 0,
      int8_t eos = 0);

  ::executorch::runtime::Error generate_from_pos(
      const std::string& prompt,
      int32_t seq_len = 1024,
      int64_t start_pos = 0,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const ::executorch::extension::llm::Stats&)>
          stats_callback = {},
      bool echo = true);

  inline void stop() {
    text_token_generator_->stop();
  }

 private:
  // metadata
  float temperature_;

  // model
  std::unordered_set<std::string> model_methods_;
  std::unique_ptr<Module> module_;
  std::unique_ptr<TextDecoderRunner> text_decoder_runner_;
  std::unique_ptr<TextPrefiller> text_prefiller_;
  std::unique_ptr<LlavaImagePrefiller> image_prefiller_;
  std::unique_ptr<IOManager> io_manager_;
  std::unique_ptr<TextTokenGenerator> text_token_generator_;
  std::string tokenizer_path_;
  std::unique_ptr<::tokenizers::Tokenizer> tokenizer_;

  // stats
  Stats stats_;

  inline static const char* kPresetPrompt =
      "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: ";
};

} // namespace example
