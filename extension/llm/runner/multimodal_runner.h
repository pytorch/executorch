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
#include <type_traits>
#include <unordered_map>

#include <executorch/extension/llm/runner/image.h>
#include <executorch/extension/llm/runner/image_prefiller.h>
#include <executorch/extension/llm/runner/stats.h>
#include <executorch/extension/llm/runner/text_decoder_runner.h>
#include <executorch/extension/llm/runner/text_prefiller.h>
#include <executorch/extension/llm/runner/text_token_generator.h>
#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/extension/llm/tokenizer/tokenizer.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/runner_util/managed_tensor.h>

namespace torch::executor {
using Stats = ::executorch::llm::Stats;

class MultimodalRunner {
 public:
  explicit MultimodalRunner(
      const std::string& model_path,
      const std::string& tokenizer_path,
      const float temperature = 0.8f)
      : temperature_(temperature),
        module_(std::make_unique<Module>(model_path, Module::LoadMode::File)),
        tokenizer_path_(tokenizer_path) {
    ET_LOG(
        Info,
        "Creating Multimodal LLM runner: model_path=%s, tokenizer_path=%s",
        model_path.c_str(),
        tokenizer_path.c_str());
  };

  virtual bool is_loaded() = 0;
  virtual Error load() = 0;
  virtual Error generate(
      std::vector<Image>& images,
      const std::string& prompt,
      int32_t seq_len = 1024,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const Stats&)> stats_callback = {}) = 0;

  inline void stop() {
    text_token_generator_->stop();
  }

 protected:
  // metadata
  int32_t vocab_size_;
  int32_t bos_id_;
  int32_t eos_id_;
  int32_t n_bos_;
  int32_t n_eos_;
  int32_t max_seq_len_;
  float temperature_;

  // model
  std::unordered_set<std::string> model_methods_;
  std::unique_ptr<Module> module_;
  std::unique_ptr<TextDecoderRunner> text_decoder_runner_;
  std::unique_ptr<TextPrefiller> text_prefiller_;
  std::unique_ptr<ImagePrefiller> image_prefiller_;
  std::unique_ptr<TextTokenGenerator> text_token_generator_;
  std::string tokenizer_path_;
  std::unique_ptr<Tokenizer> tokenizer_;

  // stats
  Stats stats_;
};

} // namespace torch::executor