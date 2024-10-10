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
// patternlint-disable-next-line executorch-cpp-nostdinc
#include <functional>
#include <memory>
// patternlint-disable-next-line executorch-cpp-nostdinc
#include <string>
#include <type_traits>
// patternlint-disable-next-line executorch-cpp-nostdinc
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

namespace executorch {
namespace extension {
namespace llm {

class ET_EXPERIMENTAL MultimodalRunner {
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
  }

  virtual bool is_loaded() = 0;
  virtual ::executorch::runtime::Error load() = 0;
  virtual ::executorch::runtime::Error generate(
      std::vector<Image> images,
      const std::string& prompt,
      int32_t seq_len = 1024,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const Stats&)> stats_callback = {},
      bool echo = true) = 0;

  /**
   * Prefill an LLaVA Module with the given images input.
   * @param images The image input to LLaVA.
   * @param start_pos The starting position in KV cache of the input in the LLM.
   * It's passed as reference and will be updated inside this function.
   * @return The error status of prefilling images.
   */
  virtual runtime::Error prefill_images(
      std::vector<Image>& images,
      int64_t& start_pos) = 0;

  /**
   * Prefill an LLaVA Module with the given text input.
   * @param prompt The text prompt to LLaVA.
   * @param start_pos The starting position in KV cache of the input in the LLM.
   * It's passed as reference and will be updated inside this function.
   * @param bos The number of BOS (begin of sequence) token.
   * @param eos The number of EOS (end of sequence) token.
   * @return The generated token of the LLaVA Module after prefill prompt.
   */
  virtual runtime::Result<uint64_t> prefill_prompt(
      const std::string& prompt,
      int64_t& start_pos,
      int8_t bos = 0,
      int8_t eos = 0) = 0;

  /**
   * Generate tokens from the given prompt, starting from the given position.
   * @param prompt The text prompt to LLaVA.
   * @param seq_len The total sequence length, including the prompt tokens and
   * new tokens.
   * @param start_pos The starting position in KV cache of the input in the LLM.
   * @param token_callback What to do after a token is generated.
   * @param stats_callback What to do with Stats.
   * @param echo Whether to echo the input prompt or not.
   * @return The error code.
   */
  virtual runtime::Error generate_from_pos(
      const std::string& prompt,
      int32_t seq_len = 1024,
      int64_t start_pos = 0,
      std::function<void(const std::string&)> token_callback = {},
      std::function<void(const ::executorch::extension::llm::Stats&)>
          stats_callback = {},
      bool echo = true) = 0;

  inline void stop() {
    text_token_generator_->stop();
  }

  virtual ~MultimodalRunner() = default;

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

} // namespace llm
} // namespace extension
} // namespace executorch

namespace torch {
namespace executor {
// TODO(T197294990): Remove these deprecated aliases once all users have moved
// to the new `::executorch` namespaces.
using ::executorch::extension::llm::MultimodalRunner;
} // namespace executor
} // namespace torch
